"""对话级路由回调 (SmartRouterCallback)

继承 BaseRouterCallback，实现对话级路由策略：
规则前置 → RouteLLM 打分 → 区间匹配 → 分发。

公共管道（合规检测、PII 脱敏、响应还原）由基类处理。
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Optional

from aegis_router.callbacks.base_router import BaseRouterCallback
from aegis_router.callbacks.degradation import (
    DegradationManager,
)
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.config import ClassifierConfig, RoutingConfig, TrivialConfig
from aegis_router.observability.audit_logger import AuditLogger
from aegis_router.router.config_watcher import ConfigWatcher
from aegis_router.router.model_classifier import ModelClassifier
from aegis_router.router.route_resolver import RouteResolver
from aegis_router.router.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

# Module-level pool instance — shared by all SmartRouterCallback instances
# within the same worker process.
_pool: ClawVaultPool = ClawVaultPool()


class SmartRouterCallback(BaseRouterCallback):
    """对话级路由插件 — 继承 BaseRouterCallback。

    路由策略：规则前置 → RouteLLM 打分 → 区间匹配 → 分发。
    公共管道（合规、PII 脱敏、还原）由基类 async_pre_call_hook 处理。
    """

    def __init__(
        self,
        pool: ClawVaultPool | None = None,
        config_dir: str | None = None,
        enable_routing: bool = True,
        rule_engine: RuleEngine | None = None,
        classifier: ModelClassifier | None = None,
        config_watcher: ConfigWatcher | None = None,
        degradation_manager: DegradationManager | None = None,
    ) -> None:
        super().__init__(
            pool=pool if pool is not None else _pool,
            degradation_manager=degradation_manager,
        )
        self._audit_logger = AuditLogger()

        # --- Routing components ---
        self._enable_routing = enable_routing
        self._rule_engine: Optional[RuleEngine] = rule_engine
        self._classifier: Optional[ModelClassifier] = classifier
        self._config_watcher: Optional[ConfigWatcher] = config_watcher
        self._route_resolver: Optional[RouteResolver] = None
        self._routing_config: Optional[RoutingConfig] = None

        if enable_routing and config_watcher is None and rule_engine is None:
            # Auto-initialize from config directory
            self._init_routing(config_dir or os.environ.get("AEGIS_CONFIG_DIR", "./config"))
        elif enable_routing and config_watcher is not None:
            # Use provided config_watcher — read config from it
            self._init_routing_from_watcher()

        logger.info(
            "SmartRouterCallback initialized (pool max_connections=%d, routing=%s)",
            self._pool.max_connections,
            "enabled" if self._enable_routing else "disabled",
        )

    # ------------------------------------------------------------------
    # Property Override
    # ------------------------------------------------------------------

    @property
    def routing_enabled(self) -> bool:
        """Whether routing is enabled for this plugin instance."""
        return self._enable_routing

    # ------------------------------------------------------------------
    # Routing Initialization
    # ------------------------------------------------------------------

    def _init_routing(self, config_dir: str) -> None:
        """Initialize routing components from config directory."""
        try:
            self._config_watcher = ConfigWatcher(
                config_dir=config_dir,
                on_routing_table_updated=self._on_routing_table_updated,
            )
            self._config_watcher.start()
            self._init_routing_from_watcher()
        except Exception as e:
            logger.error("Failed to initialize routing: %s. Routing disabled.", e)
            self._enable_routing = False

    def _init_routing_from_watcher(self) -> None:
        """Initialize routing components from an existing ConfigWatcher."""
        config = self._config_watcher.get_current_config()
        if config is None:
            logger.warning("ConfigWatcher has no config yet. Routing may not work until config is loaded.")
            return

        self._routing_config = config.routing

        # Initialize RuleEngine if not provided
        if self._rule_engine is None:
            self._rule_engine = RuleEngine(config.routing.trivial)

        # Initialize ModelClassifier if not provided
        if self._classifier is None:
            try:
                self._classifier = ModelClassifier(config.routing.classifier)
            except Exception as e:
                logger.warning("Failed to initialize ModelClassifier: %s. Will use fallback.", e)

        # Build RouteResolver from current routing table
        tiers = self._config_watcher.get_current_routing_table()
        self._route_resolver = RouteResolver(
            tiers=tiers,
            strategy=config.routing.overlap_strategy,
            fallback_model=config.routing.fallback_model,
            session_lock_ttl_minutes=config.routing.session_lock_ttl_minutes,
        )

    def _on_routing_table_updated(self, new_table: list) -> None:
        """Callback when ConfigWatcher detects config changes and rebuilds routing table."""
        config = self._config_watcher.get_current_config() if self._config_watcher else None
        if config is None:
            return

        previous_policy = (
            self._routing_config.session_policy if self._routing_config is not None else None
        )
        self._routing_config = config.routing

        # Keep active locks across table-only updates. A policy change starts a
        # fresh session-routing epoch so stale locks cannot be resurrected.
        if self._route_resolver is None:
            self._route_resolver = RouteResolver(
                tiers=new_table,
                strategy=config.routing.overlap_strategy,
                fallback_model=config.routing.fallback_model,
                session_lock_ttl_minutes=config.routing.session_lock_ttl_minutes,
            )
        else:
            self._route_resolver.update_configuration(
                tiers=new_table,
                strategy=config.routing.overlap_strategy,
                fallback_model=config.routing.fallback_model,
                session_lock_ttl_minutes=config.routing.session_lock_ttl_minutes,
                clear_session_locks=(
                    previous_policy is not None
                    and previous_policy != config.routing.session_policy
                ),
            )

        # Update RuleEngine config
        self._rule_engine = RuleEngine(config.routing.trivial)

        logger.info(
            "Routing table updated: %d tiers, strategy=%s",
            len(new_table),
            config.routing.overlap_strategy,
        )

    # ------------------------------------------------------------------
    # Routing Pipeline (implements abstract method from BaseRouterCallback)
    # ------------------------------------------------------------------

    async def _execute_routing(
        self,
        data: dict,
        masked_text: str,
        original_text: str,
        prompt_hash: str,
    ) -> None:
        """Execute the routing pipeline: rule engine → scorer → resolver → dispatch.

        Args:
            data: The LiteLLM request data dict (will be mutated to set target model).
            masked_text: The PII-masked prompt text.
            original_text: The original (unmasked) prompt text.
            prompt_hash: SHA-256 hash of the original prompt (for audit logging).
        """
        t_route_start = time.perf_counter()
        metadata = data.get("metadata", {})
        metadata["routing_plugin"] = "conversation"
        metadata["_routing_warnings"] = []

        def apply_session_policy(decision: dict[str, Any]) -> dict[str, Any]:
            """Apply FR-3.12 consistently to rule, fallback, and scored routes."""
            if self._route_resolver is None or self._routing_config is None:
                return decision
            return self._route_resolver.apply_session_policy(
                decision=decision,
                session_id=metadata.get("session_id"),
                session_policy=self._routing_config.session_policy,
            )

        # Determine which text to use for scoring
        score_input_mode = "masked"
        if self._routing_config:
            score_input_mode = self._routing_config.score_input

        scoring_text = masked_text if score_input_mode == "masked" else original_text

        # --- Step 1: Rule Engine (寒暄检测) ---
        if self._rule_engine is not None:
            rule_result = self._rule_engine.check(scoring_text)
            if rule_result.matched:
                independent_model = rule_result.target_model or "local-7b"
                route_result = apply_session_policy(
                    {
                        "model": independent_model,
                        "reason": "trivial_chat",
                        "candidates": [independent_model],
                    }
                )
                target_model = route_result["model"]
                route_reason = route_result["reason"]
                data["model"] = target_model

                latency_route_ms = (time.perf_counter() - t_route_start) * 1000
                metadata["latency_route_ms"] = latency_route_ms
                metadata["target_model"] = target_model
                metadata["route_reason"] = route_reason
                metadata["route_score"] = None
                metadata["route_matched_pattern"] = rule_result.matched_pattern

                logger.info(
                    "路由决策 (规则前置): prompt_hash=%s, target=%s, "
                    "pattern=%s, latency_route_ms=%.2f",
                    prompt_hash[:16],
                    target_model,
                    rule_result.matched_pattern,
                    latency_route_ms,
                )

                # Emit structured audit log for rule engine match
                self._audit_logger.log_route_decision(
                    request_id=metadata.get("request_id", ""),
                    session_id=metadata.get("session_id", ""),
                    prompt_hash=prompt_hash,
                    prompt_length=len(original_text),
                    route_score=None,
                    candidates=route_result.get("candidates", [target_model]),
                    target_model=target_model,
                    route_reason=route_reason,
                    latency_mask_ms=metadata.get("latency_mask_ms", 0.0),
                    latency_route_ms=latency_route_ms,
                    entities_detected=metadata.get("entities_detected"),
                )
                return

        # --- Step 2: RouteLLM Classifier (打分) ---
        score: Optional[float] = None
        fallback_model = "deepseek-v3"
        if self._routing_config:
            fallback_model = self._routing_config.fallback_model

        if self._classifier is not None:
            try:
                classifier_result = await self._classifier.aclassify(scoring_text)
                score = classifier_result.score
            except TimeoutError:
                # Classifier timeout → use fallback model
                logger.warning(
                    "Classifier timeout, using fallback model: %s (prompt_hash=%s)",
                    fallback_model,
                    prompt_hash[:16],
                )
                route_result = apply_session_policy(
                    {
                        "model": fallback_model,
                        "reason": "classifier_timeout",
                        "candidates": [fallback_model],
                    }
                )
                target_model = route_result["model"]
                route_reason = route_result["reason"]
                data["model"] = target_model

                latency_route_ms = (time.perf_counter() - t_route_start) * 1000
                metadata["latency_route_ms"] = latency_route_ms
                metadata["target_model"] = target_model
                metadata["route_reason"] = route_reason
                metadata["route_score"] = None

                # Emit structured audit log for classifier timeout
                self._audit_logger.log_route_decision(
                    request_id=metadata.get("request_id", ""),
                    session_id=metadata.get("session_id", ""),
                    prompt_hash=prompt_hash,
                    prompt_length=len(original_text),
                    route_score=None,
                    candidates=route_result.get("candidates", [target_model]),
                    target_model=target_model,
                    route_reason=route_reason,
                    latency_mask_ms=metadata.get("latency_mask_ms", 0.0),
                    latency_route_ms=latency_route_ms,
                    entities_detected=metadata.get("entities_detected"),
                )
                return
            except (RuntimeError, Exception) as e:
                # Classifier unavailable → use fallback model
                logger.warning(
                    "Classifier error (%s), using fallback model: %s (prompt_hash=%s)",
                    e,
                    fallback_model,
                    prompt_hash[:16],
                )
                route_result = apply_session_policy(
                    {
                        "model": fallback_model,
                        "reason": "classifier_error",
                        "candidates": [fallback_model],
                    }
                )
                target_model = route_result["model"]
                route_reason = route_result["reason"]
                data["model"] = target_model

                latency_route_ms = (time.perf_counter() - t_route_start) * 1000
                metadata["latency_route_ms"] = latency_route_ms
                metadata["target_model"] = target_model
                metadata["route_reason"] = route_reason
                metadata["route_score"] = None

                # Emit structured audit log for classifier error
                self._audit_logger.log_route_decision(
                    request_id=metadata.get("request_id", ""),
                    session_id=metadata.get("session_id", ""),
                    prompt_hash=prompt_hash,
                    prompt_length=len(original_text),
                    route_score=None,
                    candidates=route_result.get("candidates", [target_model]),
                    target_model=target_model,
                    route_reason=route_reason,
                    latency_mask_ms=metadata.get("latency_mask_ms", 0.0),
                    latency_route_ms=latency_route_ms,
                    entities_detected=metadata.get("entities_detected"),
                )
                return
        else:
            # No classifier available → use fallback
            logger.warning(
                "No classifier available, using fallback model: %s",
                fallback_model,
            )
            route_result = apply_session_policy(
                {
                    "model": fallback_model,
                    "reason": "no_classifier",
                    "candidates": [fallback_model],
                }
            )
            target_model = route_result["model"]
            route_reason = route_result["reason"]
            data["model"] = target_model

            latency_route_ms = (time.perf_counter() - t_route_start) * 1000
            metadata["latency_route_ms"] = latency_route_ms
            metadata["target_model"] = target_model
            metadata["route_reason"] = route_reason
            metadata["route_score"] = None

            # Emit structured audit log for no classifier
            self._audit_logger.log_route_decision(
                request_id=metadata.get("request_id", ""),
                session_id=metadata.get("session_id", ""),
                prompt_hash=prompt_hash,
                prompt_length=len(original_text),
                route_score=None,
                candidates=route_result.get("candidates", [target_model]),
                target_model=target_model,
                route_reason=route_reason,
                latency_mask_ms=metadata.get("latency_mask_ms", 0.0),
                latency_route_ms=latency_route_ms,
                entities_detected=metadata.get("entities_detected"),
            )
            return

        # --- Step 3: Route Resolver (区间匹配 + 重叠策略) ---
        if self._route_resolver is not None and score is not None:
            resolve_result = self._route_resolver.resolve(
                score,
                session_id=metadata.get("session_id"),
                session_policy=(
                    self._routing_config.session_policy
                    if self._routing_config is not None
                    else "sticky"
                ),
            )
            target_model = resolve_result["model"]
            reason = resolve_result["reason"]
            candidates = resolve_result.get("candidates", [])

            data["model"] = target_model

            latency_route_ms = (time.perf_counter() - t_route_start) * 1000
            metadata["latency_route_ms"] = latency_route_ms
            metadata["target_model"] = target_model
            metadata["route_score"] = score
            metadata["route_reason"] = reason
            metadata["route_candidates"] = candidates
            if "candidates_count" in resolve_result:
                metadata["route_candidates_count"] = resolve_result["candidates_count"]

            logger.info(
                "路由决策 (打分+匹配): prompt_hash=%s, score=%.4f, "
                "target=%s, reason=%s, latency_route_ms=%.2f",
                prompt_hash[:16],
                score,
                target_model,
                reason,
                latency_route_ms,
            )

            # Emit structured audit log
            self._audit_logger.log_route_decision(
                request_id=metadata.get("request_id", ""),
                session_id=metadata.get("session_id", ""),
                prompt_hash=prompt_hash,
                prompt_length=len(original_text),
                route_score=score,
                candidates=candidates,
                target_model=target_model,
                route_reason=reason,
                latency_mask_ms=metadata.get("latency_mask_ms", 0.0),
                latency_route_ms=latency_route_ms,
                entities_detected=metadata.get("entities_detected"),
            )
        else:
            # No resolver available → use fallback
            data["model"] = fallback_model

            latency_route_ms = (time.perf_counter() - t_route_start) * 1000
            metadata["latency_route_ms"] = latency_route_ms
            metadata["target_model"] = fallback_model
            metadata["route_reason"] = "no_resolver"
            metadata["route_score"] = score

            # Emit structured audit log for no resolver
            self._audit_logger.log_route_decision(
                request_id=metadata.get("request_id", ""),
                session_id=metadata.get("session_id", ""),
                prompt_hash=prompt_hash,
                prompt_length=len(original_text),
                route_score=score,
                candidates=[fallback_model],
                target_model=fallback_model,
                route_reason="no_resolver",
                latency_mask_ms=metadata.get("latency_mask_ms", 0.0),
                latency_route_ms=latency_route_ms,
                entities_detected=metadata.get("entities_detected"),
            )


# 全局回调实例 — 供 litellm_settings.callbacks 引用
smart_router_instance = SmartRouterCallback(enable_routing=True)
