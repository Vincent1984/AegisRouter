"""主回调类 (pre_call + post_call)

实现 LiteLLM CustomLogger 接口，作为 AegisRouter 的核心回调。
通过 UDS (Unix) 或 TCP (Windows) 与 ClawVault 伴生进程通信，
完成 PII 脱敏、合规检测、占位符还原等安全管道。
同时集成智能路由链条：规则前置 → 打分 → 区间匹配 → 分发。
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from litellm.integrations.custom_logger import CustomLogger

from aegis_router.callbacks.degradation import (
    ComponentState,
    DegradationError,
    DegradationManager,
)
from aegis_router.callbacks.stream_rehydrator import StreamRehydrator
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.config import ClassifierConfig, RoutingConfig, TrivialConfig
from aegis_router.observability.audit_logger import AuditLogger
from aegis_router.router.config_watcher import ConfigWatcher
from aegis_router.router.model_classifier import ModelClassifier
from aegis_router.router.route_resolver import RouteResolver
from aegis_router.router.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Compliance mode
COMPLIANCE_MODE = os.environ.get("AEGIS_COMPLIANCE_MODE", "strict")

# Module-level pool instance — shared by all SmartRouterCallback instances
# within the same worker process.
_pool: ClawVaultPool = ClawVaultPool()


# ---------------------------------------------------------------------------
# SmartRouterCallback
# ---------------------------------------------------------------------------


class SmartRouterCallback(CustomLogger):
    """AegisRouter 主回调 — LiteLLM CustomLogger 实现。

    在 pre_call_hook 中执行:
      1. PII 合规检测 (check_compliance)
      2. PII 脱敏 (mask)
      3. 智能路由（规则前置 → 打分 → 区间匹配 → 分发）

    在 async_log_success_event 中执行:
      4. 占位符还原 (restore)
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
        super().__init__()
        self._pool = pool if pool is not None else _pool
        self._audit_logger = AuditLogger()
        self._degradation = degradation_manager or DegradationManager()

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
    # Async Pre-Call Hook
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> None:
        """Called before each LLM API call.

        1. Extract/generate session_id and request_id
        2. Concatenate user messages for scanning
        3. Call check_compliance — block if strict mode violation
        4. Call mask — replace PII in messages
        5. Store metadata for post-call use
        """
        messages = data.get("messages")
        if not messages:
            return

        # --- Extract/generate IDs ---
        metadata = data.get("metadata") or {}
        session_id = metadata.get("session_id") or str(uuid.uuid4())
        request_id = metadata.get("request_id") or str(uuid.uuid4())

        # Store IDs back into metadata for post-call retrieval
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["session_id"] = session_id
        data["metadata"]["request_id"] = request_id

        # --- Concatenate user message contents ---
        user_texts = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("content"):
                user_texts.append(msg["content"])

        full_text = "\n".join(user_texts)
        if not full_text.strip():
            return

        t_start = time.perf_counter()

        # --- Compliance Check ---
        compliance_result = await self._pool.call(
            "check_compliance",
            {"text": full_text, "direction": "inbound"},
        )

        if compliance_result is None:
            # ClawVault unavailable — bypass compliance and PII masking,
            # but routing should still work using original text.
            self._degradation.report_clawvault_unhealthy()
            logger.critical(
                "ClawVault 不可用: 跳过合规检测和 PII 脱敏 (request_id=%s)",
                request_id,
            )
            if self._enable_routing:
                prompt_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
                data["metadata"]["prompt_hash"] = prompt_hash
                await self._execute_routing(data, full_text, full_text, prompt_hash)
            return

        if not compliance_result.get("passed", True):
            mode = compliance_result.get("mode", COMPLIANCE_MODE)
            violations = compliance_result.get("violations", [])
            violation_desc = "; ".join(
                v.get("description", v.get("pattern", "unknown"))
                for v in violations
            )

            if mode == "strict":
                logger.warning(
                    "合规拦截 (strict): request_id=%s, violations=%s",
                    request_id,
                    violation_desc,
                )
                raise Exception(
                    f"Request blocked by compliance check: {violation_desc}"
                )
            elif mode == "permissive":
                logger.warning(
                    "合规告警 (permissive): request_id=%s, violations=%s",
                    request_id,
                    violation_desc,
                )
            else:
                # interactive or other — log and continue
                logger.warning(
                    "合规检测不通过 (mode=%s): request_id=%s, violations=%s",
                    mode,
                    request_id,
                    violation_desc,
                )

        # --- PII Masking ---
        mask_result = await self._pool.call(
            "mask",
            {
                "text": full_text,
                "session_id": session_id,
                "request_id": request_id,
            },
        )

        if mask_result is None:
            # ClawVault unavailable — bypass PII masking,
            # but routing should still work using original text.
            self._degradation.report_clawvault_unhealthy()
            logger.critical(
                "ClawVault 不可用: 跳过 PII 脱敏 (request_id=%s)",
                request_id,
            )
            if self._enable_routing:
                prompt_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
                data["metadata"]["prompt_hash"] = prompt_hash
                await self._execute_routing(data, full_text, full_text, prompt_hash)
            return

        # ClawVault responded successfully — report healthy
        self._degradation.report_clawvault_healthy()

        masked_text = mask_result.get("masked_text", full_text)
        entities_found = mask_result.get("entities_found", [])

        # --- Redis Health Check (when PII detected) ---
        # If PII was detected, we need Redis to store the mapping.
        # If Redis is down, the mapping would be lost, making restore impossible.
        pii_detected = len(entities_found) > 0
        if pii_detected:
            redis_state = await self._degradation.check_redis_health()
            if redis_state == ComponentState.UNHEALTHY:
                # Reject request: PII detected but Redis unavailable
                self._degradation.enforce_redis_policy(
                    pii_detected=True,
                    request_id=request_id,
                )
                # If enforce_redis_policy didn't raise (shouldn't happen), return
                return

        # --- Replace message contents with masked versions ---
        # Strategy: if there's a single user message, replace directly;
        # if multiple messages, mask each individually
        if len(user_texts) == 1:
            # Single text → replace with masked_text directly
            for msg in messages:
                if isinstance(msg, dict) and msg.get("content"):
                    msg["content"] = masked_text
                    break
        else:
            # Multiple messages: mask each one individually
            for msg in messages:
                if isinstance(msg, dict) and msg.get("content"):
                    individual_result = await self._pool.call(
                        "mask",
                        {
                            "text": msg["content"],
                            "session_id": session_id,
                            "request_id": request_id,
                        },
                    )
                    if individual_result:
                        msg["content"] = individual_result.get(
                            "masked_text", msg["content"]
                        )

        t_end = time.perf_counter()
        latency_mask_ms = (t_end - t_start) * 1000

        # --- Audit Logging ---
        prompt_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        entity_types = list(set(
            e.get("type", "UNKNOWN") for e in entities_found
        )) if entities_found else []

        logger.info(
            "pre_call_hook 完成: request_id=%s, session_id=%s, "
            "prompt_hash=%s, entities_detected=%s, "
            "latency_mask_ms=%.2f",
            request_id,
            session_id,
            prompt_hash[:16],
            entity_types,
            latency_mask_ms,
        )

        # Store timing in metadata for observability
        data["metadata"]["latency_mask_ms"] = latency_mask_ms
        data["metadata"]["entities_detected"] = entity_types
        data["metadata"]["prompt_hash"] = prompt_hash

        # ==================================================================
        # ROUTING CHAIN: 规则前置 → 打分 → 区间匹配 → 分发
        # ==================================================================
        if self._enable_routing:
            await self._execute_routing(data, masked_text, full_text, prompt_hash)

    # ------------------------------------------------------------------
    # Routing Pipeline
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

    # ------------------------------------------------------------------
    # Async Post-Call Success Hook (Response Restoration)
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after a successful LLM response.

        Extracts response text, calls ClawVault restore to rehydrate
        placeholders back to original PII values.
        """
        # --- Extract metadata ---
        metadata = kwargs.get("metadata") or kwargs.get("litellm_params", {}).get("metadata") or {}
        request_id = metadata.get("request_id")
        session_id = metadata.get("session_id")

        if not request_id:
            # No request_id means pre_call_hook didn't run (or bypass mode)
            logger.debug("async_log_success_event: no request_id, skipping restore")
            return

        # --- Extract response text ---
        response_text = self._extract_response_text(response_obj)
        if not response_text:
            return

        t_start = time.perf_counter()

        # --- Call ClawVault restore ---
        restore_result = await self._pool.call(
            "restore",
            {
                "text": response_text,
                "request_id": request_id,
                "session_id": session_id,
            },
        )

        if restore_result is None:
            # ClawVault unavailable — leave response with placeholders
            logger.critical(
                "ClawVault 不可用: 无法还原占位符 (request_id=%s), 响应将包含占位符",
                request_id,
            )
            return

        restored_text = restore_result.get("restored_text", response_text)

        # --- Replace response content ---
        self._set_response_text(response_obj, restored_text)

        t_end = time.perf_counter()
        latency_restore_ms = (t_end - t_start) * 1000

        logger.info(
            "async_log_success_event 还原完成: request_id=%s, "
            "latency_restore_ms=%.2f",
            request_id,
            latency_restore_ms,
        )

    # ------------------------------------------------------------------
    # Async Post-Call Streaming Iterator Hook (Streaming Restoration)
    # ------------------------------------------------------------------

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: dict,
        response,
        request_data: dict,
    ) -> AsyncGenerator:
        """Called for streaming responses to restore placeholders in SSE chunks.

        Wraps the streaming response iterator to intercept each chunk, use
        StreamRehydrator to buffer and restore split placeholders, and yield
        chunks with restored content.

        Bypass mode: If ClawVault is unavailable (mapping retrieval returns None
        or empty), pass through chunks unchanged.
        """
        # --- Extract metadata ---
        metadata = request_data.get("metadata") or {}
        request_id = metadata.get("request_id")
        session_id = metadata.get("session_id")

        if not request_id:
            # No request_id means pre_call_hook didn't run — pass through
            logger.debug(
                "async_post_call_streaming_iterator_hook: no request_id, bypassing"
            )
            async for chunk in response:
                yield chunk
            return

        t_start = time.perf_counter()

        # --- Get mapping from ClawVault ---
        mapping_result = await self._pool.call(
            "get_mapping",
            {"request_id": request_id, "session_id": session_id},
        )

        if mapping_result is None:
            # ClawVault unavailable — bypass mode
            logger.critical(
                "ClawVault 不可用: 无法获取映射表 (request_id=%s), 流式响应不还原",
                request_id,
            )
            async for chunk in response:
                yield chunk
            return

        mapping = mapping_result.get("mapping", {})

        if not mapping:
            # No mapping stored (no PII was detected) — pass through
            logger.debug(
                "async_post_call_streaming_iterator_hook: 映射表为空, 跳过还原 (request_id=%s)",
                request_id,
            )
            async for chunk in response:
                yield chunk
            return

        # --- Stream with rehydration ---
        rehydrator = StreamRehydrator(mapping)
        last_chunk = None

        async for chunk in response:
            # Extract content from chunk
            content = self._extract_streaming_content(chunk)

            if content is None:
                # No text content in this chunk (e.g., role-only, tool call)
                yield chunk
                continue

            # Process through rehydrator
            restored = rehydrator.process_chunk(content)

            if restored:
                self._set_streaming_content(chunk, restored)
                last_chunk = chunk
                yield chunk
            else:
                # Content is buffered (partial placeholder) — hold the chunk
                # We'll yield it when more content arrives or at flush
                last_chunk = chunk

        # --- Flush remaining buffer ---
        remaining = rehydrator.flush_remaining()
        if remaining:
            if last_chunk is not None:
                # Create a final chunk based on the last chunk structure
                final_chunk = self._create_streaming_chunk_from(last_chunk, remaining)
                yield final_chunk
            else:
                # Edge case: no chunks were received but buffer has content
                logger.warning(
                    "flush_remaining has content but no last_chunk template "
                    "(request_id=%s)",
                    request_id,
                )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        logger.info(
            "async_post_call_streaming_iterator_hook 完成: request_id=%s, "
            "latency_total_ms=%.2f",
            request_id,
            latency_ms,
        )

    # ------------------------------------------------------------------
    # Async Failure Hook
    # ------------------------------------------------------------------

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """异步失败回调 — 记录失败信息。"""
        logger.warning("async_failure_event: model=%s", kwargs.get("model"))

    # ------------------------------------------------------------------
    # Sync Hooks (backward compatibility)
    # ------------------------------------------------------------------

    def log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        """LiteLLM pre-call 同步钩子 (非 async)。"""
        logger.debug("pre_api_call: model=%s", model)

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """LiteLLM 成功回调。"""
        logger.debug("success_event: model=%s", kwargs.get("model"))

    def log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """LiteLLM 失败回调。"""
        logger.warning("failure_event: model=%s", kwargs.get("model"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_text(response_obj: Any) -> Optional[str]:
        """Extract the text content from a LiteLLM response object.

        Handles ModelResponse objects with choices[0].message.content structure.
        """
        try:
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content
        except (IndexError, AttributeError):
            pass

        # Fallback: try dict access
        if isinstance(response_obj, dict):
            choices = response_obj.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content")

        return None

    @staticmethod
    def _set_response_text(response_obj: Any, text: str) -> None:
        """Set the response text content in a LiteLLM response object."""
        try:
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    choice.message.content = text
                    return
        except (IndexError, AttributeError):
            pass

        # Fallback: try dict access
        if isinstance(response_obj, dict):
            choices = response_obj.get("choices", [])
            if choices:
                if "message" in choices[0]:
                    choices[0]["message"]["content"] = text

    @staticmethod
    def _extract_streaming_content(chunk: Any) -> Optional[str]:
        """Extract text content from a streaming chunk.

        Streaming chunks use choices[0].delta.content instead of message.content.
        Returns None if the chunk has no text content (e.g., role-only chunk).
        """
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta"):
                    delta = choice.delta
                    if hasattr(delta, "content"):
                        return delta.content
        except (IndexError, AttributeError):
            pass

        # Fallback: dict access
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                return delta.get("content")

        return None

    @staticmethod
    def _set_streaming_content(chunk: Any, text: str) -> None:
        """Set text content in a streaming chunk's delta."""
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    choice.delta.content = text
                    return
        except (IndexError, AttributeError):
            pass

        # Fallback: dict access
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                if "delta" in choices[0]:
                    choices[0]["delta"]["content"] = text

    @staticmethod
    def _create_streaming_chunk_from(template_chunk: Any, content: str) -> Any:
        """Create a new streaming chunk with given content, based on a template chunk.

        Uses copy.deepcopy to avoid mutating the template, then sets the content.
        """
        new_chunk = copy.deepcopy(template_chunk)
        SmartRouterCallback._set_streaming_content(new_chunk, content)
        return new_chunk


# 全局回调实例 — 供 litellm_settings.callbacks 引用
smart_router_instance = SmartRouterCallback(enable_routing=True)
