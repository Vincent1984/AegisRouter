"""区间匹配、重叠策略与会话路由策略。"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import Any


class RouteResolver:
    """根据难度分数选模，并在 ``session_id`` 粒度执行会话策略。"""

    VALID_STRATEGIES = ("lowest_cost", "highest_capability", "round_robin", "random")
    VALID_SESSION_POLICIES = ("sticky", "per_turn", "escalate_only")

    def __init__(
        self,
        tiers: list[dict[str, Any]],
        strategy: str = "lowest_cost",
        fallback_model: str = "deepseek-v3",
        session_lock_ttl_minutes: float = 60,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._validate_strategy(strategy)
        if session_lock_ttl_minutes <= 0:
            raise ValueError("session_lock_ttl_minutes must be greater than 0")

        self.tiers = tiers
        self.strategy = strategy
        self.fallback = fallback_model
        self.session_lock_ttl_minutes = session_lock_ttl_minutes
        self._clock = clock or time.monotonic
        self._round_robin_counter = 0
        self._session_models: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def resolve(
        self,
        prompt_score: float,
        session_id: str | None = None,
        session_policy: str = "sticky",
    ) -> dict[str, Any]:
        """独立解析当前轮次，再应用配置的会话策略。"""
        decision = self._resolve_without_session(prompt_score)
        return self.apply_session_policy(decision, session_id, session_policy)

    def apply_session_policy(
        self,
        decision: dict[str, Any],
        session_id: str | None,
        session_policy: str = "sticky",
    ) -> dict[str, Any]:
        """将会话策略应用到任意选模结果（包括规则前置和降级结果）。"""
        if session_policy not in self.VALID_SESSION_POLICIES:
            raise ValueError(
                f"Invalid session policy '{session_policy}'. Must be one of: "
                f"{', '.join(self.VALID_SESSION_POLICIES)}"
            )
        if not session_id or session_policy == "per_turn":
            return dict(decision)

        now = self._clock()
        with self._lock:
            previous = self._session_models.get(session_id)
            if previous is not None and previous["expires_at"] <= now:
                del self._session_models[session_id]
                previous = None

            current_model = decision["model"]
            current_score = self._get_tier_score(current_model)
            if previous is None:
                self._store_session(session_id, current_model, current_score, now)
                return dict(decision)

            if session_policy == "sticky":
                self._refresh_session(previous, now)
                return self._session_decision(previous["model"], "session_sticky")

            if self._can_escalate(previous, current_model, current_score):
                reason = (
                    "session_escalated"
                    if current_model != previous["model"]
                    else "session_same_tier"
                )
                self._store_session(session_id, current_model, current_score, now)
                result = dict(decision)
                result["reason"] = reason
                return result

            self._refresh_session(previous, now)
            return self._session_decision(previous["model"], "session_no_downgrade")

    def update_configuration(
        self,
        tiers: list[dict[str, Any]],
        strategy: str,
        fallback_model: str,
        session_lock_ttl_minutes: float,
        clear_session_locks: bool = False,
    ) -> None:
        """热更新路由配置，同时在策略不变时保留有效会话锁。"""
        self._validate_strategy(strategy)
        if session_lock_ttl_minutes <= 0:
            raise ValueError("session_lock_ttl_minutes must be greater than 0")
        with self._lock:
            self.tiers = tiers
            self.strategy = strategy
            self.fallback = fallback_model
            self.session_lock_ttl_minutes = session_lock_ttl_minutes
            if clear_session_locks:
                self._session_models.clear()
            else:
                now = self._clock()
                ttl_seconds = session_lock_ttl_minutes * 60
                for session in self._session_models.values():
                    session["tier_score"] = self._get_tier_score(session["model"])
                    session["expires_at"] = now + ttl_seconds

    def clear_session(self, session_id: str) -> None:
        """显式移除单个会话锁。"""
        with self._lock:
            self._session_models.pop(session_id, None)

    def _resolve_without_session(self, prompt_score: float) -> dict[str, Any]:
        candidates = [
            tier
            for tier in self.tiers
            if tier["score_range"][0] <= prompt_score <= tier["score_range"][1]
        ]
        if not candidates:
            return {"model": self.fallback, "reason": "no_match_fallback", "candidates": []}

        candidate_models = [tier["model"] for tier in candidates]
        if len(candidates) == 1:
            return {
                "model": candidates[0]["model"],
                "reason": "single_match",
                "candidates": candidate_models,
            }

        selected = self._apply_strategy(candidates)
        return {
            "model": selected["model"],
            "reason": f"overlap_{self.strategy}",
            "candidates_count": len(candidates),
            "candidates": candidate_models,
        }

    def _can_escalate(
        self,
        previous: dict[str, Any],
        current_model: str,
        current_score: float | None,
    ) -> bool:
        if current_model == previous["model"]:
            return True
        previous_score = previous["tier_score"]
        if current_score is None:
            return False
        if previous_score is None:
            return True
        return current_score >= previous_score

    def _get_tier_score(self, model: str) -> float | None:
        for tier in self.tiers:
            if model in (tier.get("name"), tier.get("model")):
                return float(tier["computed_score"])
        return None

    def _store_session(
        self,
        session_id: str,
        model: str,
        tier_score: float | None,
        now: float,
    ) -> None:
        self._session_models[session_id] = {
            "model": model,
            "tier_score": tier_score,
            "expires_at": now + self.session_lock_ttl_minutes * 60,
        }

    def _refresh_session(self, session: dict[str, Any], now: float) -> None:
        session["expires_at"] = now + self.session_lock_ttl_minutes * 60

    @staticmethod
    def _session_decision(model: str, reason: str) -> dict[str, Any]:
        return {"model": model, "reason": reason, "candidates": [model]}

    @classmethod
    def _validate_strategy(cls, strategy: str) -> None:
        if strategy not in cls.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Must be one of: {', '.join(cls.VALID_STRATEGIES)}"
            )

    def _apply_strategy(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if self.strategy == "lowest_cost":
            return min(candidates, key=lambda tier: tier["cost_per_1m_input"])
        if self.strategy == "highest_capability":
            return max(candidates, key=lambda tier: tier["computed_score"])
        if self.strategy == "round_robin":
            index = self._round_robin_counter % len(candidates)
            self._round_robin_counter += 1
            return candidates[index]
        return random.choice(candidates)
