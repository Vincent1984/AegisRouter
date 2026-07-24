"""FR-3.12 session routing policy tests (V4-8, V4-9, V4-10)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.config import RoutingConfig, TrivialConfig
from aegis_router.router.model_classifier import ClassifierResult, ModelClassifier
from aegis_router.router.route_resolver import RouteResolver
from aegis_router.router.rule_engine import RuleEngine, RuleEngineResult


@pytest.fixture
def tiers() -> list[dict]:
    def tier(name: str, score: float, low: float, high: float) -> dict:
        return {
            "name": name,
            "model": name,
            "computed_score": score,
            "score_range": (low, high),
            "cost_per_1m_input": score,
            "overridden": False,
        }

    return [
        tier("weak", 0.2, 0.0, 0.3),
        tier("medium", 0.5, 0.31, 0.7),
        tier("strong", 0.9, 0.71, 1.0),
    ]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestSessionPolicies:
    def test_v4_8_sticky_keeps_first_selected_model(self, tiers):
        """**Validates: Requirements FR-3.12 (V4-8)**"""
        resolver = RouteResolver(tiers)

        first = resolver.resolve(0.8, session_id="sticky-1", session_policy="sticky")
        second = resolver.resolve(0.1, session_id="sticky-1", session_policy="sticky")

        assert first["model"] == "strong"
        assert second == {
            "model": "strong",
            "reason": "session_sticky",
            "candidates": ["strong"],
        }

    def test_v4_9_escalate_only_upgrades_and_never_downgrades(self, tiers):
        """**Validates: Requirements FR-3.12 (V4-9)**"""
        resolver = RouteResolver(tiers)

        first = resolver.resolve(0.5, session_id="escalate-1", session_policy="escalate_only")
        upgraded = resolver.resolve(
            0.9, session_id="escalate-1", session_policy="escalate_only"
        )
        retained = resolver.resolve(
            0.1, session_id="escalate-1", session_policy="escalate_only"
        )

        assert first["model"] == "medium"
        assert upgraded["model"] == "strong"
        assert upgraded["reason"] == "session_escalated"
        assert retained["model"] == "strong"
        assert retained["reason"] == "session_no_downgrade"

    def test_v4_10_per_turn_routes_each_turn_independently(self, tiers):
        """**Validates: Requirements FR-3.12 (V4-10)**"""
        resolver = RouteResolver(tiers)

        models = [
            resolver.resolve(score, session_id="per-turn-1", session_policy="per_turn")[
                "model"
            ]
            for score in (0.8, 0.1, 0.5)
        ]

        assert models == ["strong", "weak", "medium"]

    def test_sessions_are_isolated(self, tiers):
        resolver = RouteResolver(tiers)
        resolver.resolve(0.1, session_id="a", session_policy="sticky")
        resolver.resolve(0.8, session_id="b", session_policy="sticky")

        assert resolver.resolve(0.8, "a", "sticky")["model"] == "weak"
        assert resolver.resolve(0.1, "b", "sticky")["model"] == "strong"

    def test_lock_expires_after_configured_ttl(self, tiers):
        clock = FakeClock()
        resolver = RouteResolver(tiers, session_lock_ttl_minutes=1, clock=clock)
        resolver.resolve(0.8, session_id="ttl-1", session_policy="sticky")

        clock.advance(61)
        after_expiry = resolver.resolve(0.1, session_id="ttl-1", session_policy="sticky")

        assert after_expiry["model"] == "weak"
        assert after_expiry["reason"] == "single_match"


    def test_escalate_only_never_reduces_capability_for_all_tier_pairs(self, tiers):
        """Finite-domain property: **Validates: Requirements FR-3.12**"""
        scores = (0.1, 0.5, 0.9)
        rank = {"weak": 0, "medium": 1, "strong": 2}

        for first_score in scores:
            for next_score in scores:
                resolver = RouteResolver(tiers)
                first = resolver.resolve(first_score, "property-session", "escalate_only")
                second = resolver.resolve(next_score, "property-session", "escalate_only")
                assert rank[second["model"]] >= rank[first["model"]]

    def test_invalid_session_policy_is_rejected(self, tiers):
        resolver = RouteResolver(tiers)
        with pytest.raises(ValueError, match="Invalid session policy"):
            resolver.resolve(0.5, session_id="s1", session_policy="invalid")


class TestSmartRouterSessionIntegration:
    @pytest.fixture
    def callback(self, tiers):
        pool = MagicMock(spec=ClawVaultPool)
        pool.max_connections = 10
        pool.call = AsyncMock()

        rule_engine = MagicMock(spec=RuleEngine)
        rule_engine.check.side_effect = [
            RuleEngineResult(matched=False),
            RuleEngineResult(matched=True, target_model="weak", matched_pattern="hi"),
        ]
        classifier = MagicMock(spec=ModelClassifier)
        classifier.aclassify = AsyncMock(
            return_value=ClassifierResult(
                score=0.9, classifier_type="mf", latency_ms=1.0
            )
        )

        callback = SmartRouterCallback(
            pool=pool,
            enable_routing=True,
            rule_engine=rule_engine,
            classifier=classifier,
        )
        callback._route_resolver = RouteResolver(tiers)
        callback._routing_config = RoutingConfig(
            session_policy="sticky",
            session_lock_ttl_minutes=60,
            trivial=TrivialConfig(enabled=True, target_model="weak"),
        )
        pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "complex request", "entities_found": []},
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "hi", "entities_found": []},
        ]
        return callback

    async def test_sticky_applies_to_rule_engine_shortcut(self, callback):
        first = {
            "messages": [{"role": "user", "content": "complex request"}],
            "metadata": {"session_id": "integrated-session", "request_id": "r1"},
        }
        second = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"session_id": "integrated-session", "request_id": "r2"},
        }

        await callback.async_pre_call_hook({}, None, first, "completion")
        await callback.async_pre_call_hook({}, None, second, "completion")

        assert first["model"] == "strong"
        assert second["model"] == "strong"
        assert second["metadata"]["route_reason"] == "session_sticky"


class TestSessionPolicyConfiguration:
    def test_defaults_match_design(self):
        config = RoutingConfig()
        assert config.session_policy == "sticky"
        assert config.session_lock_ttl_minutes == 60

    @pytest.mark.parametrize("policy", ["sticky", "per_turn", "escalate_only"])
    def test_all_supported_policies_are_accepted(self, policy):
        assert RoutingConfig(session_policy=policy).session_policy == policy

    def test_invalid_policy_and_non_positive_ttl_are_rejected(self):
        with pytest.raises(ValueError):
            RoutingConfig(session_policy="unknown")
        with pytest.raises(ValueError):
            RoutingConfig(session_lock_ttl_minutes=0)
