"""Tests for routing chain integration in SmartRouterCallback.

Tests cover the complete routing pipeline:
- Rule Engine (寒暄检测) → direct route to local model
- ModelClassifier (打分) → difficulty scoring
- RouteResolver (区间匹配) → model selection
- Graceful degradation (classifier timeout/error → fallback)
- Config hot-reload via ConfigWatcher callback
- ClawVault bypass mode with routing still active
- V4-2: Full RouteLLM scoring + route resolving pipeline
"""

from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.config import (
    ClassifierConfig,
    RoutingConfig,
    ScoringConfig,
    TrivialConfig,
)
from aegis_router.router.model_classifier import ClassifierResult, ModelClassifier
from aegis_router.router.model_scorer import ModelScorer, build_routing_table
from aegis_router.router.route_resolver import RouteResolver
from aegis_router.router.rule_engine import RuleEngine, RuleEngineResult


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool that passes compliance and masking."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock()
    pool.max_connections = 10
    return pool


@pytest.fixture
def mock_rule_engine():
    """Create a mock RuleEngine."""
    engine = MagicMock(spec=RuleEngine)
    engine.check.return_value = RuleEngineResult(matched=False)
    return engine


@pytest.fixture
def mock_classifier():
    """Create a mock ModelClassifier."""
    classifier = MagicMock(spec=ModelClassifier)
    classifier.aclassify = AsyncMock(
        return_value=ClassifierResult(score=0.6, classifier_type="mf", latency_ms=5.0)
    )
    return classifier


@pytest.fixture
def mock_resolver():
    """Create a mock RouteResolver."""
    resolver = MagicMock(spec=RouteResolver)
    resolver.resolve.return_value = {
        "model": "openai/gpt-4o",
        "reason": "single_match",
    }
    resolver.apply_session_policy.side_effect = (
        lambda decision, session_id, session_policy: dict(decision)
    )
    return resolver


@pytest.fixture
def routing_config():
    """Create a routing config for tests."""
    return RoutingConfig(
        score_input="masked",
        trivial=TrivialConfig(enabled=True, max_length=30, target_model="local-7b"),
        classifier=ClassifierConfig(type="mf"),
        overlap_strategy="lowest_cost",
        fallback_model="deepseek-v3",
    )


@pytest.fixture
def callback_with_routing(mock_pool, mock_rule_engine, mock_classifier, mock_resolver, routing_config):
    """Create a SmartRouterCallback with all routing components mocked."""
    cb = SmartRouterCallback(
        pool=mock_pool,
        enable_routing=True,
        rule_engine=mock_rule_engine,
        classifier=mock_classifier,
    )
    # Inject route resolver and config directly
    cb._route_resolver = mock_resolver
    cb._routing_config = routing_config
    return cb


@pytest.fixture
def sample_data():
    """Sample request data dict."""
    return {
        "messages": [
            {"role": "user", "content": "Please explain the theory of relativity in detail."},
        ],
        "model": "gpt-4o",
        "metadata": {
            "session_id": "test-session-1",
            "request_id": "test-request-1",
        },
    }


def setup_pool_normal(mock_pool, masked_text="masked prompt text"):
    """Configure pool to return normal compliance+mask results."""
    mock_pool.call.side_effect = [
        # compliance passes
        {"passed": True, "violations": [], "mode": "strict"},
        # mask result
        {"masked_text": masked_text, "entities_found": []},
    ]


# ---------------------------------------------------------------------------
# Tests: Routing — Rule Engine (寒暄检测)
# ---------------------------------------------------------------------------


class TestRoutingRuleEngine:
    """Test routing when rule engine matches trivial chat."""

    async def test_trivial_chat_routes_to_local_model(
        self, callback_with_routing, mock_pool, mock_rule_engine, sample_data
    ):
        """Trivial chat detected by rule engine routes directly to local-7b."""
        sample_data["messages"][0]["content"] = "你好"
        setup_pool_normal(mock_pool, masked_text="你好")

        mock_rule_engine.check.return_value = RuleEngineResult(
            matched=True, target_model="local-7b", matched_pattern="你好"
        )

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        assert sample_data["model"] == "local-7b"
        assert sample_data["metadata"]["target_model"] == "local-7b"
        assert sample_data["metadata"]["route_reason"] == "trivial_chat"
        assert sample_data["metadata"]["route_matched_pattern"] == "你好"

    async def test_trivial_chat_skips_classifier(
        self, callback_with_routing, mock_pool, mock_rule_engine, mock_classifier, sample_data
    ):
        """When rule engine matches, classifier is never called."""
        sample_data["messages"][0]["content"] = "hi"
        setup_pool_normal(mock_pool, masked_text="hi")

        mock_rule_engine.check.return_value = RuleEngineResult(
            matched=True, target_model="local-7b", matched_pattern="hi"
        )

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        mock_classifier.aclassify.assert_not_called()

    async def test_non_trivial_proceeds_to_classifier(
        self, callback_with_routing, mock_pool, mock_rule_engine, mock_classifier, sample_data
    ):
        """Non-trivial prompt passes rule engine and proceeds to classifier."""
        setup_pool_normal(mock_pool, masked_text="masked prompt text")

        mock_rule_engine.check.return_value = RuleEngineResult(matched=False)

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        mock_classifier.aclassify.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Routing — Classifier + Resolver
# ---------------------------------------------------------------------------


class TestRoutingClassifierAndResolver:
    """Test routing with classifier scoring and route resolver."""

    async def test_classifier_score_passed_to_resolver(
        self, callback_with_routing, mock_pool, mock_classifier, mock_resolver, sample_data
    ):
        """Classifier score is passed to RouteResolver.resolve()."""
        setup_pool_normal(mock_pool, masked_text="masked prompt text")

        mock_classifier.aclassify.return_value = ClassifierResult(
            score=0.75, classifier_type="mf", latency_ms=3.0
        )
        mock_resolver.resolve.return_value = {
            "model": "openai/gpt-4o",
            "reason": "single_match",
        }

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        mock_resolver.resolve.assert_called_once_with(
            0.75,
            session_id="test-session-1",
            session_policy="sticky",
        )

    async def test_model_set_to_resolved_model(
        self, callback_with_routing, mock_pool, mock_classifier, mock_resolver, sample_data
    ):
        """data['model'] is set to the resolved model from RouteResolver."""
        setup_pool_normal(mock_pool, masked_text="masked prompt text")

        mock_classifier.aclassify.return_value = ClassifierResult(
            score=0.3, classifier_type="mf", latency_ms=2.0
        )
        mock_resolver.resolve.return_value = {
            "model": "deepseek/deepseek-chat",
            "reason": "single_match",
        }

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        assert sample_data["model"] == "deepseek/deepseek-chat"
        assert sample_data["metadata"]["target_model"] == "deepseek/deepseek-chat"
        assert sample_data["metadata"]["route_score"] == 0.3

    async def test_overlap_strategy_candidates_stored(
        self, callback_with_routing, mock_pool, mock_classifier, mock_resolver, sample_data
    ):
        """Overlap resolution result with candidates_count stored in metadata."""
        setup_pool_normal(mock_pool, masked_text="masked prompt text")

        mock_classifier.aclassify.return_value = ClassifierResult(
            score=0.6, classifier_type="mf", latency_ms=4.0
        )
        mock_resolver.resolve.return_value = {
            "model": "openai/gpt-4o",
            "reason": "overlap_lowest_cost",
            "candidates_count": 3,
        }

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        assert sample_data["metadata"]["route_candidates_count"] == 3
        assert sample_data["metadata"]["route_reason"] == "overlap_lowest_cost"

    async def test_latency_route_ms_stored(
        self, callback_with_routing, mock_pool, mock_classifier, mock_resolver, sample_data
    ):
        """Routing latency is recorded in metadata."""
        setup_pool_normal(mock_pool, masked_text="masked prompt text")

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        assert "latency_route_ms" in sample_data["metadata"]
        assert sample_data["metadata"]["latency_route_ms"] >= 0


# ---------------------------------------------------------------------------
# Tests: Routing — Graceful Degradation
# ---------------------------------------------------------------------------


class TestRoutingGracefulDegradation:
    """Test fallback behavior when classifier times out or errors."""

    async def test_classifier_timeout_uses_fallback(
        self, callback_with_routing, mock_pool, mock_classifier, sample_data
    ):
        """Classifier timeout routes to fallback model (deepseek-v3)."""
        setup_pool_normal(mock_pool, masked_text="masked prompt text")

        mock_classifier.aclassify.side_effect = TimeoutError("Inference exceeded timeout")

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        assert sample_data["model"] == "deepseek-v3"
        assert sample_data["metadata"]["target_model"] == "deepseek-v3"
        assert sample_data["metadata"]["route_reason"] == "classifier_timeout"

    async def test_classifier_runtime_error_uses_fallback(
        self, callback_with_routing, mock_pool, mock_classifier, sample_data
    ):
        """Classifier RuntimeError routes to fallback model."""
        setup_pool_normal(mock_pool, masked_text="masked prompt text")

        mock_classifier.aclassify.side_effect = RuntimeError("Model failed to load")

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        assert sample_data["model"] == "deepseek-v3"
        assert sample_data["metadata"]["route_reason"] == "classifier_error"

    async def test_no_classifier_uses_fallback(self, mock_pool, mock_rule_engine, routing_config):
        """When no classifier is provided, fallback model is used."""
        cb = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=mock_rule_engine,
            classifier=None,
        )
        cb._routing_config = routing_config

        data = {
            "messages": [{"role": "user", "content": "Complex question here"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "Complex question here", "entities_found": []},
        ]

        await cb.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "no_classifier"


# ---------------------------------------------------------------------------
# Tests: Routing — score_input mode
# ---------------------------------------------------------------------------


class TestRoutingScoreInput:
    """Test score_input configuration (masked vs original)."""

    async def test_masked_mode_uses_masked_text(
        self, callback_with_routing, mock_pool, mock_classifier, sample_data
    ):
        """In masked mode, classifier receives masked text."""
        setup_pool_normal(mock_pool, masked_text="Hello [PERSON_1], explain relativity")

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        # Classifier receives the masked text
        mock_classifier.aclassify.assert_called_once_with(
            "Hello [PERSON_1], explain relativity"
        )

    async def test_original_mode_uses_original_text(
        self, mock_pool, mock_rule_engine, mock_classifier, mock_resolver
    ):
        """In original mode, classifier receives original (unmasked) text."""
        routing_cfg = RoutingConfig(
            score_input="original",
            trivial=TrivialConfig(enabled=True, max_length=30, target_model="local-7b"),
            classifier=ClassifierConfig(type="mf"),
            overlap_strategy="lowest_cost",
            fallback_model="deepseek-v3",
        )

        cb = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=mock_rule_engine,
            classifier=mock_classifier,
        )
        cb._route_resolver = mock_resolver
        cb._routing_config = routing_cfg

        data = {
            "messages": [{"role": "user", "content": "My name is John, explain relativity"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "My name is [PERSON_1], explain relativity", "entities_found": []},
        ]

        await cb.async_pre_call_hook({}, None, data, "completion")

        # Classifier receives the original text
        mock_classifier.aclassify.assert_called_once_with(
            "My name is John, explain relativity"
        )


# ---------------------------------------------------------------------------
# Tests: Routing — ClawVault bypass with routing still active
# ---------------------------------------------------------------------------


class TestRoutingWithClawVaultBypass:
    """Test that routing works even when ClawVault is unavailable."""

    async def test_routing_works_when_compliance_unavailable(
        self, callback_with_routing, mock_pool, mock_classifier, mock_resolver, sample_data
    ):
        """Routing executes using original text when ClawVault compliance check fails."""
        # ClawVault returns None (unavailable)
        mock_pool.call.return_value = None

        mock_classifier.aclassify.return_value = ClassifierResult(
            score=0.8, classifier_type="mf", latency_ms=3.0
        )
        mock_resolver.resolve.return_value = {
            "model": "openai/gpt-4o",
            "reason": "single_match",
        }

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        # Model should still be routed
        assert sample_data["model"] == "openai/gpt-4o"
        assert sample_data["metadata"]["target_model"] == "openai/gpt-4o"

    async def test_routing_works_when_mask_unavailable(
        self, callback_with_routing, mock_pool, mock_classifier, mock_resolver, sample_data
    ):
        """Routing executes using original text when masking is unavailable."""
        mock_pool.call.side_effect = [
            # Compliance passes
            {"passed": True, "violations": [], "mode": "strict"},
            # Mask returns None (unavailable)
            None,
        ]

        mock_classifier.aclassify.return_value = ClassifierResult(
            score=0.5, classifier_type="mf", latency_ms=4.0
        )
        mock_resolver.resolve.return_value = {
            "model": "deepseek/deepseek-chat",
            "reason": "single_match",
        }

        await callback_with_routing.async_pre_call_hook({}, None, sample_data, "completion")

        assert sample_data["model"] == "deepseek/deepseek-chat"
        assert sample_data["metadata"]["target_model"] == "deepseek/deepseek-chat"


# ---------------------------------------------------------------------------
# Tests: Routing Disabled
# ---------------------------------------------------------------------------


class TestRoutingDisabled:
    """Test behavior when routing is disabled."""

    async def test_routing_disabled_does_not_change_model(self, mock_pool):
        """When routing is disabled, model is not changed."""
        cb = SmartRouterCallback(pool=mock_pool, enable_routing=False)

        data = {
            "messages": [{"role": "user", "content": "Hello world"}],
            "model": "original-model",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "Hello world", "entities_found": []},
        ]

        await cb.async_pre_call_hook({}, None, data, "completion")

        # Model remains unchanged
        assert data["model"] == "original-model"
        assert "target_model" not in data["metadata"]


# ---------------------------------------------------------------------------
# Tests: Config Hot-Reload
# ---------------------------------------------------------------------------


class TestRoutingConfigReload:
    """Test routing table update via ConfigWatcher callback."""

    def test_on_routing_table_updated_rebuilds_resolver(
        self, mock_pool, mock_rule_engine, mock_classifier
    ):
        """Routing table update callback rebuilds RouteResolver."""
        mock_watcher = MagicMock()
        mock_config = MagicMock()
        mock_config.routing = RoutingConfig(
            score_input="masked",
            trivial=TrivialConfig(enabled=True, max_length=30, target_model="local-7b"),
            classifier=ClassifierConfig(type="mf"),
            overlap_strategy="highest_capability",
            fallback_model="openai/o1",
        )
        mock_watcher.get_current_config.return_value = mock_config
        mock_watcher.get_current_routing_table.return_value = []

        cb = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=mock_rule_engine,
            classifier=mock_classifier,
            config_watcher=mock_watcher,
        )

        new_table = [
            {
                "name": "gpt-4o",
                "model": "openai/gpt-4o",
                "computed_score": 0.8,
                "score_range": (0.5, 0.9),
                "cost_per_1m_input": 2.5,
                "overridden": False,
            }
        ]

        cb._on_routing_table_updated(new_table)

        # Verify resolver was rebuilt
        assert cb._route_resolver is not None
        assert cb._routing_config.overlap_strategy == "highest_capability"
        assert cb._routing_config.fallback_model == "openai/o1"


# ---------------------------------------------------------------------------
# V4-1: 验证发送 "你好" → 路由到 local-7b（规则前置命中）
# ---------------------------------------------------------------------------


class TestV4_1_TrivialChatFullPipeline:
    """V4-1: Integration test verifying the full routing pipeline routes
    trivial chat (e.g. "你好") to local-7b via the real RuleEngine with
    actual patterns file.

    This tests the complete path:
      SmartRouterCallback.async_pre_call_hook →
      RuleEngine.check() (real, with ./patterns/trivial_chat.txt) →
      data["model"] set to "local-7b" + metadata populated
    """

    @pytest.fixture
    def real_rule_engine(self) -> RuleEngine:
        """Create a real RuleEngine using the actual patterns file."""
        config = TrivialConfig(
            enabled=True,
            max_length=30,
            patterns_file="./patterns/trivial_chat.txt",
            target_model="local-7b",
        )
        return RuleEngine(config)

    @pytest.fixture
    def v4_callback(self, mock_pool, real_rule_engine) -> SmartRouterCallback:
        """Create SmartRouterCallback with real RuleEngine, no classifier (falls to fallback)."""
        cb = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=real_rule_engine,
            classifier=None,
        )
        cb._routing_config = RoutingConfig(
            score_input="masked",
            trivial=TrivialConfig(
                enabled=True,
                max_length=30,
                patterns_file="./patterns/trivial_chat.txt",
                target_model="local-7b",
            ),
            classifier=ClassifierConfig(type="mf"),
            overlap_strategy="lowest_cost",
            fallback_model="deepseek-v3",
        )
        return cb

    def _setup_pool_passthrough(self, mock_pool, text: str):
        """Configure mock pool to pass compliance and return text unchanged."""
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": text, "entities_found": []},
        ]

    # --- Core test: "你好" routes to local-7b ---

    async def test_nihao_routes_to_local_7b(self, v4_callback, mock_pool):
        """发送 '你好' → 规则前置命中 → 路由到 local-7b。"""
        data = {
            "messages": [{"role": "user", "content": "你好"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, "你好")

        await v4_callback.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "local-7b"
        assert data["metadata"]["target_model"] == "local-7b"
        assert data["metadata"]["route_reason"] == "trivial_chat"
        assert data["metadata"]["route_matched_pattern"] == "你好"

    # --- Additional trivial patterns ---

    @pytest.mark.parametrize(
        "prompt,expected_pattern",
        [
            ("hello", "hello"),
            ("hi", "hi"),
            ("谢谢", "谢谢"),
            ("再见", "再见"),
            ("早上好", "早上好"),
            ("good morning", "good morning"),
            ("thanks", "thanks"),
            ("bye", "bye"),
        ],
    )
    async def test_trivial_patterns_route_to_local_7b(
        self, v4_callback, mock_pool, prompt: str, expected_pattern: str
    ):
        """Multiple trivial patterns all route to local-7b via rule engine."""
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, prompt)

        await v4_callback.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "local-7b"
        assert data["metadata"]["target_model"] == "local-7b"
        assert data["metadata"]["route_reason"] == "trivial_chat"
        assert data["metadata"]["route_matched_pattern"] == expected_pattern

    # --- Non-trivial prompt does NOT hit rule engine ---

    async def test_non_trivial_prompt_does_not_route_to_local_7b(
        self, v4_callback, mock_pool
    ):
        """非寒暄 prompt '帮我写一篇产品分析报告' 不命中规则引擎，走 fallback。"""
        data = {
            "messages": [{"role": "user", "content": "帮我写一篇产品分析报告"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, "帮我写一篇产品分析报告")

        await v4_callback.async_pre_call_hook({}, None, data, "completion")

        # Should NOT be local-7b — no classifier so falls to fallback
        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "no_classifier"

    # --- Case insensitivity ---

    async def test_hello_uppercase_routes_to_local_7b(self, v4_callback, mock_pool):
        """'HELLO' (uppercase) still matches rule engine (case-insensitive)."""
        data = {
            "messages": [{"role": "user", "content": "HELLO"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, "HELLO")

        await v4_callback.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "local-7b"
        assert data["metadata"]["route_reason"] == "trivial_chat"

    # --- Long text with greeting does NOT match ---

    async def test_long_text_with_greeting_not_routed(self, v4_callback, mock_pool):
        """Long text (>30 chars) containing a greeting word does NOT trigger rule engine."""
        long_prompt = "你好，我想了解一下量子计算的基本概念以及它在现代密码学中的应用"
        assert len(long_prompt.strip()) > 30

        data = {
            "messages": [{"role": "user", "content": long_prompt}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, long_prompt)

        await v4_callback.async_pre_call_hook({}, None, data, "completion")

        # Should NOT be local-7b
        assert data["model"] != "local-7b"
        assert data["metadata"]["route_reason"] != "trivial_chat"


# ---------------------------------------------------------------------------
# V4-2: 发送 "帮我写一篇产品分析报告" → RouteLLM 打分 → 路由到配置的对应模型
# ---------------------------------------------------------------------------


class TestV4_2_RouteLLMScoringPipeline:
    """V4-2: Integration test verifying the full RouteLLM scoring + route
    resolving pipeline.

    Tests the complete path:
      SmartRouterCallback.async_pre_call_hook →
      RuleEngine.check() (NOT matched — non-trivial prompt) →
      ModelClassifier.aclassify() (mocked, returns controlled score) →
      RouteResolver.resolve() (real, with actual routing table from models.yaml) →
      data["model"] set to resolved model + metadata populated

    The routing table is built from the REAL models.yaml and route_overrides.yaml
    using the actual ModelScorer with weights from route_config.yaml.
    Only the RouteLLM classifier is mocked (since the model may not be available
    in the test environment).
    """

    PROMPT = "帮我写一篇产品分析报告"

    @pytest.fixture
    def real_rule_engine(self) -> RuleEngine:
        """Create a real RuleEngine using the actual patterns file."""
        config = TrivialConfig(
            enabled=True,
            max_length=30,
            patterns_file="./patterns/trivial_chat.txt",
            target_model="local-7b",
        )
        return RuleEngine(config)

    @pytest.fixture
    def scoring_config(self) -> dict:
        """Load the real scoring config from route_config.yaml."""
        with open("./config/route_config.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        scoring = raw["routing"]["scoring"]
        return scoring

    @pytest.fixture
    def real_scorer(self, scoring_config) -> ModelScorer:
        """Create a real ModelScorer using weights/normalization from route_config.yaml."""
        return ModelScorer(
            weights=scoring_config["weights"],
            normalization=scoring_config["normalization"],
            tolerance=scoring_config["range_tolerance"],
        )

    @pytest.fixture
    def models_config(self) -> list:
        """Load the real models config from models.yaml."""
        with open("./config/models.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return raw["models"]

    @pytest.fixture
    def overrides_config(self) -> dict:
        """Load the real route overrides from route_overrides.yaml."""
        with open("./config/route_overrides.yaml", "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return raw.get("overrides", {})

    @pytest.fixture
    def real_routing_table(self, models_config, real_scorer, overrides_config) -> list:
        """Build a real routing table from models.yaml + overrides."""
        return build_routing_table(models_config, real_scorer, overrides_config)

    @pytest.fixture
    def real_resolver(self, real_routing_table) -> RouteResolver:
        """Create a real RouteResolver with actual routing table, lowest_cost strategy."""
        return RouteResolver(
            tiers=real_routing_table,
            strategy="lowest_cost",
            fallback_model="deepseek-v3",
        )

    @pytest.fixture
    def valid_model_names(self, real_routing_table) -> set:
        """Get all valid model names from the routing table."""
        return {tier["model"] for tier in real_routing_table}

    @pytest.fixture
    def mock_classifier_for_v4_2(self) -> MagicMock:
        """Create a mock classifier that returns a controlled score."""
        classifier = MagicMock(spec=ModelClassifier)
        classifier.aclassify = AsyncMock(
            return_value=ClassifierResult(score=0.5, classifier_type="mf", latency_ms=5.0)
        )
        return classifier

    @pytest.fixture
    def v4_2_callback(
        self,
        mock_pool,
        real_rule_engine,
        mock_classifier_for_v4_2,
        real_resolver,
    ) -> SmartRouterCallback:
        """Create SmartRouterCallback with real RuleEngine + real RouteResolver,
        mock classifier."""
        cb = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=real_rule_engine,
            classifier=mock_classifier_for_v4_2,
        )
        cb._route_resolver = real_resolver
        cb._routing_config = RoutingConfig(
            score_input="masked",
            trivial=TrivialConfig(
                enabled=True,
                max_length=30,
                patterns_file="./patterns/trivial_chat.txt",
                target_model="local-7b",
            ),
            classifier=ClassifierConfig(type="mf"),
            overlap_strategy="lowest_cost",
            fallback_model="deepseek-v3",
        )
        return cb

    def _setup_pool_passthrough(self, mock_pool, text: str):
        """Configure mock pool to pass compliance and return text unchanged."""
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": text, "entities_found": []},
        ]

    # --- Core test: prompt does NOT match rule engine ---

    async def test_prompt_does_not_match_trivial_rule(self, real_rule_engine):
        """'帮我写一篇产品分析报告' is NOT a trivial chat — rule engine should not match."""
        result = real_rule_engine.check(self.PROMPT)
        assert result.matched is False

    # --- Core test: full pipeline with score 0.5 ---

    async def test_full_pipeline_score_0_5_routes_to_valid_model(
        self, v4_2_callback, mock_pool, mock_classifier_for_v4_2, valid_model_names
    ):
        """Full pipeline: prompt → classifier score 0.5 → resolves to valid model."""
        data = {
            "messages": [{"role": "user", "content": self.PROMPT}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, self.PROMPT)

        mock_classifier_for_v4_2.aclassify.return_value = ClassifierResult(
            score=0.5, classifier_type="mf", latency_ms=5.0
        )

        await v4_2_callback.async_pre_call_hook({}, None, data, "completion")

        # Model should be one of the configured models
        assert data["model"] in valid_model_names
        assert data["metadata"]["target_model"] in valid_model_names
        assert data["metadata"]["route_score"] == 0.5
        assert data["metadata"]["route_reason"] != "trivial_chat"
        assert data["metadata"]["route_reason"] != "no_classifier"
        assert "latency_route_ms" in data["metadata"]

    # --- Verify classifier is called with the prompt text ---

    async def test_classifier_called_with_prompt_text(
        self, v4_2_callback, mock_pool, mock_classifier_for_v4_2
    ):
        """Classifier receives the prompt text for scoring."""
        data = {
            "messages": [{"role": "user", "content": self.PROMPT}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, self.PROMPT)

        await v4_2_callback.async_pre_call_hook({}, None, data, "completion")

        mock_classifier_for_v4_2.aclassify.assert_called_once_with(self.PROMPT)

    # --- Score-dependent routing: low score ---

    async def test_low_score_routes_to_cheap_model(
        self, v4_2_callback, mock_pool, mock_classifier_for_v4_2, real_routing_table
    ):
        """Low score (0.1) routes to a low-cost model (local-7b or deepseek-v3)."""
        data = {
            "messages": [{"role": "user", "content": self.PROMPT}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, self.PROMPT)

        mock_classifier_for_v4_2.aclassify.return_value = ClassifierResult(
            score=0.1, classifier_type="mf", latency_ms=3.0
        )

        await v4_2_callback.async_pre_call_hook({}, None, data, "completion")

        # Low score should route to local-7b (cheapest) based on overrides [0.0, 0.18]
        assert data["model"] == "ollama/qwen2-7b"
        assert data["metadata"]["route_score"] == 0.1

    # --- Score-dependent routing: high score ---

    async def test_high_score_routes_to_capable_model(
        self, v4_2_callback, mock_pool, mock_classifier_for_v4_2, valid_model_names
    ):
        """High score (0.9) routes to a high-capability model."""
        data = {
            "messages": [{"role": "user", "content": self.PROMPT}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, self.PROMPT)

        mock_classifier_for_v4_2.aclassify.return_value = ClassifierResult(
            score=0.9, classifier_type="mf", latency_ms=4.0
        )

        await v4_2_callback.async_pre_call_hook({}, None, data, "completion")

        # High score should route to a high-capability model (o1 or gpt-4o)
        assert data["model"] in valid_model_names
        assert data["metadata"]["route_score"] == 0.9
        # Should NOT be the cheapest model
        assert data["model"] != "ollama/qwen2-7b"

    # --- Metadata completeness check ---

    async def test_metadata_fully_populated(
        self, v4_2_callback, mock_pool, mock_classifier_for_v4_2
    ):
        """After routing, metadata contains route_score, target_model, and route_reason."""
        data = {
            "messages": [{"role": "user", "content": self.PROMPT}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        self._setup_pool_passthrough(mock_pool, self.PROMPT)

        mock_classifier_for_v4_2.aclassify.return_value = ClassifierResult(
            score=0.6, classifier_type="mf", latency_ms=6.0
        )

        await v4_2_callback.async_pre_call_hook({}, None, data, "completion")

        # Verify all expected metadata keys are present
        assert "route_score" in data["metadata"]
        assert "target_model" in data["metadata"]
        assert "route_reason" in data["metadata"]
        assert "latency_route_ms" in data["metadata"]
        assert data["metadata"]["route_score"] == 0.6
        assert data["metadata"]["target_model"] == data["model"]

    # --- Different scores produce different models ---

    async def test_different_scores_may_select_different_models(
        self, mock_pool, real_rule_engine, real_resolver, valid_model_names
    ):
        """Different classifier scores route to (potentially) different models,
        demonstrating the scoring pipeline works end-to-end."""
        scores = [0.1, 0.5, 0.9]
        selected_models = []

        for score in scores:
            classifier = MagicMock(spec=ModelClassifier)
            classifier.aclassify = AsyncMock(
                return_value=ClassifierResult(
                    score=score, classifier_type="mf", latency_ms=3.0
                )
            )

            cb = SmartRouterCallback(
                pool=mock_pool,
                enable_routing=True,
                rule_engine=real_rule_engine,
                classifier=classifier,
            )
            cb._route_resolver = real_resolver
            cb._routing_config = RoutingConfig(
                score_input="masked",
                trivial=TrivialConfig(
                    enabled=True,
                    max_length=30,
                    patterns_file="./patterns/trivial_chat.txt",
                    target_model="local-7b",
                ),
                classifier=ClassifierConfig(type="mf"),
                overlap_strategy="lowest_cost",
                fallback_model="deepseek-v3",
                session_policy="per_turn",
            )

            data = {
                "messages": [{"role": "user", "content": self.PROMPT}],
                "model": "gpt-4o",
                "metadata": {"session_id": "s1", "request_id": "r1"},
            }
            mock_pool.call.side_effect = [
                {"passed": True, "violations": [], "mode": "strict"},
                {"masked_text": self.PROMPT, "entities_found": []},
            ]

            await cb.async_pre_call_hook({}, None, data, "completion")

            assert data["model"] in valid_model_names
            selected_models.append(data["model"])

        # At least 2 different models should be selected for scores 0.1, 0.5, 0.9
        assert len(set(selected_models)) >= 2, (
            f"Expected at least 2 different models for scores {scores}, "
            f"but got: {selected_models}"
        )
