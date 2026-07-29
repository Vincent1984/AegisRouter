"""Tests for TransactionRouterCallback Failover Integration.

Tests cover FR-6.1 ~ FR-6.4:
- TC-TXN-FAILOVER-001: Failover selects next model from chain on failure
- TC-TXN-FAILOVER-002: Global plan store is unchanged after failover
- TC-TXN-FAILOVER-003: Failover logs WARNING with AGENT_FAILOVER
- TC-TXN-FAILOVER-004: If entire chain exhausted, appropriate error logged
- TC-TXN-FAILOVER-005: Failover chain stored in metadata during routing
- TC-TXN-FAILOVER-006: Failover disabled → no chain processing
- TC-TXN-FAILOVER-007: Model without failover chain → no failover
"""

from __future__ import annotations

import logging

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

FAILOVER_CHAINS = {
    "gpt-5.5": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
    "gemini-2.5-pro": ["gemini-3.1-pro", "claude-sonnet", "deepseek-v4-pro"],
    "codex-mini": ["deepseek-v4-pro", "gpt-5.4-mini"],
    "local-7b": ["deepseek-v4-pro"],
}


@pytest.fixture
def plan_store():
    """Create a RoutingPlanStore with test data."""
    store = RoutingPlanStore()
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("resume_screening", "skill_matcher", "gpt-5.5")
    store.set_model("code_review", "code_analyzer", "codex-mini")
    store.set_model("code_review", "issue_detector", "gpt-5.5")
    return store


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool that simulates successful PII masking."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.max_connections = 10

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            return {
                "masked_text": params["text"],
                "entities_found": [],
            }
        elif method == "restore":
            return {"restored_text": params["text"]}
        elif method == "get_mapping":
            return {"mapping": {}}
        return None

    pool.call = AsyncMock(side_effect=mock_call)
    return pool


@pytest.fixture
def router(plan_store, mock_pool):
    """Create a TransactionRouterCallback with failover chains."""
    return TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        failover_chains=FAILOVER_CHAINS,
        failover_enabled=True,
        pool=mock_pool,
    )


@pytest.fixture
def router_no_failover(plan_store, mock_pool):
    """Create a TransactionRouterCallback with failover disabled."""
    return TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        failover_chains=FAILOVER_CHAINS,
        failover_enabled=False,
        pool=mock_pool,
    )


# ---------------------------------------------------------------------------
# TC-TXN-FAILOVER-001: Failover selects next model from chain on failure
# ---------------------------------------------------------------------------


class TestFailoverSelectsNextModel:
    """TC-TXN-FAILOVER-001: Failover selects next model from chain."""

    @pytest.mark.asyncio
    async def test_first_failure_selects_first_in_chain(self, router):
        """第一次失败选择 failover 链中第一个模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "_failover_index": 0,
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "gpt-5.5", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "gpt-5.2"
        assert metadata["_failover_from"] == "gpt-5.5"
        assert metadata["_failover_index"] == 1

    @pytest.mark.asyncio
    async def test_second_failure_selects_second_in_chain(self, router):
        """第二次失败选择 failover 链中第二个模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "_failover_index": 1,
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "gpt-5.2", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "claude-sonnet"
        assert metadata["_failover_from"] == "gpt-5.2"
        assert metadata["_failover_index"] == 2

    @pytest.mark.asyncio
    async def test_third_failure_selects_third_in_chain(self, router):
        """第三次失败选择 failover 链中第三个模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "_failover_index": 2,
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "claude-sonnet", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "deepseek-v4-pro"
        assert metadata["_failover_from"] == "claude-sonnet"
        assert metadata["_failover_index"] == 3

    @pytest.mark.asyncio
    async def test_failover_without_metadata_uses_config_chains(self, router):
        """没有 metadata 中的链信息时，从配置中查找。"""
        metadata = {}
        kwargs = {"model": "codex-mini", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "deepseek-v4-pro"
        assert metadata["_failover_from"] == "codex-mini"
        assert metadata["_failover_index"] == 1


# ---------------------------------------------------------------------------
# TC-TXN-FAILOVER-002: Global plan store unchanged after failover
# ---------------------------------------------------------------------------


class TestFailoverDoesNotModifyPlanStore:
    """TC-TXN-FAILOVER-002: Global plan store is unchanged after failover."""

    @pytest.mark.asyncio
    async def test_plan_store_unchanged_after_single_failover(self, router, plan_store):
        """单次 failover 后，plan_store 保持不变。"""
        # 记录原始方案
        original_plans = plan_store.get_all_plans()

        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet"],
            "_failover_index": 0,
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "gpt-5.5", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        # 验证 plan_store 未被修改
        assert plan_store.get_all_plans() == original_plans
        assert plan_store.get_model("resume_screening", "skill_matcher") == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_plan_store_unchanged_after_multiple_failovers(
        self, router, plan_store
    ):
        """多次 failover 后，plan_store 仍保持不变。"""
        original_plans = plan_store.get_all_plans()

        # 模拟三次连续失败
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "_failover_index": 0,
            "_original_model": "gpt-5.5",
        }

        for i in range(3):
            kwargs = {"model": f"model-{i}", "metadata": metadata}
            await router.async_log_failure_event(kwargs, None, None, None)

        # plan_store 完全不变
        assert plan_store.get_all_plans() == original_plans

    @pytest.mark.asyncio
    async def test_plan_store_unchanged_after_chain_exhausted(
        self, router, plan_store
    ):
        """failover 链耗尽后，plan_store 仍不变。"""
        original_plans = plan_store.get_all_plans()

        metadata = {
            "_failover_chain": ["deepseek-v4-pro"],
            "_failover_index": 1,  # 已耗尽
            "_original_model": "local-7b",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert plan_store.get_all_plans() == original_plans


# ---------------------------------------------------------------------------
# TC-TXN-FAILOVER-003: Failover logs WARNING with AGENT_FAILOVER
# ---------------------------------------------------------------------------


class TestFailoverLogsWarning:
    """TC-TXN-FAILOVER-003: Failover logs WARNING with AGENT_FAILOVER."""

    @pytest.mark.asyncio
    async def test_failover_logs_agent_failover_warning(self, router, caplog):
        """Failover 时记录包含 AGENT_FAILOVER 的 WARNING 日志。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet"],
            "_failover_index": 0,
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "gpt-5.5", "metadata": metadata}

        with caplog.at_level(logging.WARNING):
            await router.async_log_failure_event(kwargs, None, None, None)

        assert "AGENT_FAILOVER" in caplog.text

    @pytest.mark.asyncio
    async def test_failover_log_contains_model_names(self, router, caplog):
        """Failover 日志包含原始模型和下一模型名称。"""
        metadata = {
            "_failover_chain": ["claude-sonnet", "deepseek-v4-pro"],
            "_failover_index": 0,
            "_original_model": "gemini-2.5-pro",
        }
        kwargs = {"model": "gemini-2.5-pro", "metadata": metadata}

        with caplog.at_level(logging.WARNING):
            await router.async_log_failure_event(kwargs, None, None, None)

        assert "gemini-2.5-pro" in caplog.text
        assert "claude-sonnet" in caplog.text

    @pytest.mark.asyncio
    async def test_failover_log_contains_chain_index(self, router, caplog):
        """Failover 日志包含链索引信息。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "_failover_index": 1,
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "gpt-5.2", "metadata": metadata}

        with caplog.at_level(logging.WARNING):
            await router.async_log_failure_event(kwargs, None, None, None)

        # Should log chain_index=2/3
        assert "2/3" in caplog.text


# ---------------------------------------------------------------------------
# TC-TXN-FAILOVER-004: Chain exhausted → appropriate error logged
# ---------------------------------------------------------------------------


class TestFailoverChainExhausted:
    """TC-TXN-FAILOVER-004: Chain exhausted → appropriate error logged."""

    @pytest.mark.asyncio
    async def test_exhausted_chain_logs_error(self, router, caplog):
        """Failover 链耗尽时记录 ERROR 日志。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet"],
            "_failover_index": 2,  # 已超出链长度
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "claude-sonnet", "metadata": metadata}

        with caplog.at_level(logging.ERROR):
            await router.async_log_failure_event(kwargs, None, None, None)

        assert "AGENT_FAILOVER_EXHAUSTED" in caplog.text

    @pytest.mark.asyncio
    async def test_exhausted_chain_does_not_set_failover_model(self, router):
        """链耗尽时不设置 _failover_model。"""
        metadata = {
            "_failover_chain": ["deepseek-v4-pro"],
            "_failover_index": 1,
            "_original_model": "local-7b",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert "_failover_model" not in metadata


# ---------------------------------------------------------------------------
# TC-TXN-FAILOVER-005: Failover chain stored in metadata during routing
# ---------------------------------------------------------------------------


class TestFailoverChainInMetadata:
    """TC-TXN-FAILOVER-005: Failover chain stored in metadata during routing."""

    @pytest.mark.asyncio
    async def test_routing_stores_failover_chain_in_metadata(self, router):
        """路由时将 failover 链存入 metadata。"""
        data = {
            "messages": [{"role": "user", "content": "Test routing."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        # gpt-5.5 has a failover chain
        assert data["metadata"]["_failover_chain"] == [
            "gpt-5.2", "claude-sonnet", "deepseek-v4-pro"
        ]
        assert data["metadata"]["_failover_index"] == 0
        assert data["metadata"]["_original_model"] == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_routing_no_chain_for_model_without_failover(self, router):
        """没有 failover 链的模型不存储链信息。"""
        # Add a model that's not in failover chains
        router.plan_store.set_model("resume_screening", "special_agent", "unknown-model")

        data = {
            "messages": [{"role": "user", "content": "Test routing."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "special_agent",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert "_failover_chain" not in data["metadata"]


# ---------------------------------------------------------------------------
# TC-TXN-FAILOVER-006: Failover disabled → no chain processing
# ---------------------------------------------------------------------------


class TestFailoverDisabled:
    """TC-TXN-FAILOVER-006: Failover disabled → no chain processing."""

    @pytest.mark.asyncio
    async def test_disabled_failover_no_chain_in_metadata(self, router_no_failover):
        """Failover 禁用时，routing 不存储链信息。"""
        data = {
            "messages": [{"role": "user", "content": "Test."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                }
            },
        }

        await router_no_failover.async_pre_call_hook({}, None, data, "completion")

        assert "_failover_chain" not in data["metadata"]
        # Model is still correctly assigned
        assert data["model"] == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_disabled_failover_no_next_model_on_failure(self, router_no_failover):
        """Failover 禁用时，失败不选择下一模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.2"],
            "_failover_index": 0,
            "_original_model": "gpt-5.5",
        }
        kwargs = {"model": "gpt-5.5", "metadata": metadata}

        await router_no_failover.async_log_failure_event(kwargs, None, None, None)

        assert "_failover_model" not in metadata


# ---------------------------------------------------------------------------
# TC-TXN-FAILOVER-007: Model without failover chain → no failover
# ---------------------------------------------------------------------------


class TestNoChainForModel:
    """TC-TXN-FAILOVER-007: Model without failover chain → no failover."""

    @pytest.mark.asyncio
    async def test_model_without_chain_no_failover(self, router, caplog):
        """没有 failover 链的模型失败时不执行 failover。"""
        metadata = {}
        kwargs = {"model": "unknown-model-xyz", "metadata": metadata}

        with caplog.at_level(logging.WARNING):
            await router.async_log_failure_event(kwargs, None, None, None)

        assert "_failover_model" not in metadata
        assert "无 failover 链可用" in caplog.text


# ---------------------------------------------------------------------------
# Test: get_next_failover_model helper
# ---------------------------------------------------------------------------


class TestGetNextFailoverModel:
    """Tests for get_next_failover_model helper method."""

    def test_returns_first_model_with_no_metadata(self, router):
        """无 metadata 时返回链中第一个模型。"""
        result = router.get_next_failover_model("gpt-5.5")
        assert result == "gpt-5.2"

    def test_returns_model_at_index(self, router):
        """根据 metadata 中的 index 返回对应模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "_failover_index": 1,
        }
        result = router.get_next_failover_model("gpt-5.5", metadata)
        assert result == "claude-sonnet"

    def test_returns_none_when_chain_exhausted(self, router):
        """链耗尽时返回 None。"""
        metadata = {
            "_failover_chain": ["gpt-5.2", "claude-sonnet"],
            "_failover_index": 2,
        }
        result = router.get_next_failover_model("gpt-5.5", metadata)
        assert result is None

    def test_returns_none_for_unknown_model(self, router):
        """未知模型返回 None。"""
        result = router.get_next_failover_model("unknown-model")
        assert result is None

    def test_returns_none_when_disabled(self, router_no_failover):
        """Failover 禁用时返回 None。"""
        result = router_no_failover.get_next_failover_model("gpt-5.5")
        assert result is None


# ---------------------------------------------------------------------------
# Test: FailoverConfig in config.py
# ---------------------------------------------------------------------------


class TestFailoverConfig:
    """Tests for FailoverConfig pydantic model."""

    def test_failover_config_loads_from_route_config(self):
        """验证 FailoverConfig 从 route_config.yaml 正确解析。"""
        from aegis_router.config import FailoverConfig

        config = FailoverConfig(
            enabled=True,
            timeout_ms=50,
            chains={
                "gpt-5.5": ["gpt-5.2", "claude-sonnet"],
                "local-7b": ["deepseek-v4-pro"],
            },
        )
        assert config.enabled is True
        assert config.timeout_ms == 50
        assert config.chains["gpt-5.5"] == ["gpt-5.2", "claude-sonnet"]
        assert config.chains["local-7b"] == ["deepseek-v4-pro"]

    def test_failover_config_defaults(self):
        """FailoverConfig 默认值正确。"""
        from aegis_router.config import FailoverConfig

        config = FailoverConfig()
        assert config.enabled is True
        assert config.timeout_ms == 50
        assert config.chains == {}

    def test_aegis_config_includes_failover(self):
        """AegisConfig 包含 failover 字段。"""
        from aegis_router.config import AegisConfig, FailoverConfig

        config = AegisConfig(
            failover=FailoverConfig(
                enabled=True,
                chains={"model-a": ["model-b"]},
            )
        )
        assert config.failover.enabled is True
        assert config.failover.chains == {"model-a": ["model-b"]}
