"""Verification Checkpoint V2-9: failover 链在 LLM 错误时触发，不修改全局方案。

Tests verify:
1. When a known agent routes to model X and model X fails (triggering async_log_failure_event),
   the next failover model is selected from the chain.
2. After the failover event, plan_store.get_model(agent) still returns original model X
   (global plan unchanged).
3. The failover metadata (_failover_model, _failover_from, _failover_index) is correctly
   set in the request metadata.

Requirements: FR-5.1 (Agent 级重试), FR-5.2 (仅影响当次请求)
Design: Property 11 (Failover 隔离), Property 12 (Failover 链遍历)
"""

from __future__ import annotations

import copy
import logging

import pytest
from unittest.mock import patch

from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.agent_plan_store import AgentPlanStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAILOVER_CHAINS = {
    "deepseek-v4-pro": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
    "gpt-5.5": ["deepseek-v4-pro", "claude-sonnet"],
    "codex-mini": ["deepseek-v4-pro"],
}


@pytest.fixture
def plan_store():
    """Create an AgentPlanStore with test data."""
    store = AgentPlanStore()
    store.set_model("intent_classifier", "deepseek-v4-pro")
    store.set_model("document_parser", "gpt-5.5")
    store.set_model("code_assistant", "codex-mini")
    store.set_model("reasoning_engine", "gpt-5.5")
    store.set_model("general_assistant", "deepseek-v4-pro")
    return store


@pytest.fixture
def router(plan_store):
    """Create an AgentWorkbuddyCallback with failover chains enabled."""
    with patch.object(ClawVaultPool, "__init__", return_value=None):
        return AgentWorkbuddyCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            failover_chains=FAILOVER_CHAINS,
            failover_enabled=True,
        )


# ---------------------------------------------------------------------------
# Test: Failover 链在 LLM 错误时正确触发
# ---------------------------------------------------------------------------


class TestFailoverTriggersOnLLMError:
    """V2-9: failover 链在 LLM 错误时触发。

    验证 async_log_failure_event 在 LLM 调用失败时正确选择下一个模型。
    需求: FR-5.1 (Agent 级重试)
    """

    async def test_first_failure_selects_first_failover_model(self, router):
        """第一次 LLM 失败 → 选择 failover 链中第一个模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "gpt-5.5"
        assert metadata["_failover_from"] == "deepseek-v4-pro"
        assert metadata["_failover_index"] == 1

    async def test_second_failure_selects_second_failover_model(self, router):
        """第二次 LLM 失败 → 选择 failover 链中第二个模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 1,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "gpt-5.5", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "claude-sonnet"
        assert metadata["_failover_from"] == "gpt-5.5"
        assert metadata["_failover_index"] == 2

    async def test_third_failure_selects_third_failover_model(self, router):
        """第三次 LLM 失败 → 选择 failover 链中第三个模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 2,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "claude-sonnet", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "gpt-4o"
        assert metadata["_failover_from"] == "claude-sonnet"
        assert metadata["_failover_index"] == 3

    async def test_failover_without_metadata_chain_uses_config(self, router):
        """metadata 中无 failover 链信息 → 从配置中查找并触发。"""
        metadata = {}
        kwargs = {"model": "codex-mini", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "deepseek-v4-pro"
        assert metadata["_failover_from"] == "codex-mini"
        assert metadata["_failover_index"] == 1

    async def test_chain_exhausted_does_not_set_failover_model(self, router):
        """Failover 链耗尽 → 不设置 _failover_model。"""
        metadata = {
            "_failover_chain": ["deepseek-v4-pro"],
            "_failover_index": 1,  # 已超过链长度
            "_original_model": "codex-mini",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert "_failover_model" not in metadata

    async def test_chain_exhausted_logs_error(self, router, caplog):
        """Failover 链耗尽 → 记录 AGENT_FAILOVER_EXHAUSTED 错误日志。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet"],
            "_failover_index": 2,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "claude-sonnet", "metadata": metadata}

        with caplog.at_level(logging.ERROR):
            await router.async_log_failure_event(kwargs, None, None, None)

        assert "AGENT_FAILOVER_EXHAUSTED" in caplog.text


# ---------------------------------------------------------------------------
# Test: Failover 不修改全局 AgentPlanStore (FR-5.2)
# ---------------------------------------------------------------------------


class TestFailoverDoesNotModifyGlobalPlan:
    """V2-9: failover 不修改全局方案。

    验证 async_log_failure_event 执行后，AgentPlanStore 中所有映射保持不变。
    需求: FR-5.2 (仅影响当次请求 — failover 不修改全局 AgentPlanStore)
    """

    async def test_plan_store_unchanged_after_single_failover(self, router, plan_store):
        """单次 failover 后 plan_store.get_model() 返回原始值。"""
        # 记录 failover 前的完整方案
        original_plans = plan_store.get_all_plans()

        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        # 验证全局方案完全不变
        assert plan_store.get_all_plans() == original_plans
        assert plan_store.get_model("intent_classifier") == "deepseek-v4-pro"
        assert plan_store.get_model("document_parser") == "gpt-5.5"
        assert plan_store.get_model("code_assistant") == "codex-mini"

    async def test_plan_store_unchanged_after_multiple_failovers(
        self, router, plan_store
    ):
        """多次连续 failover 后 plan_store 仍保持不变。"""
        original_plans = plan_store.get_all_plans()

        # 模拟 intent_classifier 的三次连续失败
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }

        # First failure
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}
        await router.async_log_failure_event(kwargs, None, None, None)

        # Second failure
        kwargs = {"model": "gpt-5.5", "metadata": metadata}
        await router.async_log_failure_event(kwargs, None, None, None)

        # Third failure
        kwargs = {"model": "claude-sonnet", "metadata": metadata}
        await router.async_log_failure_event(kwargs, None, None, None)

        # 全局方案完全不变
        assert plan_store.get_all_plans() == original_plans
        assert plan_store.get_model("intent_classifier") == "deepseek-v4-pro"

    async def test_plan_store_unchanged_after_chain_exhausted(
        self, router, plan_store
    ):
        """Failover 链完全耗尽后 plan_store 仍不变。"""
        original_plans = plan_store.get_all_plans()

        metadata = {
            "_failover_chain": ["deepseek-v4-pro"],
            "_failover_index": 1,
            "_original_model": "codex-mini",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert plan_store.get_all_plans() == original_plans
        assert plan_store.get_model("code_assistant") == "codex-mini"

    async def test_different_agents_failover_independently(self, router, plan_store):
        """不同 agent 各自的 failover 互不影响全局方案。"""
        original_plans = plan_store.get_all_plans()

        # Agent 1 (intent_classifier → deepseek-v4-pro) 失败
        meta1 = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        await router.async_log_failure_event(
            {"model": "deepseek-v4-pro", "metadata": meta1}, None, None, None
        )

        # Agent 2 (document_parser → gpt-5.5) 失败
        meta2 = {
            "_failover_chain": ["deepseek-v4-pro", "claude-sonnet"],
            "_failover_index": 0,
            "_original_model": "gpt-5.5",
        }
        await router.async_log_failure_event(
            {"model": "gpt-5.5", "metadata": meta2}, None, None, None
        )

        # Agent 3 (code_assistant → codex-mini) 失败
        meta3 = {
            "_failover_chain": ["deepseek-v4-pro"],
            "_failover_index": 0,
            "_original_model": "codex-mini",
        }
        await router.async_log_failure_event(
            {"model": "codex-mini", "metadata": meta3}, None, None, None
        )

        # 全局方案完全不变 — 所有 agent 的映射保持原样
        assert plan_store.get_all_plans() == original_plans
        assert plan_store.get_model("intent_classifier") == "deepseek-v4-pro"
        assert plan_store.get_model("document_parser") == "gpt-5.5"
        assert plan_store.get_model("code_assistant") == "codex-mini"
        assert plan_store.get_model("reasoning_engine") == "gpt-5.5"
        assert plan_store.get_model("general_assistant") == "deepseek-v4-pro"

    async def test_plan_store_identity_unchanged(self, router, plan_store):
        """plan_store 对象引用本身不变（非替换）。"""
        store_id_before = id(router.plan_store)

        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert id(router.plan_store) == store_id_before


# ---------------------------------------------------------------------------
# Test: Failover metadata 正确设置
# ---------------------------------------------------------------------------


class TestFailoverMetadataCorrectness:
    """V2-9: failover metadata 正确设置。

    验证 async_log_failure_event 正确设置 _failover_model、_failover_from、
    _failover_index 等 metadata 字段。
    需求: FR-5.1
    """

    async def test_failover_model_is_next_in_chain(self, router):
        """_failover_model 为 failover 链中下一个可用模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_model"] == "gpt-5.5"

    async def test_failover_from_records_failed_model(self, router):
        """_failover_from 记录本次失败的模型名称。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)

        assert metadata["_failover_from"] == "deepseek-v4-pro"

    async def test_failover_index_increments(self, router):
        """_failover_index 在每次失败后正确递增。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(kwargs, None, None, None)
        assert metadata["_failover_index"] == 1

        kwargs = {"model": "gpt-5.5", "metadata": metadata}
        await router.async_log_failure_event(kwargs, None, None, None)
        assert metadata["_failover_index"] == 2

        kwargs = {"model": "claude-sonnet", "metadata": metadata}
        await router.async_log_failure_event(kwargs, None, None, None)
        assert metadata["_failover_index"] == 3

    async def test_original_model_preserved_across_failovers(self, router):
        """_original_model 在多次 failover 中始终保持为首次分配的模型。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }

        # 连续三次失败
        for model in ["deepseek-v4-pro", "gpt-5.5", "claude-sonnet"]:
            kwargs = {"model": model, "metadata": metadata}
            await router.async_log_failure_event(kwargs, None, None, None)

        # _original_model 始终不变
        assert metadata["_original_model"] == "deepseek-v4-pro"

    async def test_failover_logs_agent_failover_warning(self, router, caplog):
        """Failover 触发时记录 AGENT_FAILOVER 警告日志。"""
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        with caplog.at_level(logging.WARNING):
            await router.async_log_failure_event(kwargs, None, None, None)

        assert "AGENT_FAILOVER" in caplog.text
        assert "deepseek-v4-pro" in caplog.text
        assert "gpt-5.5" in caplog.text


# ---------------------------------------------------------------------------
# Test: End-to-end — 路由 + 失败 + failover 全流程
# ---------------------------------------------------------------------------


class TestEndToEndRoutingFailover:
    """V2-9 端到端: 路由分配 → LLM 失败 → failover → 全局方案不变。

    模拟完整流程：agent 路由到主模型，主模型失败触发 failover，
    验证全局方案不变且 failover metadata 正确。
    """

    async def test_full_flow_routing_then_failover(self, router, plan_store):
        """完整流程: _execute_routing → async_log_failure_event → 验证。"""
        # 记录原始方案
        original_plans = plan_store.get_all_plans()

        # Step 1: 正常路由 — intent_classifier → deepseek-v4-pro
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请分类这个意图",
                    "agent": "intent_classifier",
                }
            ],
            "metadata": {},
        }
        await router._execute_routing(data, "masked", "original", "abc123")

        assert data["model"] == "deepseek-v4-pro"
        assert data["metadata"]["_failover_chain"] == ["gpt-5.5", "claude-sonnet", "gpt-4o"]
        assert data["metadata"]["_failover_index"] == 0
        assert data["metadata"]["_original_model"] == "deepseek-v4-pro"

        # Step 2: LLM 调用失败 — 触发 failover
        kwargs = {"model": "deepseek-v4-pro", "metadata": data["metadata"]}
        await router.async_log_failure_event(kwargs, None, None, None)

        # Step 3: 验证 failover metadata
        assert data["metadata"]["_failover_model"] == "gpt-5.5"
        assert data["metadata"]["_failover_from"] == "deepseek-v4-pro"
        assert data["metadata"]["_failover_index"] == 1

        # Step 4: 验证全局方案完全不变
        assert plan_store.get_all_plans() == original_plans
        assert plan_store.get_model("intent_classifier") == "deepseek-v4-pro"
        assert plan_store.get_model("document_parser") == "gpt-5.5"
        assert plan_store.get_model("code_assistant") == "codex-mini"

    async def test_full_flow_multiple_agents_failover_isolated(
        self, router, plan_store
    ):
        """多个 agent 各自经历 failover，彼此隔离且全局方案不变。"""
        original_plans = plan_store.get_all_plans()

        # Agent 1: intent_classifier → deepseek-v4-pro → 失败 → gpt-5.5
        data1 = {
            "messages": [
                {"role": "user", "content": "意图分类", "agent": "intent_classifier"}
            ],
            "metadata": {},
        }
        await router._execute_routing(data1, "masked", "original", "hash1")
        await router.async_log_failure_event(
            {"model": "deepseek-v4-pro", "metadata": data1["metadata"]},
            None, None, None,
        )
        assert data1["metadata"]["_failover_model"] == "gpt-5.5"

        # Agent 2: document_parser → gpt-5.5 → 失败 → deepseek-v4-pro
        data2 = {
            "messages": [
                {"role": "user", "content": "解析文档", "agent": "document_parser"}
            ],
            "metadata": {},
        }
        await router._execute_routing(data2, "masked", "original", "hash2")
        await router.async_log_failure_event(
            {"model": "gpt-5.5", "metadata": data2["metadata"]},
            None, None, None,
        )
        assert data2["metadata"]["_failover_model"] == "deepseek-v4-pro"

        # 全局方案完全不变
        assert plan_store.get_all_plans() == original_plans
        assert plan_store.get_model("intent_classifier") == "deepseek-v4-pro"
        assert plan_store.get_model("document_parser") == "gpt-5.5"

    async def test_get_next_failover_model_helper(self, router):
        """get_next_failover_model 辅助方法正确返回下一个模型。"""
        # 无 metadata — 从配置中查找
        result = router.get_next_failover_model("deepseek-v4-pro")
        assert result == "gpt-5.5"

        # 带 metadata index
        metadata = {
            "_failover_chain": ["gpt-5.5", "claude-sonnet", "gpt-4o"],
            "_failover_index": 1,
        }
        result = router.get_next_failover_model("deepseek-v4-pro", metadata)
        assert result == "claude-sonnet"

        # 链耗尽
        metadata["_failover_index"] = 3
        result = router.get_next_failover_model("deepseek-v4-pro", metadata)
        assert result is None

        # 未知模型无链
        result = router.get_next_failover_model("unknown-model-xyz")
        assert result is None
