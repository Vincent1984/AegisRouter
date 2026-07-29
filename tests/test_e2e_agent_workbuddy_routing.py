"""E2E Integration Tests — Agent-WorkBuddy Routing with Mock LLM.

Tests cover:
- TC-E2E-WB-001: 完整启动 → 方案生成 → 请求分发 → 验证模型正确
- TC-E2E-WB-002: WorkBuddy 请求格式（user 消息含 agent 字段）→ 正确路由
- TC-E2E-WB-003: 响应包含正确 aegis_metadata
- TC-E2E-WB-004: 配置变更 → 方案重算 → 新请求使用新方案
- TC-E2E-WB-005: 插件切换 transaction ↔ agent_workbuddy 无副作用
- TC-E2E-WB-006: Failover 场景 — 主模型失败 → 自动切换，全局方案不变
- TC-E2E-WB-007: PII 脱敏 + agent_workbuddy 路由同时工作
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback
from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.agent_plan_store import AgentPlanStore
from aegis_router.router.agent_plan_generator import (
    AgentPlanGenerator,
    AgentWorkbuddyDef,
)
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Mock Response Classes
# ---------------------------------------------------------------------------


@dataclass
class MockMessage:
    """Mock LiteLLM message object."""
    content: str
    role: str = "assistant"


@dataclass
class MockChoice:
    """Mock LiteLLM choice object."""
    message: MockMessage
    index: int = 0


@dataclass
class MockResponse:
    """Mock LiteLLM ModelResponse object."""
    choices: list


def make_response(content: str) -> MockResponse:
    """Create a mock LiteLLM response with given content."""
    return MockResponse(choices=[MockChoice(message=MockMessage(content=content))])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_store():
    """Create an AgentPlanStore with comprehensive test data for E2E tests."""
    store = AgentPlanStore()
    store.set_model("resume_parser", "gemini-2.5-pro")
    store.set_model("intent_classifier", "local-7b")
    store.set_model("skill_matcher", "gpt-5.5")
    store.set_model("compliance_checker", "deepseek-v4-pro")
    store.set_model("code_analyzer", "codex-mini")
    store.set_model("report_generator", "claude-sonnet")
    return store


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool with passthrough PII behavior."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.max_connections = 10

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            return {"masked_text": params["text"], "entities_found": []}
        elif method == "restore":
            return {"restored_text": params["text"]}
        elif method == "get_mapping":
            return {"mapping": {}}
        return None

    pool.call = AsyncMock(side_effect=mock_call)
    return pool


@pytest.fixture
def router(plan_store, mock_pool):
    """Create an AgentWorkbuddyCallback for basic E2E tests."""
    return AgentWorkbuddyCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        pool=mock_pool,
    )


@pytest.fixture
def router_with_failover(plan_store, mock_pool):
    """Create an AgentWorkbuddyCallback with failover chains."""
    return AgentWorkbuddyCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        failover_chains={
            "gpt-5.5": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "gemini-2.5-pro": ["gemini-3.1-pro", "claude-sonnet"],
        },
        failover_enabled=True,
        pool=mock_pool,
    )


# ---------------------------------------------------------------------------
# TC-E2E-WB-001: 完整启动 → 方案生成 → 请求分发 → 验证模型正确
# ---------------------------------------------------------------------------


class TestE2EWB001_StartupPlanGenerationAndRouting:
    """TC-E2E-WB-001: 完整启动 → 方案生成 → 请求分发 → 验证模型正确。

    验证完整流程: AgentPlanGenerator 生成方案表 → AgentWorkbuddyCallback 使用方案
    → 请求正确路由到预计算模型。
    """

    @pytest.mark.asyncio
    async def test_plan_generation_and_routing(self, mock_pool):
        """AgentPlanGenerator 生成方案 → 路由使用生成的方案。"""
        # Mock CapabilityProfileManager
        mock_profile_manager = MagicMock()
        mock_profile = MagicMock()
        mock_profile.prefer_models = []
        mock_profile_manager.get_profile.return_value = mock_profile
        mock_profile_manager.profiles = {"medium": mock_profile}
        mock_profile_manager.select_best_model.return_value = "gpt-5.5"
        mock_profile_manager.score_model.return_value = 0.85

        # Define models pool
        models = [
            {"name": "gpt-5.5", "params": {"cost": 2.0}},
            {"name": "local-7b", "params": {"cost": 0.1}},
            {"name": "gemini-2.5-pro", "params": {"cost": 3.0}},
        ]

        # Define agents
        agents = [
            AgentWorkbuddyDef(
                name="task_planner",
                capability_profile="medium",
            ),
            AgentWorkbuddyDef(
                name="code_writer",
                capability_profile="medium",
                override_model="gemini-2.5-pro",
            ),
        ]

        # Generate plan
        generator = AgentPlanGenerator(
            profile_manager=mock_profile_manager,
            models=models,
            fallback_model="deepseek-v3",
            trigger_reason="startup",
        )
        store = generator.generate_all(agents)

        # Verify plan was generated correctly
        assert store.get_model("task_planner") == "gpt-5.5"
        assert store.get_model("code_writer") == "gemini-2.5-pro"  # override

        # Create router with generated store
        router = AgentWorkbuddyCallback(
            plan_store=store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        # Route a request using generated plan
        data = {
            "messages": [
                {"role": "user", "content": "Plan the task.", "agent": "task_planner"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-001",
                "request_id": "req-e2e-wb-001",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "gpt-5.5"
        assert data["metadata"]["target_model"] == "gpt-5.5"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"

    @pytest.mark.asyncio
    async def test_override_model_routes_correctly(self, mock_pool):
        """override_model Agent 路由到管理员指定的模型。"""
        mock_profile_manager = MagicMock()
        mock_profile = MagicMock()
        mock_profile.prefer_models = []
        mock_profile_manager.get_profile.return_value = mock_profile
        mock_profile_manager.profiles = {"medium": mock_profile}
        mock_profile_manager.select_best_model.return_value = "gpt-5.5"
        mock_profile_manager.score_model.return_value = 0.85

        models = [{"name": "gpt-5.5", "params": {}}, {"name": "codex-mini", "params": {}}]
        agents = [
            AgentWorkbuddyDef(
                name="code_writer",
                capability_profile="medium",
                override_model="codex-mini",
            ),
        ]

        generator = AgentPlanGenerator(
            profile_manager=mock_profile_manager,
            models=models,
            fallback_model="deepseek-v3",
        )
        store = generator.generate_all(agents)

        router = AgentWorkbuddyCallback(
            plan_store=store, fallback_model="deepseek-v3", pool=mock_pool
        )

        data = {
            "messages": [
                {"role": "user", "content": "Write code.", "agent": "code_writer"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-001b",
                "request_id": "req-e2e-wb-001b",
            },
        }
        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "codex-mini"
        assert data["metadata"]["route_reason"] == "plan"


# ---------------------------------------------------------------------------
# TC-E2E-WB-002: WorkBuddy 请求格式（user 消息含 agent 字段）→ 正确路由
# ---------------------------------------------------------------------------


class TestE2EWB002_WorkBuddyRequestFormat:
    """TC-E2E-WB-002: WorkBuddy 请求格式（user 消息含 agent 字段）→ 正确路由。

    验证 Agent-WorkBuddy 的核心路由机制：
    从最后一条 role=user 消息的 agent 字段提取 agent 名称 → 查表路由。
    """

    @pytest.mark.asyncio
    async def test_agent_in_user_message_routes_correctly(self, router):
        """user 消息含 agent 字段 → 正确路由到对应模型。"""
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Parse resume.", "agent": "resume_parser"},
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-002",
                "request_id": "req-e2e-wb-002",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "gemini-2.5-pro"
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["transaction_agent"] == "resume_parser"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"

    @pytest.mark.asyncio
    async def test_multiple_user_messages_uses_last(self, router):
        """多条 user 消息时，使用最后一条的 agent 字段。"""
        data = {
            "messages": [
                {"role": "user", "content": "First msg.", "agent": "intent_classifier"},
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": "Second msg.", "agent": "skill_matcher"},
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-002b",
                "request_id": "req-e2e-wb-002b",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        # Should use last user message's agent field
        assert data["model"] == "gpt-5.5"
        assert data["metadata"]["transaction_agent"] == "skill_matcher"

    @pytest.mark.asyncio
    async def test_fallback_to_metadata_agent(self, router):
        """user 消息无 agent → 降级到 metadata.agent。"""
        data = {
            "messages": [
                {"role": "user", "content": "Process this."},
            ],
            "metadata": {
                "agent": "code_analyzer",
                "session_id": "sess-e2e-wb-002c",
                "request_id": "req-e2e-wb-002c",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "codex-mini"
        assert data["metadata"]["transaction_agent"] == "code_analyzer"
        assert data["metadata"]["route_reason"] == "plan"

    @pytest.mark.asyncio
    async def test_no_agent_uses_fallback(self, router):
        """无 agent 字段 → 使用 fallback 模型 + NO_AGENT 警告。"""
        data = {
            "messages": [
                {"role": "user", "content": "Hello."},
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-002d",
                "request_id": "req-e2e-wb-002d",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert "NO_AGENT" in data["metadata"]["_routing_warnings"]

    @pytest.mark.asyncio
    async def test_invalid_agent_name_uses_fallback(self, router):
        """非法 agent 名称 → fallback + INVALID_AGENT 警告。"""
        data = {
            "messages": [
                {"role": "user", "content": "Test.", "agent": "invalid agent!@#"},
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-002e",
                "request_id": "req-e2e-wb-002e",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert "INVALID_AGENT" in data["metadata"]["_routing_warnings"]

    @pytest.mark.asyncio
    async def test_unknown_agent_uses_fallback(self, router):
        """Agent 不在方案表中 → fallback + UNKNOWN_AGENT 警告。"""
        data = {
            "messages": [
                {"role": "user", "content": "Test.", "agent": "nonexistent_agent"},
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-002f",
                "request_id": "req-e2e-wb-002f",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "unknown_agent"
        assert "UNKNOWN_AGENT" in data["metadata"]["_routing_warnings"]


# ---------------------------------------------------------------------------
# TC-E2E-WB-003: 响应包含正确 aegis_metadata
# ---------------------------------------------------------------------------


class TestE2EWB003_ResponseAegisMetadata:
    """TC-E2E-WB-003: 响应包含正确 aegis_metadata。

    验证 async_log_success_event 完成后，response 对象包含
    正确的 aegis_metadata 字段。
    """

    @pytest.mark.asyncio
    async def test_success_response_contains_aegis_metadata(self, router):
        """成功响应包含完整 aegis_metadata。"""
        data = {
            "messages": [
                {"role": "user", "content": "Analyze code.", "agent": "code_analyzer"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-003",
                "request_id": "req-e2e-wb-003",
            },
        }

        # Route
        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "codex-mini"

        # Mock LLM response
        llm_response = make_response("Code analysis complete. No issues found.")

        # Log success (triggers restore + aegis_metadata injection)
        kwargs = {"metadata": data["metadata"], "model": "codex-mini"}
        await router.async_log_success_event(kwargs, llm_response, None, None)

        # Verify aegis_metadata
        assert hasattr(llm_response, "aegis_metadata")
        meta = llm_response.aegis_metadata
        assert meta["agent"] == "code_analyzer"
        assert meta["assigned_model"] == "codex-mini"
        assert meta["routing_plugin"] == "agent_workbuddy"
        assert meta["template"] == ""
        assert meta["warnings"] == []

    @pytest.mark.asyncio
    async def test_aegis_metadata_with_unknown_agent_warning(self, router):
        """unknown agent 路由后，aegis_metadata 包含 UNKNOWN_AGENT 警告。"""
        data = {
            "messages": [
                {"role": "user", "content": "Test.", "agent": "ghost_agent"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-003b",
                "request_id": "req-e2e-wb-003b",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "deepseek-v3"

        llm_response = make_response("Fallback response.")
        kwargs = {"metadata": data["metadata"], "model": "deepseek-v3"}
        await router.async_log_success_event(kwargs, llm_response, None, None)

        assert hasattr(llm_response, "aegis_metadata")
        meta = llm_response.aegis_metadata
        assert meta["routing_plugin"] == "agent_workbuddy"
        assert meta["assigned_model"] == "deepseek-v3"
        assert "UNKNOWN_AGENT" in meta["warnings"]

    @pytest.mark.asyncio
    async def test_full_pipeline_route_and_response(self, router):
        """完整管道: 路由 → Mock LLM → 响应还原 → 客户端收到正确内容。"""
        data = {
            "messages": [
                {"role": "user", "content": "Generate report.", "agent": "report_generator"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-003c",
                "request_id": "req-e2e-wb-003c",
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "claude-sonnet"

        llm_response = make_response("Report generated successfully.")
        kwargs = {"metadata": data["metadata"], "model": "claude-sonnet"}
        await router.async_log_success_event(kwargs, llm_response, None, None)

        # Verify content is preserved (no PII, passthrough)
        assert llm_response.choices[0].message.content == "Report generated successfully."
        assert llm_response.aegis_metadata["agent"] == "report_generator"
        assert llm_response.aegis_metadata["assigned_model"] == "claude-sonnet"


# ---------------------------------------------------------------------------
# TC-E2E-WB-004: 配置变更 → 方案重算 → 新请求使用新方案
# ---------------------------------------------------------------------------


class TestE2EWB004_ConfigChangeAndPlanRecalculation:
    """TC-E2E-WB-004: 配置变更 → 方案重算 → 新请求使用新方案。

    验证原子替换 plan_store 后，后续请求使用新方案。
    """

    @pytest.mark.asyncio
    async def test_plan_store_atomic_replacement(self, router):
        """原子替换 plan_store → 新请求使用新方案。"""
        # Initial routing: resume_parser → gemini-2.5-pro
        data1 = {
            "messages": [
                {"role": "user", "content": "Parse.", "agent": "resume_parser"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-004",
                "request_id": "req-e2e-wb-004-1",
            },
        }
        await router.async_pre_call_hook({}, None, data1, "completion")
        assert data1["model"] == "gemini-2.5-pro"

        # Create new plan store with different mapping
        new_store = AgentPlanStore()
        new_store.set_model("resume_parser", "claude-sonnet")
        new_store.set_model("intent_classifier", "gpt-5.5")

        # Atomically replace plan store
        router.plan_store = new_store

        # Route same agent again → should use new model
        data2 = {
            "messages": [
                {"role": "user", "content": "Parse again.", "agent": "resume_parser"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-004",
                "request_id": "req-e2e-wb-004-2",
            },
        }
        await router.async_pre_call_hook({}, None, data2, "completion")
        assert data2["model"] == "claude-sonnet"

    @pytest.mark.asyncio
    async def test_old_agents_become_unknown_after_replacement(self, router):
        """替换后，旧方案中有但新方案中没有的 agent → UNKNOWN_AGENT。"""
        # Verify initial routing works
        data1 = {
            "messages": [
                {"role": "user", "content": "Analyze.", "agent": "code_analyzer"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-004b",
                "request_id": "req-e2e-wb-004b-1",
            },
        }
        await router.async_pre_call_hook({}, None, data1, "completion")
        assert data1["model"] == "codex-mini"

        # Replace with store that doesn't have code_analyzer
        new_store = AgentPlanStore()
        new_store.set_model("resume_parser", "gemini-2.5-pro")
        router.plan_store = new_store

        # code_analyzer is now unknown
        data2 = {
            "messages": [
                {"role": "user", "content": "Analyze.", "agent": "code_analyzer"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-004b",
                "request_id": "req-e2e-wb-004b-2",
            },
        }
        await router.async_pre_call_hook({}, None, data2, "completion")
        assert data2["model"] == "deepseek-v3"
        assert "UNKNOWN_AGENT" in data2["metadata"]["_routing_warnings"]


# ---------------------------------------------------------------------------
# TC-E2E-WB-005: 插件切换 transaction ↔ agent_workbuddy 无副作用
# ---------------------------------------------------------------------------


class TestE2EWB005_PluginSwitchNoSideEffects:
    """TC-E2E-WB-005: 插件切换 transaction ↔ agent_workbuddy 无副作用。

    验证两个插件可独立实例化，各自路由互不影响，无共享可变状态。
    """

    @pytest.mark.asyncio
    async def test_both_plugins_route_independently(self, mock_pool):
        """两个插件各自独立路由，使用各自的策略。"""
        # Setup agent_workbuddy plugin
        wb_store = AgentPlanStore()
        wb_store.set_model("analyzer", "gpt-5.5")

        wb_router = AgentWorkbuddyCallback(
            plan_store=wb_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        # Setup transaction plugin
        txn_store = RoutingPlanStore()
        txn_store.set_model("code_review", "analyzer", "codex-mini")

        txn_router = TransactionRouterCallback(
            plan_store=txn_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        # Route with agent_workbuddy (agent field in user message)
        wb_data = {
            "messages": [
                {"role": "user", "content": "Analyze.", "agent": "analyzer"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-005",
                "request_id": "req-e2e-wb-005-wb",
            },
        }
        await wb_router.async_pre_call_hook({}, None, wb_data, "completion")

        assert wb_data["model"] == "gpt-5.5"
        assert wb_data["metadata"]["routing_plugin"] == "agent_workbuddy"

        # Route with transaction (metadata.transaction)
        txn_data = {
            "messages": [
                {"role": "user", "content": "Analyze code."}
            ],
            "metadata": {
                "transaction": {"template": "code_review", "agent": "analyzer"},
                "session_id": "sess-e2e-wb-005",
                "request_id": "req-e2e-wb-005-txn",
            },
        }
        await txn_router.async_pre_call_hook({}, None, txn_data, "completion")

        assert txn_data["model"] == "codex-mini"
        assert txn_data["metadata"]["routing_plugin"] == "transaction"

    @pytest.mark.asyncio
    async def test_no_shared_mutable_state(self, mock_pool):
        """修改一个插件的 plan_store 不影响另一个插件。"""
        # Both plugins share the same agent name but with different models
        wb_store = AgentPlanStore()
        wb_store.set_model("shared_agent", "model-A")

        txn_store = RoutingPlanStore()
        txn_store.set_model("template_x", "shared_agent", "model-B")

        wb_router = AgentWorkbuddyCallback(
            plan_store=wb_store, fallback_model="deepseek-v3", pool=mock_pool
        )
        txn_router = TransactionRouterCallback(
            plan_store=txn_store, fallback_model="deepseek-v3", pool=mock_pool
        )

        # Modify wb_router's plan_store
        new_wb_store = AgentPlanStore()
        new_wb_store.set_model("shared_agent", "model-C")
        wb_router.plan_store = new_wb_store

        # txn_router is unaffected
        txn_data = {
            "messages": [{"role": "user", "content": "Test."}],
            "metadata": {
                "transaction": {"template": "template_x", "agent": "shared_agent"},
                "session_id": "sess-e2e-wb-005b",
                "request_id": "req-e2e-wb-005b",
            },
        }
        await txn_router.async_pre_call_hook({}, None, txn_data, "completion")
        assert txn_data["model"] == "model-B"  # Unchanged

        # wb_router uses new store
        wb_data = {
            "messages": [
                {"role": "user", "content": "Test.", "agent": "shared_agent"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-005b",
                "request_id": "req-e2e-wb-005b-2",
            },
        }
        await wb_router.async_pre_call_hook({}, None, wb_data, "completion")
        assert wb_data["model"] == "model-C"  # Updated


# ---------------------------------------------------------------------------
# TC-E2E-WB-006: Failover 场景 — 主模型失败 → 自动切换，全局方案不变
# ---------------------------------------------------------------------------


class TestE2EWB006_FailoverScenario:
    """TC-E2E-WB-006: Failover 场景 — 主模型失败 → 自动切换，全局方案不变。

    验证完整 failover 流程:
    1. pre_call_hook 路由到主模型，并注入 failover 链信息
    2. Mock LLM 调用失败
    3. async_log_failure_event 选择 failover 链中下一个模型
    4. 全局方案表不受影响
    """

    @pytest.mark.asyncio
    async def test_failover_complete_flow(self, router_with_failover, plan_store):
        """完整 failover 流程: 路由 → 失败 → 切换到 failover 模型。"""
        data = {
            "messages": [
                {"role": "user", "content": "Match skills.", "agent": "skill_matcher"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-006",
                "request_id": "req-e2e-wb-006",
            },
        }

        # Phase 1: Route to primary model (gpt-5.5)
        await router_with_failover.async_pre_call_hook(
            {}, None, data, "completion"
        )
        assert data["model"] == "gpt-5.5"
        assert data["metadata"]["target_model"] == "gpt-5.5"

        # Verify failover chain info was injected
        assert data["metadata"]["_failover_chain"] == [
            "gpt-5.2", "claude-sonnet", "deepseek-v4-pro"
        ]
        assert data["metadata"]["_failover_index"] == 0
        assert data["metadata"]["_original_model"] == "gpt-5.5"

        # Phase 2: Simulate LLM failure
        kwargs = {"model": "gpt-5.5", "metadata": data["metadata"]}
        await router_with_failover.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )

        # Phase 3: Verify failover selected next model
        assert data["metadata"]["_failover_model"] == "gpt-5.2"
        assert data["metadata"]["_failover_from"] == "gpt-5.5"
        assert data["metadata"]["_failover_index"] == 1

        # Phase 4: Verify global plan store is unchanged
        assert plan_store.get_model("skill_matcher") == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_failover_chain_progression(self, router_with_failover):
        """Failover 链连续失败: gpt-5.5 → gpt-5.2 → claude-sonnet → deepseek-v4-pro。"""
        data = {
            "messages": [
                {"role": "user", "content": "Process.", "agent": "skill_matcher"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-006-chain",
                "request_id": "req-e2e-wb-006-chain",
            },
        }

        await router_with_failover.async_pre_call_hook(
            {}, None, data, "completion"
        )
        assert data["model"] == "gpt-5.5"

        # First failure: gpt-5.5 → gpt-5.2
        kwargs = {"model": "gpt-5.5", "metadata": data["metadata"]}
        await router_with_failover.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert data["metadata"]["_failover_model"] == "gpt-5.2"
        assert data["metadata"]["_failover_index"] == 1

        # Second failure: gpt-5.2 → claude-sonnet
        kwargs["model"] = "gpt-5.2"
        await router_with_failover.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert data["metadata"]["_failover_model"] == "claude-sonnet"
        assert data["metadata"]["_failover_index"] == 2

        # Third failure: claude-sonnet → deepseek-v4-pro
        kwargs["model"] = "claude-sonnet"
        await router_with_failover.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert data["metadata"]["_failover_model"] == "deepseek-v4-pro"
        assert data["metadata"]["_failover_index"] == 3

    @pytest.mark.asyncio
    async def test_failover_does_not_affect_subsequent_requests(
        self, router_with_failover, plan_store
    ):
        """Failover 后，下一次同 agent 请求仍使用原计划模型。"""
        # First request: route → fail → failover
        data1 = {
            "messages": [
                {"role": "user", "content": "First.", "agent": "skill_matcher"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-006-next",
                "request_id": "req-e2e-wb-006-next-1",
            },
        }
        await router_with_failover.async_pre_call_hook(
            {}, None, data1, "completion"
        )
        assert data1["model"] == "gpt-5.5"

        # Trigger failover
        kwargs = {"model": "gpt-5.5", "metadata": data1["metadata"]}
        await router_with_failover.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )

        # Second request (new request): should still route to original model
        data2 = {
            "messages": [
                {"role": "user", "content": "Second.", "agent": "skill_matcher"}
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-006-next",
                "request_id": "req-e2e-wb-006-next-2",
            },
        }
        await router_with_failover.async_pre_call_hook(
            {}, None, data2, "completion"
        )

        # Still routes to original model, not failover
        assert data2["model"] == "gpt-5.5"
        assert plan_store.get_model("skill_matcher") == "gpt-5.5"


# ---------------------------------------------------------------------------
# TC-E2E-WB-007: PII 脱敏 + agent_workbuddy 路由同时工作
# ---------------------------------------------------------------------------


class TestE2EWB007_PIIMaskingWithAgentWorkbuddyRouting:
    """TC-E2E-WB-007: PII 脱敏 + agent_workbuddy 路由同时工作。

    验证含中文 PII 的请求经过完整管道:
    PII 脱敏 → agent_workbuddy 路由 → LLM 响应含占位符 → 还原为原始 PII。
    """

    @pytest.fixture
    def pii_mock_pool(self):
        """Create a mock pool that performs actual PII masking/restoration."""
        pool = MagicMock(spec=ClawVaultPool)
        pool.max_connections = 10

        async def mock_call(method, params):
            if method == "check_compliance":
                return {"passed": True}
            elif method == "mask":
                text = params["text"]
                masked = text.replace("张三", "[PERSON_1]").replace(
                    "13800138000", "[PHONE_1]"
                )
                entities = []
                if "张三" in text:
                    entities.append(
                        {"type": "PERSON", "start": 0, "end": 2, "score": 0.95}
                    )
                if "13800138000" in text:
                    entities.append(
                        {"type": "PHONE_NUMBER", "start": 0, "end": 11, "score": 0.99}
                    )
                return {"masked_text": masked, "entities_found": entities}
            elif method == "restore":
                text = params["text"]
                restored = text.replace("[PERSON_1]", "张三").replace(
                    "[PHONE_1]", "13800138000"
                )
                return {"restored_text": restored}
            elif method == "get_mapping":
                return {
                    "mapping": {
                        "[PERSON_1]": "张三",
                        "[PHONE_1]": "13800138000",
                    }
                }
            return None

        pool.call = AsyncMock(side_effect=mock_call)
        return pool

    @pytest.fixture
    def pii_router(self, pii_mock_pool):
        """Create an agent_workbuddy router with PII masking pool."""
        store = AgentPlanStore()
        store.set_model("resume_parser", "gemini-2.5-pro")
        store.set_model("skill_matcher", "gpt-5.5")
        return AgentWorkbuddyCallback(
            plan_store=store,
            fallback_model="deepseek-v3",
            pool=pii_mock_pool,
        )

    @pytest.mark.asyncio
    async def test_full_pii_pipeline_mask_route_restore(self, pii_router):
        """完整管道: PII 脱敏 → agent_workbuddy 路由 → LLM 响应含占位符 → 还原。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫张三，手机号是13800138000，请帮我审核简历",
                    "agent": "resume_parser",
                }
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-007",
                "request_id": "req-e2e-wb-007",
            },
        }

        # Phase 1: pre_call_hook (compliance + PII masking + routing)
        await pii_router.async_pre_call_hook({}, None, data, "completion")

        # Verify: PII was masked
        masked_content = data["messages"][0]["content"]
        assert "[PERSON_1]" in masked_content
        assert "[PHONE_1]" in masked_content
        assert "张三" not in masked_content
        assert "13800138000" not in masked_content

        # Verify: routing to correct model via agent_workbuddy
        assert data["model"] == "gemini-2.5-pro"
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["transaction_agent"] == "resume_parser"

        # Phase 2: Mock LLM response with placeholders
        llm_response = make_response(
            "[PERSON_1] 您好！已收到您的简历审核请求。联系方式: [PHONE_1]。"
        )

        # Phase 3: Response restoration
        kwargs = {"metadata": data["metadata"], "model": "gemini-2.5-pro"}
        await pii_router.async_log_success_event(
            kwargs, llm_response, None, None
        )

        # Verify: placeholders restored to original PII
        client_response = llm_response.choices[0].message.content
        assert "张三" in client_response
        assert "13800138000" in client_response
        assert "[PERSON_1]" not in client_response
        assert "[PHONE_1]" not in client_response
        assert client_response == (
            "张三 您好！已收到您的简历审核请求。联系方式: 13800138000。"
        )

        # Verify aegis_metadata
        assert hasattr(llm_response, "aegis_metadata")
        assert llm_response.aegis_metadata["routing_plugin"] == "agent_workbuddy"
        assert llm_response.aegis_metadata["assigned_model"] == "gemini-2.5-pro"
        assert llm_response.aegis_metadata["agent"] == "resume_parser"

    @pytest.mark.asyncio
    async def test_pii_masking_with_streaming_response(self, pii_mock_pool):
        """PII 脱敏 + agent_workbuddy 路由 + 流式响应还原验证。"""
        store = AgentPlanStore()
        store.set_model("skill_matcher", "gpt-5.5")

        router = AgentWorkbuddyCallback(
            plan_store=store,
            fallback_model="deepseek-v3",
            pool=pii_mock_pool,
        )

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "张三的电话号码是13800138000，请确认",
                    "agent": "skill_matcher",
                }
            ],
            "metadata": {
                "session_id": "sess-e2e-wb-007-stream",
                "request_id": "req-e2e-wb-007-stream",
            },
        }

        # Phase 1: Route
        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "gpt-5.5"
        assert "[PERSON_1]" in data["messages"][0]["content"]
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"

        # Phase 2: Streaming response with placeholders
        streaming_chunks = [
            {"choices": [{"delta": {"content": "确认 "}}]},
            {"choices": [{"delta": {"content": "[PERSON_1]"}}]},
            {"choices": [{"delta": {"content": " 的电话为 "}}]},
            {"choices": [{"delta": {"content": "[PHONE_1]"}}]},
            {"choices": [{"delta": {"content": "，已验证。"}}]},
        ]

        async def async_chunk_gen():
            for chunk in streaming_chunks:
                yield chunk

        # Phase 3: Stream restoration
        restored_parts = []
        async for chunk in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=async_chunk_gen(),
            request_data=data,
        ):
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                restored_parts.append(content)

        full_text = "".join(restored_parts)
        assert "张三" in full_text
        assert "13800138000" in full_text
        assert "[PERSON_1]" not in full_text
        assert "[PHONE_1]" not in full_text
