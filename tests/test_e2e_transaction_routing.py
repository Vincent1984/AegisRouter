"""E2E Integration Tests — Transaction-Level Routing with Mock LLM.

Tests cover:
- TC-E2E-TXN-001: Supervisor 注入 metadata → Agent 请求经过 AegisRouter → 路由到预计算模型 → 响应正常
- TC-E2E-TXN-002: 同一流程中多个 Agent 依次调用 → 各自路由到各自的模型
- TC-E2E-TXN-003: PII 脱敏 + 事务路由 + 响应还原完整管道验证
- TC-E2E-TXN-004: Failover 场景 — Agent 的模型返回 429 → 自动切换到 failover 链模型
- TC-E2E-TXN-005: 同一 Agent (compliance_checker) 在不同模板下路由到不同模型
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
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
    """Create a RoutingPlanStore with comprehensive test data for E2E tests."""
    store = RoutingPlanStore()
    # resume_screening template
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("resume_screening", "skill_matcher", "gpt-5.5")
    store.set_model("resume_screening", "compliance_checker", "deepseek-v4-pro")
    # code_review template
    store.set_model("code_review", "code_analyzer", "codex-mini")
    store.set_model("code_review", "issue_detector", "gpt-5.5")
    store.set_model("code_review", "compliance_checker", "claude-sonnet")
    # supplier_evaluation template
    store.set_model("supplier_evaluation", "data_collector", "local-7b")
    store.set_model("supplier_evaluation", "compliance_checker", "gpt-5.5")
    store.set_model("supplier_evaluation", "report_generator", "gemini-2.5-pro")
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
    """Create a TransactionRouterCallback for basic E2E tests."""
    return TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        pool=mock_pool,
    )


@pytest.fixture
def router_with_failover(plan_store, mock_pool):
    """Create a TransactionRouterCallback with failover chains."""
    return TransactionRouterCallback(
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
# TC-E2E-TXN-001: Supervisor 注入 metadata → 路由到预计算模型 → 响应正常
# ---------------------------------------------------------------------------


class TestE2ETXN001_SupervisorMetadataRouting:
    """TC-E2E-TXN-001: Supervisor 注入 metadata → Agent 请求经过 AegisRouter
    → 路由到预计算模型 → 响应正常。

    验证完整流程: Supervisor 注入 transaction context → pre_call_hook 路由
    → Mock LLM 响应 → log_success_event 还原 → 客户端收到正确响应。
    """

    @pytest.mark.asyncio
    async def test_supervisor_injects_metadata_routes_to_precomputed_model(
        self, router
    ):
        """Supervisor 注入 resume_screening/resume_parser → 路由到 gemini-2.5-pro。"""
        # Simulate Supervisor injecting transaction metadata
        data = {
            "messages": [{"role": "user", "content": "Parse this resume for skills."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                },
                "session_id": "sess-e2e-txn-001",
                "request_id": "req-e2e-txn-001",
            },
        }

        # Phase 1: pre_call_hook (compliance + masking + routing)
        await router.async_pre_call_hook({}, None, data, "completion")

        # Verify routing to precomputed model
        assert data["model"] == "gemini-2.5-pro"
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["transaction_template"] == "resume_screening"
        assert data["metadata"]["transaction_agent"] == "resume_parser"
        assert data["metadata"]["routing_plugin"] == "transaction"

    @pytest.mark.asyncio
    async def test_full_pipeline_route_and_response_restore(self, router):
        """完整管道: 路由 → Mock LLM 响应 → 响应还原 → 客户端收到正常内容。"""
        data = {
            "messages": [{"role": "user", "content": "Analyze this code snippet."}],
            "metadata": {
                "transaction": {
                    "template": "code_review",
                    "agent": "code_analyzer",
                },
                "session_id": "sess-e2e-txn-001b",
                "request_id": "req-e2e-txn-001b",
            },
        }

        # Phase 1: Route
        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "codex-mini"

        # Phase 2: Mock LLM response
        llm_response = make_response(
            "The code looks good. No critical issues found."
        )

        # Phase 3: Response restoration (no PII, so passthrough)
        kwargs = {"metadata": data["metadata"], "model": "codex-mini"}
        await router.async_log_success_event(kwargs, llm_response, None, None)

        # Verify client receives correct response
        assert llm_response.choices[0].message.content == (
            "The code looks good. No critical issues found."
        )


# ---------------------------------------------------------------------------
# TC-E2E-TXN-002: 同一流程中多个 Agent 依次调用 → 各自路由到各自的模型
# ---------------------------------------------------------------------------


class TestE2ETXN002_MultiAgentSequentialRouting:
    """TC-E2E-TXN-002: 同一流程中多个 Agent 依次调用 → 各自路由到各自的模型。

    模拟 Supervisor 编排 resume_screening 流程：
    intent_classifier → resume_parser → skill_matcher → compliance_checker
    每个 Agent 路由到各自预计算的不同模型。
    """

    @pytest.mark.asyncio
    async def test_sequential_agents_route_to_different_models(self, router):
        """4 个 Agent 依次调用，各自路由到各自预计算的模型。"""
        agents_and_expected_models = [
            ("intent_classifier", "local-7b"),
            ("resume_parser", "gemini-2.5-pro"),
            ("skill_matcher", "gpt-5.5"),
            ("compliance_checker", "deepseek-v4-pro"),
        ]

        routed_models = []

        for agent_name, expected_model in agents_and_expected_models:
            data = {
                "messages": [
                    {"role": "user", "content": f"Agent {agent_name} processing."}
                ],
                "metadata": {
                    "transaction": {
                        "template": "resume_screening",
                        "agent": agent_name,
                    },
                    "session_id": f"sess-e2e-txn-002-{agent_name}",
                    "request_id": f"req-e2e-txn-002-{agent_name}",
                },
            }

            await router.async_pre_call_hook({}, None, data, "completion")

            # Verify each agent routes to its own model
            assert data["model"] == expected_model, (
                f"Agent '{agent_name}' expected model '{expected_model}', "
                f"got '{data['model']}'"
            )
            assert data["metadata"]["route_reason"] == "plan"
            routed_models.append(data["model"])

        # Verify all agents got different models (except where design allows same)
        # In this case, all 4 should be distinct
        assert len(set(routed_models)) == 4, (
            f"Expected 4 distinct models, got: {routed_models}"
        )

    @pytest.mark.asyncio
    async def test_sequential_agents_with_responses(self, router):
        """多个 Agent 依次调用，每个都完成完整的请求-响应周期。"""
        agents = [
            ("intent_classifier", "Classify intent: hiring request"),
            ("resume_parser", "Parse resume: extract skills"),
            ("skill_matcher", "Match skills to requirements"),
        ]

        for agent_name, prompt in agents:
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "metadata": {
                    "transaction": {
                        "template": "resume_screening",
                        "agent": agent_name,
                    },
                    "session_id": "sess-e2e-txn-002-full",
                    "request_id": f"req-e2e-txn-002-{agent_name}-full",
                },
            }

            # Route
            await router.async_pre_call_hook({}, None, data, "completion")
            routed_model = data["model"]

            # Mock LLM response
            llm_response = make_response(
                f"Response from {routed_model} for {agent_name}."
            )

            # Restore
            kwargs = {"metadata": data["metadata"], "model": routed_model}
            await router.async_log_success_event(
                kwargs, llm_response, None, None
            )

            # Verify response content
            assert f"Response from {routed_model}" in (
                llm_response.choices[0].message.content
            )


# ---------------------------------------------------------------------------
# TC-E2E-TXN-003: PII 脱敏 + 事务路由 + 响应还原完整管道验证
# ---------------------------------------------------------------------------


class TestE2ETXN003_PIIMaskingAndRoutingPipeline:
    """TC-E2E-TXN-003: PII 脱敏 + 事务路由 + 响应还原完整管道验证。

    验证含中文 PII 的请求经过完整管道:
    PII 脱敏 → 事务路由 → LLM 响应含占位符 → 还原为原始 PII → 客户端收到完整原文。
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
    def pii_router(self, plan_store, pii_mock_pool):
        """Create a router with PII masking pool."""
        return TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=pii_mock_pool,
        )

    @pytest.mark.asyncio
    async def test_full_pii_pipeline_mask_route_restore(self, pii_router):
        """完整管道: PII 脱敏 → 事务路由 → LLM 响应含占位符 → 还原 → 原文。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫张三，手机号是13800138000，请帮我审核简历",
                }
            ],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                },
                "session_id": "sess-e2e-txn-003",
                "request_id": "req-e2e-txn-003",
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

        # Verify: routing to precomputed model
        assert data["model"] == "gemini-2.5-pro"
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"

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

    @pytest.mark.asyncio
    async def test_pii_masking_with_streaming_response(self, plan_store, pii_mock_pool):
        """PII 脱敏 + 事务路由 + 流式响应还原验证。"""
        router = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=pii_mock_pool,
        )

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "张三的电话号码是13800138000，请确认",
                }
            ],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                },
                "session_id": "sess-e2e-txn-003-stream",
                "request_id": "req-e2e-txn-003-stream",
            },
        }

        # Phase 1: Route
        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "gpt-5.5"
        assert "[PERSON_1]" in data["messages"][0]["content"]

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


# ---------------------------------------------------------------------------
# TC-E2E-TXN-004: Failover 场景 — Agent 的模型返回 429 → 自动切换
# ---------------------------------------------------------------------------


class TestE2ETXN004_FailoverScenario:
    """TC-E2E-TXN-004: Failover 场景 — Agent 的模型返回 429 → 自动切换到 failover 链模型。

    验证完整 failover 流程:
    1. pre_call_hook 路由到主模型，并注入 failover 链信息
    2. Mock LLM 调用失败 (429)
    3. async_log_failure_event 选择 failover 链中下一个模型
    4. 全局方案表不受影响
    """

    @pytest.mark.asyncio
    async def test_failover_complete_flow(self, router_with_failover, plan_store):
        """完整 failover 流程: 路由 → 失败 → 切换到 failover 模型。"""
        data = {
            "messages": [
                {"role": "user", "content": "Match skills for candidate."}
            ],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                },
                "session_id": "sess-e2e-txn-004",
                "request_id": "req-e2e-txn-004",
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

        # Phase 2: Simulate LLM failure (429 Too Many Requests)
        kwargs = {"model": "gpt-5.5", "metadata": data["metadata"]}
        await router_with_failover.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # Phase 3: Verify failover selected next model
        assert data["metadata"]["_failover_model"] == "gpt-5.2"
        assert data["metadata"]["_failover_from"] == "gpt-5.5"
        assert data["metadata"]["_failover_index"] == 1

        # Phase 4: Verify global plan store is unchanged (FR-6.3)
        assert plan_store.get_model("resume_screening", "skill_matcher") == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_failover_chain_progression(self, router_with_failover):
        """Failover 链连续失败: gpt-5.5 → gpt-5.2 → claude-sonnet。"""
        data = {
            "messages": [
                {"role": "user", "content": "Process this request."}
            ],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                },
                "session_id": "sess-e2e-txn-004-chain",
                "request_id": "req-e2e-txn-004-chain",
            },
        }

        # Route to primary model
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
        """Failover 后，下一次同 template+agent 请求仍使用原计划模型。"""
        # First request: route → fail → failover
        data1 = {
            "messages": [{"role": "user", "content": "First request."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                },
                "session_id": "sess-e2e-txn-004-next",
                "request_id": "req-e2e-txn-004-next-1",
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
            "messages": [{"role": "user", "content": "Second request."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                },
                "session_id": "sess-e2e-txn-004-next",
                "request_id": "req-e2e-txn-004-next-2",
            },
        }
        await router_with_failover.async_pre_call_hook(
            {}, None, data2, "completion"
        )

        # Verify: still routes to the original plan model, not the failover
        assert data2["model"] == "gpt-5.5"
        assert plan_store.get_model("resume_screening", "skill_matcher") == "gpt-5.5"


# ---------------------------------------------------------------------------
# TC-E2E-TXN-005: 同一 Agent (compliance_checker) 在不同模板下路由到不同模型
# ---------------------------------------------------------------------------


class TestE2ETXN005_SameAgentDifferentTemplates:
    """TC-E2E-TXN-005: 同一 Agent (compliance_checker) 在不同模板下路由到不同模型。

    验证核心设计: 路由 key 是 (template, agent) 而非单独的 agent。
    同一个 compliance_checker 在 resume_screening / code_review / supplier_evaluation
    三个模板下分别路由到不同模型。
    """

    @pytest.mark.asyncio
    async def test_compliance_checker_routes_differently_per_template(
        self, router
    ):
        """compliance_checker 在 3 个模板下路由到 3 个不同模型。"""
        templates_and_expected = [
            ("resume_screening", "deepseek-v4-pro"),
            ("code_review", "claude-sonnet"),
            ("supplier_evaluation", "gpt-5.5"),
        ]

        routed_models = []

        for template, expected_model in templates_and_expected:
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Run compliance check for {template}.",
                    }
                ],
                "metadata": {
                    "transaction": {
                        "template": template,
                        "agent": "compliance_checker",
                    },
                    "session_id": f"sess-e2e-txn-005-{template}",
                    "request_id": f"req-e2e-txn-005-{template}",
                },
            }

            await router.async_pre_call_hook({}, None, data, "completion")

            assert data["model"] == expected_model, (
                f"Template '{template}': expected '{expected_model}', "
                f"got '{data['model']}'"
            )
            assert data["metadata"]["target_model"] == expected_model
            assert data["metadata"]["route_reason"] == "plan"
            assert data["metadata"]["transaction_template"] == template
            assert data["metadata"]["transaction_agent"] == "compliance_checker"
            routed_models.append(data["model"])

        # Verify all 3 templates route to distinct models
        assert len(set(routed_models)) == 3, (
            f"Expected 3 distinct models for compliance_checker, "
            f"got: {routed_models}"
        )

    @pytest.mark.asyncio
    async def test_same_agent_different_template_full_pipeline(self, router):
        """同一 Agent 在不同模板下完整执行请求-响应周期。"""
        test_cases = [
            {
                "template": "resume_screening",
                "agent": "compliance_checker",
                "expected_model": "deepseek-v4-pro",
                "prompt": "Check resume compliance.",
                "response": "Resume passes all compliance checks.",
            },
            {
                "template": "code_review",
                "agent": "compliance_checker",
                "expected_model": "claude-sonnet",
                "prompt": "Check code compliance.",
                "response": "Code meets security standards.",
            },
        ]

        for tc in test_cases:
            data = {
                "messages": [{"role": "user", "content": tc["prompt"]}],
                "metadata": {
                    "transaction": {
                        "template": tc["template"],
                        "agent": tc["agent"],
                    },
                    "session_id": f"sess-e2e-txn-005-{tc['template']}-full",
                    "request_id": f"req-e2e-txn-005-{tc['template']}-full",
                },
            }

            # Route
            await router.async_pre_call_hook({}, None, data, "completion")
            assert data["model"] == tc["expected_model"]

            # Mock LLM response
            llm_response = make_response(tc["response"])

            # Restore
            kwargs = {"metadata": data["metadata"], "model": tc["expected_model"]}
            await router.async_log_success_event(
                kwargs, llm_response, None, None
            )

            # Verify response
            assert llm_response.choices[0].message.content == tc["response"]
