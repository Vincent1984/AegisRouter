"""回归测试 — RequestLoggerCallback 不影响事务路由行为。

Task 8.2: 事务路由回归
- 发送 3 个不同 Agent 请求，确认路由到正确模型
- 确认 RequestLoggerCallback 不影响 model 选择
- 确认 RequestLoggerCallback 不修改 data 字典

测试策略:
  1. 构建一个仅含 TransactionRouterCallback 的环境，记录路由结果（基线）
  2. 构建一个同时包含 TransactionRouterCallback + RequestLoggerCallback 的环境
  3. 对比两者的路由结果，确认完全一致
  4. 额外验证 RequestLoggerCallback 不修改 data 字典
"""

from __future__ import annotations

import copy

import pytest
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.observability.request_logger import (
    RequestLoggerCallback,
    RequestLoggingConfig,
)
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_store():
    """Create a RoutingPlanStore mimicking production plan for regression."""
    store = RoutingPlanStore()
    # resume_screening template
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "skill_matcher", "gpt-5.5")
    store.set_model("resume_screening", "compliance_checker", "deepseek-v4-pro")
    # code_review template
    store.set_model("code_review", "code_analyzer", "codex-mini")
    store.set_model("code_review", "issue_detector", "gpt-5.5")
    store.set_model("code_review", "fix_suggester", "codex-mini")
    # supplier_evaluation template
    store.set_model("supplier_evaluation", "data_collector", "local-7b")
    store.set_model("supplier_evaluation", "performance_scorer", "deepseek-v4-pro")
    store.set_model("supplier_evaluation", "compliance_checker", "gpt-5.5")
    store.set_model("supplier_evaluation", "tier_determiner", "gpt-5.5")
    return store


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool with passthrough behavior."""
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
def transaction_router(plan_store, mock_pool):
    """TransactionRouterCallback instance."""
    return TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        pool=mock_pool,
    )


@pytest.fixture
def request_logger():
    """RequestLoggerCallback configured for testing (stdout output to avoid file IO)."""
    config = RequestLoggingConfig(
        enabled=True,
        output="stdout",
        file_path="./logs/test_regression.jsonl",
        max_message_length=4096,
        retention_days=7,
    )
    return RequestLoggerCallback(config=config)


# ---------------------------------------------------------------------------
# Test data: 3 agent requests from 3 different templates
# ---------------------------------------------------------------------------

REGRESSION_TEST_CASES = [
    {
        "id": "resume_screening/skill_matcher",
        "template": "resume_screening",
        "agent": "skill_matcher",
        "expected_model": "gpt-5.5",
        "prompt": "Match candidate skills against job requirements.",
    },
    {
        "id": "code_review/code_analyzer",
        "template": "code_review",
        "agent": "code_analyzer",
        "expected_model": "codex-mini",
        "prompt": "Analyze this Python module for potential issues.",
    },
    {
        "id": "supplier_evaluation/compliance_checker",
        "template": "supplier_evaluation",
        "agent": "compliance_checker",
        "expected_model": "gpt-5.5",
        "prompt": "Check supplier compliance against regulations.",
    },
]


def _build_request_data(test_case: dict) -> dict:
    """Build a request data dict for the given test case."""
    return {
        "messages": [{"role": "user", "content": test_case["prompt"]}],
        "metadata": {
            "transaction": {
                "template": test_case["template"],
                "agent": test_case["agent"],
            },
            "session_id": f"sess-regression-{test_case['id']}",
            "request_id": f"req-regression-{test_case['id']}",
        },
    }


# ---------------------------------------------------------------------------
# TC-REGRESSION-001: Routing results are identical with RequestLoggerCallback
# ---------------------------------------------------------------------------


class TestRegressionTransactionRoutingWithRequestLogger:
    """Verify RequestLoggerCallback does NOT affect transaction routing decisions."""

    @pytest.mark.asyncio
    async def test_routing_results_identical_with_request_logger(
        self, transaction_router, request_logger
    ):
        """Send 3 requests through router + logger chain and confirm
        model selection matches expected results (identical to router-only).
        """
        for tc in REGRESSION_TEST_CASES:
            # --- Baseline: router only ---
            data_baseline = _build_request_data(tc)
            await transaction_router.async_pre_call_hook(
                {}, None, data_baseline, "completion"
            )
            baseline_model = data_baseline["model"]
            baseline_target = data_baseline["metadata"]["target_model"]
            baseline_reason = data_baseline["metadata"]["route_reason"]

            # --- With RequestLogger: router + logger ---
            data_with_logger = _build_request_data(tc)
            # Simulate the LiteLLM callback chain: router first, then logger
            await transaction_router.async_pre_call_hook(
                {}, None, data_with_logger, "completion"
            )
            # RequestLogger observes after routing
            await request_logger.async_pre_call_hook(
                {}, None, data_with_logger, "completion"
            )

            # --- Assert routing decisions are IDENTICAL ---
            assert data_with_logger["model"] == baseline_model, (
                f"[{tc['id']}] Model mismatch: "
                f"baseline={baseline_model}, with_logger={data_with_logger['model']}"
            )
            assert data_with_logger["metadata"]["target_model"] == baseline_target, (
                f"[{tc['id']}] target_model mismatch"
            )
            assert data_with_logger["metadata"]["route_reason"] == baseline_reason, (
                f"[{tc['id']}] route_reason mismatch"
            )

            # --- Assert expected model from plan ---
            assert data_with_logger["model"] == tc["expected_model"], (
                f"[{tc['id']}] Expected model={tc['expected_model']}, "
                f"got={data_with_logger['model']}"
            )

    @pytest.mark.asyncio
    async def test_request_logger_does_not_modify_data_dict(
        self, transaction_router, request_logger
    ):
        """Verify RequestLoggerCallback does NOT modify the data dict.

        After routing, take a deep copy, then run logger, then compare.
        """
        for tc in REGRESSION_TEST_CASES:
            data = _build_request_data(tc)

            # Route first
            await transaction_router.async_pre_call_hook(
                {}, None, data, "completion"
            )

            # Deep copy AFTER routing (this is the state logger receives)
            data_before_logger = copy.deepcopy(data)

            # Run RequestLogger
            result = await request_logger.async_pre_call_hook(
                {}, None, data, "completion"
            )

            # Verify data is unchanged
            assert data == data_before_logger, (
                f"[{tc['id']}] RequestLoggerCallback modified the data dict! "
                f"Diff detected."
            )

            # Verify logger returns data (not None)
            assert result is data, (
                f"[{tc['id']}] RequestLoggerCallback should return the "
                f"original data object."
            )

    @pytest.mark.asyncio
    async def test_three_different_templates_route_correctly(
        self, transaction_router, request_logger
    ):
        """Confirm all 3 test cases route to different models as expected,
        covering resume_screening, code_review, and supplier_evaluation.
        """
        routed_models = []

        for tc in REGRESSION_TEST_CASES:
            data = _build_request_data(tc)

            # Full callback chain: router → logger
            await transaction_router.async_pre_call_hook(
                {}, None, data, "completion"
            )
            await request_logger.async_pre_call_hook(
                {}, None, data, "completion"
            )

            # Verify routing metadata
            assert data["metadata"]["routing_plugin"] == "transaction"
            assert data["metadata"]["route_reason"] == "plan"
            assert data["metadata"]["transaction_template"] == tc["template"]
            assert data["metadata"]["transaction_agent"] == tc["agent"]
            assert data["model"] == tc["expected_model"]

            routed_models.append(data["model"])

        # Verify we tested 3 different templates
        templates = [tc["template"] for tc in REGRESSION_TEST_CASES]
        assert len(set(templates)) == 3, "Should test 3 different templates"

    @pytest.mark.asyncio
    async def test_disabled_request_logger_has_zero_effect(
        self, transaction_router
    ):
        """When RequestLoggerCallback is disabled, it should do nothing."""
        disabled_config = RequestLoggingConfig(enabled=False)
        disabled_logger = RequestLoggerCallback(config=disabled_config)

        for tc in REGRESSION_TEST_CASES:
            data = _build_request_data(tc)

            # Route
            await transaction_router.async_pre_call_hook(
                {}, None, data, "completion"
            )
            data_after_routing = copy.deepcopy(data)

            # Disabled logger
            result = await disabled_logger.async_pre_call_hook(
                {}, None, data, "completion"
            )

            # Zero modification
            assert data == data_after_routing
            assert result is data
            assert data["model"] == tc["expected_model"]
