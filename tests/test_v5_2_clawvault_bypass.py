"""V5-2 验证检查点: ClawVault 进程被 kill → 网关进入 bypass 模式

集成风格测试，验证以下场景：
1. ClawVault 连接池返回 None（模拟进程挂掉/连接拒绝）时，请求直通（不脱敏）
2. CRITICAL 级别日志被输出，包含相关关键字（"不可用" / "bypass"）
3. DegradationManager 状态转为 UNHEALTHY
4. bypass 模式下路由仍然可用
5. ClawVault 恢复后重新启用脱敏功能，并记录恢复日志
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from aegis_router.callbacks.degradation import (
    ComponentState,
    DegradationManager,
)
from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """Mock ClawVaultPool that simulates connection failure (returns None)."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock(return_value=None)
    pool.max_connections = 10
    return pool


@pytest.fixture
def mock_redis_client():
    """Healthy Redis client."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


@pytest.fixture
def degradation_manager(mock_redis_client):
    """Fresh DegradationManager with healthy Redis."""
    return DegradationManager(redis_client=mock_redis_client, fallback_model="deepseek-v3")


@pytest.fixture
def callback(mock_pool, degradation_manager):
    """SmartRouterCallback wired to the mock pool and degradation manager, routing disabled."""
    return SmartRouterCallback(
        pool=mock_pool,
        degradation_manager=degradation_manager,
        enable_routing=False,
    )


@pytest.fixture
def callback_with_routing(mock_pool, degradation_manager):
    """SmartRouterCallback with routing enabled via a trivial rule engine."""
    from aegis_router.router.rule_engine import RuleEngine
    from aegis_router.config import TrivialConfig

    # Rule engine that never matches — forces fallback routing path
    rule_engine = MagicMock(spec=RuleEngine)
    rule_result = MagicMock()
    rule_result.matched = False
    rule_engine.check.return_value = rule_result

    return SmartRouterCallback(
        pool=mock_pool,
        degradation_manager=degradation_manager,
        enable_routing=True,
        rule_engine=rule_engine,
        classifier=None,  # No classifier → fallback model
    )


@pytest.fixture
def pii_request():
    """Request containing PII data."""
    return {
        "messages": [
            {"role": "user", "content": "我叫张三，身份证号是110101199001011234"},
        ],
        "model": "gpt-4o",
        "metadata": {
            "session_id": "v5-2-session",
            "request_id": "v5-2-request",
        },
    }


@pytest.fixture
def simple_request():
    """Simple request without PII."""
    return {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "model": "gpt-4o",
        "metadata": {
            "session_id": "v5-2-session-2",
            "request_id": "v5-2-request-2",
        },
    }


# ---------------------------------------------------------------------------
# Test Class: ClawVault Kill → Bypass Mode
# ---------------------------------------------------------------------------


class TestClawVaultKillBypass:
    """V5-2: ClawVault 进程被 kill 后网关进入 bypass 模式的集成测试。"""

    # ------------------------------------------------------------------
    # Scenario 1: Request passes through WITHOUT masking (bypass)
    # ------------------------------------------------------------------

    async def test_pii_content_unchanged_when_clawvault_dead(
        self, callback, mock_pool, pii_request
    ):
        """When ClawVault returns None (process killed), message content is NOT masked."""
        original_content = pii_request["messages"][0]["content"]

        await callback.async_pre_call_hook({}, None, pii_request, "completion")

        # The PII-containing message should remain unchanged — no masking occurred
        assert pii_request["messages"][0]["content"] == original_content
        assert "张三" in pii_request["messages"][0]["content"]
        assert "110101199001011234" in pii_request["messages"][0]["content"]

    async def test_bypass_on_compliance_check_failure(
        self, callback, mock_pool, pii_request
    ):
        """Compliance check returning None triggers bypass — content unchanged."""
        original_content = pii_request["messages"][0]["content"]
        # Pool returns None on first call (check_compliance)
        mock_pool.call.return_value = None

        await callback.async_pre_call_hook({}, None, pii_request, "completion")

        assert pii_request["messages"][0]["content"] == original_content

    async def test_bypass_on_mask_failure_after_compliance_passes(
        self, callback, mock_pool, pii_request
    ):
        """Compliance passes but mask returns None → bypass, content unchanged."""
        original_content = pii_request["messages"][0]["content"]
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},  # compliance OK
            None,  # mask returns None (ClawVault died between calls)
        ]

        await callback.async_pre_call_hook({}, None, pii_request, "completion")

        assert pii_request["messages"][0]["content"] == original_content

    # ------------------------------------------------------------------
    # Scenario 2: CRITICAL log is emitted
    # ------------------------------------------------------------------

    async def test_critical_log_emitted_on_compliance_none(
        self, callback, mock_pool, pii_request, caplog
    ):
        """CRITICAL log with '不可用' keyword when compliance returns None."""
        mock_pool.call.return_value = None

        with caplog.at_level(logging.CRITICAL):
            await callback.async_pre_call_hook({}, None, pii_request, "completion")

        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) >= 1
        # Check for relevant keywords
        combined_messages = " ".join(r.message for r in critical_records)
        assert "不可用" in combined_messages or "bypass" in combined_messages.lower()

    async def test_critical_log_emitted_on_mask_none(
        self, callback, mock_pool, pii_request, caplog
    ):
        """CRITICAL log with '不可用' keyword when mask returns None."""
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            None,  # mask fails
        ]

        with caplog.at_level(logging.CRITICAL):
            await callback.async_pre_call_hook({}, None, pii_request, "completion")

        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_records) >= 1
        combined_messages = " ".join(r.message for r in critical_records)
        assert "不可用" in combined_messages

    async def test_critical_log_contains_request_id(
        self, callback, mock_pool, pii_request, caplog
    ):
        """CRITICAL log includes the request_id for traceability."""
        mock_pool.call.return_value = None

        with caplog.at_level(logging.CRITICAL):
            await callback.async_pre_call_hook({}, None, pii_request, "completion")

        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        combined_messages = " ".join(r.message for r in critical_records)
        assert "v5-2-request" in combined_messages

    # ------------------------------------------------------------------
    # Scenario 3: DegradationManager transitions to UNHEALTHY
    # ------------------------------------------------------------------

    async def test_degradation_state_becomes_unhealthy(
        self, callback, mock_pool, degradation_manager, pii_request
    ):
        """DegradationManager.clawvault_state becomes UNHEALTHY after failure."""
        assert degradation_manager.clawvault_state != ComponentState.UNHEALTHY

        mock_pool.call.return_value = None
        await callback.async_pre_call_hook({}, None, pii_request, "completion")

        assert degradation_manager.clawvault_state == ComponentState.UNHEALTHY

    async def test_repeated_failures_keep_unhealthy_state(
        self, callback, mock_pool, degradation_manager, pii_request
    ):
        """Multiple failures keep state UNHEALTHY without duplicate transitions."""
        mock_pool.call.return_value = None

        # First request
        await callback.async_pre_call_hook({}, None, pii_request, "completion")
        assert degradation_manager.clawvault_state == ComponentState.UNHEALTHY

        # Second request (still dead)
        request2 = {
            "messages": [{"role": "user", "content": "Another request"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s2", "request_id": "r2"},
        }
        await callback.async_pre_call_hook({}, None, request2, "completion")
        assert degradation_manager.clawvault_state == ComponentState.UNHEALTHY

    # ------------------------------------------------------------------
    # Scenario 4: Routing still works in bypass mode
    # ------------------------------------------------------------------

    async def test_routing_works_in_bypass_mode(
        self, callback_with_routing, mock_pool, pii_request
    ):
        """Even in bypass mode, routing pipeline still executes (fallback model)."""
        mock_pool.call.return_value = None

        await callback_with_routing.async_pre_call_hook(
            {}, None, pii_request, "completion"
        )

        # Content should be unchanged (bypass)
        assert "张三" in pii_request["messages"][0]["content"]
        # Model should be routed to fallback since no classifier available
        assert pii_request["model"] == "deepseek-v3"
        # Metadata should contain routing info
        assert pii_request["metadata"].get("target_model") == "deepseek-v3"

    # ------------------------------------------------------------------
    # Scenario 5: Recovery after ClawVault comes back
    # ------------------------------------------------------------------

    async def test_recovery_restores_masking(
        self, callback, mock_pool, degradation_manager
    ):
        """After ClawVault recovers, masking is re-enabled and state is HEALTHY."""
        # --- Phase 1: ClawVault is dead ---
        mock_pool.call.return_value = None
        request1 = {
            "messages": [{"role": "user", "content": "我叫李四"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }
        await callback.async_pre_call_hook({}, None, request1, "completion")

        assert degradation_manager.clawvault_state == ComponentState.UNHEALTHY
        assert request1["messages"][0]["content"] == "我叫李四"  # Unchanged

        # --- Phase 2: ClawVault recovers ---
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "我叫[PERSON_1]",
                "entities_found": [{"type": "PERSON", "start": 2, "end": 4, "score": 0.9}],
            },
        ]
        request2 = {
            "messages": [{"role": "user", "content": "我叫李四"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s2", "request_id": "r2"},
        }
        await callback.async_pre_call_hook({}, None, request2, "completion")

        # State restored
        assert degradation_manager.clawvault_state == ComponentState.HEALTHY
        # Masking re-enabled
        assert request2["messages"][0]["content"] == "我叫[PERSON_1]"

    async def test_recovery_emits_critical_log(
        self, callback, mock_pool, degradation_manager, caplog
    ):
        """Recovery from UNHEALTHY emits a CRITICAL log about restoration."""
        # Force into unhealthy state
        degradation_manager.report_clawvault_unhealthy()
        assert degradation_manager.clawvault_state == ComponentState.UNHEALTHY

        # ClawVault responds again
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "test content", "entities_found": []},
        ]
        request = {
            "messages": [{"role": "user", "content": "test content"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }

        with caplog.at_level(logging.CRITICAL):
            await callback.async_pre_call_hook({}, None, request, "completion")

        assert degradation_manager.clawvault_state == ComponentState.HEALTHY
        critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        combined_messages = " ".join(r.message for r in critical_records)
        assert "恢复正常" in combined_messages

    # ------------------------------------------------------------------
    # Scenario 6: Full lifecycle — down → bypass → recovery → masking
    # ------------------------------------------------------------------

    async def test_full_lifecycle_down_bypass_recovery(
        self, callback, mock_pool, degradation_manager, caplog
    ):
        """End-to-end: ClawVault down → bypass → recovery → masking restored."""
        # Step 1: ClawVault dead
        mock_pool.call.return_value = None
        req1 = {
            "messages": [{"role": "user", "content": "我的电话是13912345678"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "lifecycle-s1", "request_id": "lifecycle-r1"},
        }

        with caplog.at_level(logging.CRITICAL):
            await callback.async_pre_call_hook({}, None, req1, "completion")

        # Verify bypass
        assert req1["messages"][0]["content"] == "我的电话是13912345678"
        assert degradation_manager.clawvault_state == ComponentState.UNHEALTHY

        # Verify CRITICAL log
        critical_records_phase1 = [
            r for r in caplog.records if r.levelno == logging.CRITICAL
        ]
        assert any("不可用" in r.message for r in critical_records_phase1)

        # Step 2: ClawVault recovers
        caplog.clear()
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "我的电话是[PHONE_1]",
                "entities_found": [
                    {"type": "PHONE_NUMBER", "start": 5, "end": 16, "score": 0.95}
                ],
            },
        ]
        req2 = {
            "messages": [{"role": "user", "content": "我的电话是13912345678"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "lifecycle-s2", "request_id": "lifecycle-r2"},
        }

        with caplog.at_level(logging.CRITICAL):
            await callback.async_pre_call_hook({}, None, req2, "completion")

        # Verify masking restored
        assert req2["messages"][0]["content"] == "我的电话是[PHONE_1]"
        assert degradation_manager.clawvault_state == ComponentState.HEALTHY

        # Verify recovery log
        critical_records_phase2 = [
            r for r in caplog.records if r.levelno == logging.CRITICAL
        ]
        assert any("恢复正常" in r.message for r in critical_records_phase2)
