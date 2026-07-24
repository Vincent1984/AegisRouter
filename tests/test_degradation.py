"""Tests for degradation strategy (降级策略).

Covers:
- TC-DEGRADE-001: ClawVault process down → bypass masking, log CRITICAL
- TC-DEGRADE-002: Redis unavailable + PII detected → return 503 error
- TC-DEGRADE-002b: Redis unavailable + no PII → request passes normally
- TC-DEGRADE-003: RouteLLM timeout → default route to fallback_model
- TC-DEGRADE-004: ClawVault recovery → auto re-enable masking
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aegis_router.callbacks.degradation import (
    ComponentState,
    DegradationError,
    DegradationManager,
)
from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client with health_check method."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


@pytest.fixture
def mock_redis_client_unhealthy():
    """Create a mock Redis client that reports unhealthy."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value={"status": "unhealthy", "error": "Connection refused"})
    return client


@pytest.fixture
def degradation_manager(mock_redis_client):
    """Create a DegradationManager with healthy Redis."""
    return DegradationManager(redis_client=mock_redis_client, fallback_model="deepseek-v3")


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock()
    pool.max_connections = 10
    return pool


@pytest.fixture
def callback_with_degradation(mock_pool, mock_redis_client):
    """Create a SmartRouterCallback with DegradationManager and mock Redis."""
    dm = DegradationManager(redis_client=mock_redis_client, fallback_model="deepseek-v3")
    return SmartRouterCallback(pool=mock_pool, degradation_manager=dm)


@pytest.fixture
def sample_data():
    """Sample request data dict as LiteLLM would pass to pre_call_hook."""
    return {
        "messages": [
            {"role": "user", "content": "我叫张三，手机号是13800138000"},
        ],
        "model": "gpt-4o",
        "metadata": {
            "session_id": "test-session-1",
            "request_id": "test-request-1",
        },
    }


@pytest.fixture
def sample_data_no_pii():
    """Sample request data with no PII content."""
    return {
        "messages": [
            {"role": "user", "content": "今天天气怎么样？"},
        ],
        "model": "gpt-4o",
        "metadata": {
            "session_id": "test-session-2",
            "request_id": "test-request-2",
        },
    }


# ---------------------------------------------------------------------------
# TC-DEGRADE-001: ClawVault process down → bypass masking, log CRITICAL
# ---------------------------------------------------------------------------


class TestClawVaultBypass:
    """TC-DEGRADE-001: ClawVault 进程挂掉时 bypass 脱敏。"""

    async def test_bypass_when_clawvault_down_on_compliance(
        self, callback_with_degradation, mock_pool, sample_data
    ):
        """ClawVault down during compliance → bypass, message unchanged."""
        original_content = sample_data["messages"][0]["content"]
        mock_pool.call.return_value = None  # ClawVault unavailable

        await callback_with_degradation.async_pre_call_hook(
            {}, None, sample_data, "completion"
        )

        # Message content unchanged (bypass)
        assert sample_data["messages"][0]["content"] == original_content

    async def test_bypass_when_clawvault_down_on_mask(
        self, callback_with_degradation, mock_pool, sample_data
    ):
        """ClawVault down during masking → bypass, message unchanged."""
        original_content = sample_data["messages"][0]["content"]
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},  # compliance passes
            None,  # mask returns None
        ]

        await callback_with_degradation.async_pre_call_hook(
            {}, None, sample_data, "completion"
        )

        # Message content unchanged (bypass)
        assert sample_data["messages"][0]["content"] == original_content

    async def test_clawvault_down_reports_unhealthy_state(
        self, callback_with_degradation, mock_pool, sample_data
    ):
        """ClawVault down → DegradationManager state set to UNHEALTHY."""
        mock_pool.call.return_value = None

        await callback_with_degradation.async_pre_call_hook(
            {}, None, sample_data, "completion"
        )

        assert (
            callback_with_degradation._degradation.clawvault_state
            == ComponentState.UNHEALTHY
        )

    async def test_clawvault_down_logs_critical(
        self, callback_with_degradation, mock_pool, sample_data, caplog
    ):
        """ClawVault down → CRITICAL log emitted."""
        import logging

        mock_pool.call.return_value = None

        with caplog.at_level(logging.CRITICAL):
            await callback_with_degradation.async_pre_call_hook(
                {}, None, sample_data, "completion"
            )

        assert any("不可用" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# TC-DEGRADE-002: Redis unavailable + PII detected → return 503 error
# ---------------------------------------------------------------------------


class TestRedisUnavailablePiiDetected:
    """TC-DEGRADE-002: Redis 不可用 + 检测到 PII → 拒绝请求 (503)。"""

    async def test_rejects_with_degradation_error_when_redis_down_and_pii(
        self, mock_pool, mock_redis_client_unhealthy, sample_data
    ):
        """Redis down + PII detected → raises DegradationError."""
        dm = DegradationManager(
            redis_client=mock_redis_client_unhealthy, fallback_model="deepseek-v3"
        )
        callback = SmartRouterCallback(pool=mock_pool, degradation_manager=dm)

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},  # compliance passes
            {
                "masked_text": "我叫[PERSON_1]，手机号是[PHONE_1]",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.9},
                    {"type": "PHONE_NUMBER", "start": 9, "end": 20, "score": 0.95},
                ],
            },
        ]

        with pytest.raises(DegradationError) as exc_info:
            await callback.async_pre_call_hook({}, None, sample_data, "completion")

        assert exc_info.value.component == "redis"
        assert "503" in exc_info.value.message or "unavailable" in exc_info.value.message.lower()

    async def test_redis_unhealthy_state_recorded(
        self, mock_pool, mock_redis_client_unhealthy, sample_data
    ):
        """Redis down → DegradationManager state set to UNHEALTHY."""
        dm = DegradationManager(
            redis_client=mock_redis_client_unhealthy, fallback_model="deepseek-v3"
        )
        callback = SmartRouterCallback(pool=mock_pool, degradation_manager=dm)

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "我叫[PERSON_1]",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.9},
                ],
            },
        ]

        with pytest.raises(DegradationError):
            await callback.async_pre_call_hook({}, None, sample_data, "completion")

        assert dm.redis_state == ComponentState.UNHEALTHY


# ---------------------------------------------------------------------------
# TC-DEGRADE-002b: Redis unavailable + no PII → request passes normally
# ---------------------------------------------------------------------------


class TestRedisUnavailableNoPii:
    """TC-DEGRADE-002b: Redis 不可用 + 无 PII → 请求正常放行。"""

    async def test_allows_through_when_redis_down_but_no_pii(
        self, mock_pool, mock_redis_client_unhealthy, sample_data_no_pii
    ):
        """Redis down + no PII → request passes through normally."""
        dm = DegradationManager(
            redis_client=mock_redis_client_unhealthy, fallback_model="deepseek-v3"
        )
        callback = SmartRouterCallback(pool=mock_pool, degradation_manager=dm)

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "今天天气怎么样？",
                "entities_found": [],  # No PII
            },
        ]

        # Should NOT raise — no PII means no Redis dependency
        await callback.async_pre_call_hook({}, None, sample_data_no_pii, "completion")

        # Request went through normally
        assert sample_data_no_pii["messages"][0]["content"] == "今天天气怎么样？"


# ---------------------------------------------------------------------------
# TC-DEGRADE-003: RouteLLM timeout → default route to fallback_model
# ---------------------------------------------------------------------------


class TestRouteLLMTimeout:
    """TC-DEGRADE-003: RouteLLM 超时 → 默认路由到 fallback_model。"""

    async def test_timeout_routes_to_fallback(self, mock_pool):
        """Classifier timeout → routes to fallback_model with reason=classifier_timeout."""
        from aegis_router.router.model_classifier import ModelClassifier
        from aegis_router.router.rule_engine import RuleEngine
        from aegis_router.config import TrivialConfig

        # Create a classifier that raises TimeoutError
        mock_classifier = MagicMock(spec=ModelClassifier)
        mock_classifier.aclassify = AsyncMock(side_effect=TimeoutError("inference timeout"))

        # Create a rule engine that doesn't match
        mock_rule_engine = MagicMock(spec=RuleEngine)
        mock_rule_result = MagicMock()
        mock_rule_result.matched = False
        mock_rule_engine.check.return_value = mock_rule_result

        dm = DegradationManager(fallback_model="deepseek-v3")
        callback = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=mock_rule_engine,
            classifier=mock_classifier,
            degradation_manager=dm,
        )

        # Masking succeeds with no PII
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "写一段复杂的代码", "entities_found": []},
        ]

        data = {
            "messages": [{"role": "user", "content": "写一段复杂的代码"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }

        await callback.async_pre_call_hook({}, None, data, "completion")

        # Model should be changed to fallback
        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "classifier_timeout"


# ---------------------------------------------------------------------------
# TC-DEGRADE-004: ClawVault recovery → auto re-enable masking
# ---------------------------------------------------------------------------


class TestClawVaultRecovery:
    """TC-DEGRADE-004: ClawVault 恢复时自动重新启用脱敏。"""

    async def test_auto_recovery_after_clawvault_returns(
        self, mock_pool, mock_redis_client
    ):
        """ClawVault recovers → state goes back to HEALTHY, masking works."""
        dm = DegradationManager(redis_client=mock_redis_client, fallback_model="deepseek-v3")
        callback = SmartRouterCallback(pool=mock_pool, degradation_manager=dm)

        # --- First request: ClawVault down ---
        mock_pool.call.return_value = None
        data1 = {
            "messages": [{"role": "user", "content": "Hello John"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }

        await callback.async_pre_call_hook({}, None, data1, "completion")
        assert dm.clawvault_state == ComponentState.UNHEALTHY
        # Message unchanged (bypass)
        assert data1["messages"][0]["content"] == "Hello John"

        # --- Second request: ClawVault recovered ---
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "Hello [PERSON_1]",
                "entities_found": [{"type": "PERSON", "start": 6, "end": 10, "score": 0.9}],
            },
        ]

        data2 = {
            "messages": [{"role": "user", "content": "Hello John"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s2", "request_id": "r2"},
        }

        await callback.async_pre_call_hook({}, None, data2, "completion")

        # State should be healthy again
        assert dm.clawvault_state == ComponentState.HEALTHY
        # Message should be masked now
        assert data2["messages"][0]["content"] == "Hello [PERSON_1]"

    async def test_recovery_logs_critical(
        self, mock_pool, mock_redis_client, caplog
    ):
        """ClawVault recovery → CRITICAL log about re-enabling."""
        import logging

        dm = DegradationManager(redis_client=mock_redis_client, fallback_model="deepseek-v3")
        callback = SmartRouterCallback(pool=mock_pool, degradation_manager=dm)

        # Force unhealthy state first
        dm.report_clawvault_unhealthy()
        assert dm.clawvault_state == ComponentState.UNHEALTHY

        # Now ClawVault responds
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "test", "entities_found": []},
        ]

        data = {
            "messages": [{"role": "user", "content": "test"}],
            "model": "gpt-4o",
            "metadata": {"session_id": "s1", "request_id": "r1"},
        }

        with caplog.at_level(logging.CRITICAL):
            await callback.async_pre_call_hook({}, None, data, "completion")

        assert dm.clawvault_state == ComponentState.HEALTHY
        assert any("恢复正常" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Unit Tests: DegradationManager standalone
# ---------------------------------------------------------------------------


class TestDegradationManagerUnit:
    """Unit tests for DegradationManager internal logic."""

    def test_initial_state_is_unknown(self, degradation_manager):
        """All component states start as UNKNOWN."""
        assert degradation_manager.clawvault_state == ComponentState.UNKNOWN
        assert degradation_manager.redis_state == ComponentState.UNKNOWN
        assert degradation_manager.classifier_state == ComponentState.UNKNOWN

    def test_report_clawvault_unhealthy(self, degradation_manager):
        """report_clawvault_unhealthy sets state to UNHEALTHY."""
        degradation_manager.report_clawvault_unhealthy()
        assert degradation_manager.clawvault_state == ComponentState.UNHEALTHY

    def test_report_clawvault_healthy(self, degradation_manager):
        """report_clawvault_healthy sets state to HEALTHY."""
        degradation_manager.report_clawvault_unhealthy()
        degradation_manager.report_clawvault_healthy()
        assert degradation_manager.clawvault_state == ComponentState.HEALTHY

    def test_report_redis_unhealthy(self, degradation_manager):
        """report_redis_unhealthy sets state to UNHEALTHY."""
        degradation_manager.report_redis_unhealthy()
        assert degradation_manager.redis_state == ComponentState.UNHEALTHY

    def test_report_redis_healthy(self, degradation_manager):
        """report_redis_healthy sets state to HEALTHY."""
        degradation_manager.report_redis_unhealthy()
        degradation_manager.report_redis_healthy()
        assert degradation_manager.redis_state == ComponentState.HEALTHY

    def test_should_reject_true_when_unhealthy_and_pii(self, degradation_manager):
        """should_reject_for_redis returns True when Redis unhealthy + PII."""
        degradation_manager.report_redis_unhealthy()
        assert degradation_manager.should_reject_for_redis(pii_detected=True) is True

    def test_should_reject_false_when_unhealthy_and_no_pii(self, degradation_manager):
        """should_reject_for_redis returns False when Redis unhealthy + no PII."""
        degradation_manager.report_redis_unhealthy()
        assert degradation_manager.should_reject_for_redis(pii_detected=False) is False

    def test_should_reject_false_when_healthy(self, degradation_manager):
        """should_reject_for_redis returns False when Redis healthy."""
        degradation_manager.report_redis_healthy()
        assert degradation_manager.should_reject_for_redis(pii_detected=True) is False

    def test_enforce_redis_policy_raises_on_reject(self, degradation_manager):
        """enforce_redis_policy raises DegradationError when rejection needed."""
        degradation_manager.report_redis_unhealthy()
        with pytest.raises(DegradationError) as exc_info:
            degradation_manager.enforce_redis_policy(
                pii_detected=True, request_id="test-123"
            )
        assert exc_info.value.component == "redis"

    def test_enforce_redis_policy_no_raise_when_no_pii(self, degradation_manager):
        """enforce_redis_policy does not raise when no PII detected."""
        degradation_manager.report_redis_unhealthy()
        # Should NOT raise
        degradation_manager.enforce_redis_policy(
            pii_detected=False, request_id="test-456"
        )

    async def test_check_redis_health_healthy(self, mock_redis_client):
        """check_redis_health returns HEALTHY when Redis responds."""
        dm = DegradationManager(redis_client=mock_redis_client)
        state = await dm.check_redis_health()
        assert state == ComponentState.HEALTHY
        assert dm.redis_state == ComponentState.HEALTHY

    async def test_check_redis_health_unhealthy(self, mock_redis_client_unhealthy):
        """check_redis_health returns UNHEALTHY when Redis fails."""
        dm = DegradationManager(redis_client=mock_redis_client_unhealthy)
        state = await dm.check_redis_health()
        assert state == ComponentState.UNHEALTHY
        assert dm.redis_state == ComponentState.UNHEALTHY

    async def test_check_redis_health_exception(self):
        """check_redis_health returns UNHEALTHY when health_check raises."""
        client = MagicMock()
        client.health_check = AsyncMock(side_effect=ConnectionError("connection lost"))
        dm = DegradationManager(redis_client=client)
        state = await dm.check_redis_health()
        assert state == ComponentState.UNHEALTHY

    async def test_check_redis_health_no_client(self):
        """check_redis_health returns HEALTHY when no client configured."""
        dm = DegradationManager(redis_client=None)
        state = await dm.check_redis_health()
        assert state == ComponentState.HEALTHY

    def test_get_status_summary(self, degradation_manager):
        """get_status returns all component states."""
        degradation_manager.report_clawvault_healthy()
        degradation_manager.report_redis_unhealthy()
        degradation_manager.report_classifier_healthy()

        status = degradation_manager.get_status()
        assert status["clawvault"]["state"] == "healthy"
        assert status["redis"]["state"] == "unhealthy"
        assert status["classifier"]["state"] == "healthy"
        assert status["clawvault"]["last_change"] is not None

    def test_no_duplicate_log_on_repeated_unhealthy(self, degradation_manager, caplog):
        """Repeated unhealthy reports don't emit duplicate CRITICAL logs."""
        import logging

        with caplog.at_level(logging.CRITICAL):
            degradation_manager.report_clawvault_unhealthy()
            degradation_manager.report_clawvault_unhealthy()
            degradation_manager.report_clawvault_unhealthy()

        # Only one CRITICAL log should be emitted (state didn't change)
        critical_msgs = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert len(critical_msgs) == 1

    def test_fallback_model_property(self, degradation_manager):
        """fallback_model returns the configured fallback."""
        assert degradation_manager.fallback_model == "deepseek-v3"
