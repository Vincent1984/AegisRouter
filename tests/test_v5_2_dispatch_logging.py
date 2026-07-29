"""Integration tests for V5-2: dispatch audit logging through TransactionRouterCallback.

Verifies that the full routing flow (via async_pre_call_hook) emits valid JSON
audit logs to the `aegis_router.audit` logger with correct fields for ALL
dispatch scenarios:

- Normal plan dispatch: template + agent hit → reason="plan"
- Fallback dispatch: no transaction metadata → reason="fallback"
- Unknown agent dispatch: agent not in template → reason="unknown" + UNKNOWN_AGENT warning
"""

from __future__ import annotations

import json
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_store():
    """Create a RoutingPlanStore with test data."""
    store = RoutingPlanStore()
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("code_review", "code_analyzer", "codex-mini")
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
    """Create a TransactionRouterCallback with test store and mock pool."""
    return TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        pool=mock_pool,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_dispatch_records(caplog) -> list[dict]:
    """Extract and parse transaction_dispatch audit records from caplog."""
    records = []
    for r in caplog.records:
        if r.name == "aegis_router.audit":
            try:
                parsed = json.loads(r.getMessage())
                if parsed.get("event") == "transaction_dispatch":
                    records.append(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
    return records


# ---------------------------------------------------------------------------
# Test: Normal plan dispatch emits audit log
# ---------------------------------------------------------------------------


class TestNormalPlanDispatchLogging:
    """Verify dispatch audit log during normal plan routing."""

    @pytest.mark.asyncio
    async def test_plan_dispatch_emits_audit_log(self, router, caplog):
        """Normal routing emits a transaction_dispatch audit log."""
        data = {
            "messages": [{"role": "user", "content": "Parse this resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        assert len(dispatch_records) == 1

    @pytest.mark.asyncio
    async def test_plan_dispatch_has_correct_fields(self, router, caplog):
        """Plan dispatch log contains template, agent, assigned_model, reason='plan'."""
        data = {
            "messages": [{"role": "user", "content": "Parse this resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        entry = dispatch_records[0]

        assert entry["template"] == "resume_screening"
        assert entry["agent"] == "resume_parser"
        assert entry["assigned_model"] == "gemini-2.5-pro"
        assert entry["reason"] == "plan"
        assert entry["warnings"] == []
        assert "ts" in entry
        assert "request_id" in entry

    @pytest.mark.asyncio
    async def test_plan_dispatch_emits_valid_json(self, router, caplog):
        """Plan dispatch audit output is valid single-line JSON."""
        data = {
            "messages": [{"role": "user", "content": "Classify intent."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "intent_classifier",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) >= 1

        # Every audit record must be valid JSON
        for record in audit_records:
            parsed = json.loads(record.getMessage())
            assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_plan_dispatch_different_template(self, router, caplog):
        """Plan dispatch works for different template/agent combinations."""
        data = {
            "messages": [{"role": "user", "content": "Review code."}],
            "metadata": {
                "transaction": {
                    "template": "code_review",
                    "agent": "code_analyzer",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        assert len(dispatch_records) == 1

        entry = dispatch_records[0]
        assert entry["template"] == "code_review"
        assert entry["agent"] == "code_analyzer"
        assert entry["assigned_model"] == "codex-mini"
        assert entry["reason"] == "plan"


# ---------------------------------------------------------------------------
# Test: Fallback dispatch emits audit log
# ---------------------------------------------------------------------------


class TestFallbackDispatchLogging:
    """Verify dispatch audit log when no transaction metadata is present."""

    @pytest.mark.asyncio
    async def test_fallback_dispatch_emits_audit_log(self, router, caplog):
        """No transaction metadata emits a fallback dispatch audit log."""
        data = {
            "messages": [{"role": "user", "content": "Hello world."}],
            "metadata": {},
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        assert len(dispatch_records) == 1

    @pytest.mark.asyncio
    async def test_fallback_dispatch_has_correct_fields(self, router, caplog):
        """Fallback dispatch log has empty template/agent, reason='fallback'."""
        data = {
            "messages": [{"role": "user", "content": "Hello world."}],
            "metadata": {},
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        entry = dispatch_records[0]

        assert entry["template"] == ""
        assert entry["agent"] == ""
        assert entry["assigned_model"] == "deepseek-v3"
        assert entry["reason"] == "fallback"
        assert entry["warnings"] == []
        assert "ts" in entry

    @pytest.mark.asyncio
    async def test_fallback_dispatch_emits_valid_json(self, router, caplog):
        """Fallback dispatch audit output is valid JSON."""
        data = {
            "messages": [{"role": "user", "content": "Generic request."}],
            "metadata": {},
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) >= 1

        for record in audit_records:
            parsed = json.loads(record.getMessage())
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Test: Unknown agent dispatch emits audit log with UNKNOWN_AGENT warning
# ---------------------------------------------------------------------------


class TestUnknownAgentDispatchLogging:
    """Verify dispatch audit log when agent is not in the template."""

    @pytest.mark.asyncio
    async def test_unknown_agent_emits_audit_log(self, router, caplog):
        """Unknown agent routing emits a dispatch audit log."""
        data = {
            "messages": [{"role": "user", "content": "Test."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "nonexistent_agent",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        assert len(dispatch_records) == 1

    @pytest.mark.asyncio
    async def test_unknown_agent_has_correct_fields(self, router, caplog):
        """Unknown agent dispatch has reason='unknown' and UNKNOWN_AGENT warning."""
        data = {
            "messages": [{"role": "user", "content": "Test."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "ghost_agent",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        entry = dispatch_records[0]

        assert entry["template"] == "resume_screening"
        assert entry["agent"] == "ghost_agent"
        assert entry["assigned_model"] == "deepseek-v3"
        assert entry["reason"] == "unknown"
        assert "UNKNOWN_AGENT" in entry["warnings"]
        assert "ts" in entry

    @pytest.mark.asyncio
    async def test_unknown_agent_emits_valid_json(self, router, caplog):
        """Unknown agent dispatch audit output is valid JSON."""
        data = {
            "messages": [{"role": "user", "content": "Test."}],
            "metadata": {
                "transaction": {
                    "template": "code_review",
                    "agent": "missing_agent",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) >= 1

        for record in audit_records:
            parsed = json.loads(record.getMessage())
            assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_unknown_agent_warning_list_content(self, router, caplog):
        """UNKNOWN_AGENT is the only warning for unknown agent dispatch."""
        data = {
            "messages": [{"role": "user", "content": "Test."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "unknown_one",
                }
            },
        }

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await router.async_pre_call_hook({}, None, data, "completion")

        dispatch_records = _get_dispatch_records(caplog)
        entry = dispatch_records[0]

        assert entry["warnings"] == ["UNKNOWN_AGENT"]
