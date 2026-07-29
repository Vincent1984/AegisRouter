"""Tests for /health/routing endpoint (Task 16 — FR-9.1, FR-9.3).

Verifies that the endpoint returns:
1. The correct routing plugin type
2. Plan summary data when transaction plugin is active
3. Minimal routing info (null plan_summary) when conversation plugin is active
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis_router.health import health_router
from aegis_router.router.routing_plan_store import RoutingPlanStore

_PATCH_PLUGIN_INFO = "aegis_router.health._get_routing_plugin_info"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a FastAPI app with the health router included."""
    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture
def plan_store_with_data() -> RoutingPlanStore:
    """Create a RoutingPlanStore with sample plan data."""
    store = RoutingPlanStore()
    # resume_screening template (4 agents)
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "skill_matcher", "gpt-5.5")
    store.set_model("resume_screening", "compliance_checker", "deepseek-v4-pro")
    # code_review template (3 agents)
    store.set_model("code_review", "code_analyzer", "codex-mini")
    store.set_model("code_review", "issue_detector", "gpt-5.5")
    store.set_model("code_review", "fix_suggester", "codex-mini")
    # supplier_evaluation template (4 agents, compliance_checker shared)
    store.set_model("supplier_evaluation", "data_collector", "local-7b")
    store.set_model("supplier_evaluation", "performance_scorer", "deepseek-v4-pro")
    store.set_model("supplier_evaluation", "compliance_checker", "deepseek-v4-pro")
    store.set_model("supplier_evaluation", "tier_determiner", "gpt-5.5")
    return store


@pytest.fixture
def mock_transaction_plugin(plan_store_with_data):
    """Create a mock TransactionRouterCallback with plan data."""
    from aegis_router.callbacks.transaction_router import TransactionRouterCallback

    mock_instance = MagicMock(spec=TransactionRouterCallback)
    mock_instance.__class__ = TransactionRouterCallback
    mock_instance.plan_store = plan_store_with_data
    return mock_instance


# ---------------------------------------------------------------------------
# Tests: Transaction plugin active
# ---------------------------------------------------------------------------


class TestTransactionPluginActive:
    """Endpoint returns plan summary when transaction plugin is loaded."""

    def test_returns_transaction_plugin_type(self, app, mock_transaction_plugin):
        """routing_plugin field is 'transaction'."""
        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("transaction", mock_transaction_plugin),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        assert response.status_code == 200
        data = response.json()
        assert data["routing_plugin"] == "transaction"

    def test_returns_plan_summary(self, app, mock_transaction_plugin):
        """plan_summary contains template/agent/model data."""
        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("transaction", mock_transaction_plugin),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        data = response.json()
        summary = data["plan_summary"]
        assert summary is not None
        assert summary["total_templates"] == 3
        assert summary["total_agent_model_mappings"] == 11
        assert summary["per_template_agent_count"]["resume_screening"] == 4
        assert summary["per_template_agent_count"]["code_review"] == 3
        assert summary["per_template_agent_count"]["supplier_evaluation"] == 4

    def test_plan_table_contains_correct_mappings(self, app, mock_transaction_plugin):
        """plan_table has full template → {agent → model} data."""
        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("transaction", mock_transaction_plugin),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        data = response.json()
        plan_table = data["plan_summary"]["plan_table"]
        assert plan_table["resume_screening"]["intent_classifier"] == "local-7b"
        assert plan_table["resume_screening"]["resume_parser"] == "gemini-2.5-pro"
        assert plan_table["code_review"]["code_analyzer"] == "codex-mini"

    def test_cross_template_agents(self, app, mock_transaction_plugin):
        """cross_template_agents shows agents appearing in multiple templates."""
        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("transaction", mock_transaction_plugin),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        data = response.json()
        cross = data["plan_summary"]["cross_template_agents"]
        # compliance_checker appears in resume_screening and supplier_evaluation
        assert "compliance_checker" in cross
        assert "resume_screening" in cross["compliance_checker"]
        assert "supplier_evaluation" in cross["compliance_checker"]
        # gpt-5.5 agents (skill_matcher, issue_detector, tier_determiner) are different agents
        # so they should NOT appear in cross_template_agents individually

    def test_empty_plan_store(self, app):
        """Transaction plugin with empty plan store returns zeroed summary."""
        from aegis_router.callbacks.transaction_router import TransactionRouterCallback

        empty_store = RoutingPlanStore()
        mock_instance = MagicMock(spec=TransactionRouterCallback)
        mock_instance.__class__ = TransactionRouterCallback
        mock_instance.plan_store = empty_store

        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("transaction", mock_instance),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        data = response.json()
        summary = data["plan_summary"]
        assert summary["total_templates"] == 0
        assert summary["total_agent_model_mappings"] == 0
        assert summary["per_template_agent_count"] == {}
        assert summary["plan_table"] == {}
        assert summary["cross_template_agents"] == {}


# ---------------------------------------------------------------------------
# Tests: Conversation plugin active
# ---------------------------------------------------------------------------


class TestConversationPluginActive:
    """Endpoint returns null plan_summary when conversation plugin is loaded."""

    def test_returns_conversation_plugin_type(self, app):
        """routing_plugin field is 'conversation'."""
        mock_instance = MagicMock()  # Not a TransactionRouterCallback
        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("conversation", mock_instance),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        assert response.status_code == 200
        data = response.json()
        assert data["routing_plugin"] == "conversation"

    def test_plan_summary_is_null(self, app):
        """plan_summary is null for conversation plugin."""
        mock_instance = MagicMock()
        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("conversation", mock_instance),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        data = response.json()
        assert data["plan_summary"] is None


# ---------------------------------------------------------------------------
# Tests: No plugin loaded yet
# ---------------------------------------------------------------------------


class TestNoPluginLoaded:
    """Endpoint handles the case when no plugin has been loaded."""

    def test_unknown_plugin_type_when_not_loaded(self, app):
        """Returns 'unknown' when no plugin has been loaded yet."""
        with patch(
            _PATCH_PLUGIN_INFO,
            return_value=("unknown", None),
        ):
            client = TestClient(app)
            response = client.get("/health/routing")

        assert response.status_code == 200
        data = response.json()
        assert data["routing_plugin"] == "unknown"
        assert data["plan_summary"] is None
