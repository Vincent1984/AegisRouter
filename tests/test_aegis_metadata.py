"""Tests for aegis_metadata response injection (FR-7.3).

Tests cover:
- Transaction router responses include correct aegis_metadata
- Conversation router responses include aegis_metadata with routing_plugin="conversation"
- UNKNOWN_AGENT scenarios include warning in aegis_metadata.warnings
- Dict-based response objects get aegis_metadata injected
- Object-based response objects get aegis_metadata injected
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aegis_router.callbacks.base_router import BaseRouterCallback
from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_store():
    """Create a RoutingPlanStore with test data."""
    store = RoutingPlanStore()
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("resume_screening", "skill_matcher", "gpt-5.5")
    store.set_model("code_review", "code_analyzer", "codex-mini")
    return store


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool that simulates successful operations."""
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
# Helper: Create response objects
# ---------------------------------------------------------------------------


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, message: MockMessage):
        self.message = message


class MockResponse:
    """Mock LLM response object with attribute access (like Pydantic model)."""

    def __init__(self, content: str = "Hello response"):
        self.choices = [MockChoice(MockMessage(content))]


def make_dict_response(content: str = "Hello response") -> dict:
    """Create a dict-based LLM response."""
    return {
        "choices": [{"message": {"content": content}}],
    }


# ---------------------------------------------------------------------------
# Test: _inject_aegis_metadata on dict response
# ---------------------------------------------------------------------------


class TestInjectAegisMetadataDict:
    """Test aegis_metadata injection into dict response objects."""

    def test_transaction_metadata_injected_into_dict_response(self):
        """Transaction routing metadata is correctly injected into dict response."""
        response = make_dict_response()
        metadata = {
            "transaction_template": "resume_screening",
            "transaction_agent": "resume_parser",
            "target_model": "gemini-2.5-pro",
            "routing_plugin": "transaction",
            "_routing_warnings": [],
        }

        BaseRouterCallback._inject_aegis_metadata(response, metadata)

        assert "aegis_metadata" in response
        am = response["aegis_metadata"]
        assert am["template"] == "resume_screening"
        assert am["agent"] == "resume_parser"
        assert am["assigned_model"] == "gemini-2.5-pro"
        assert am["routing_plugin"] == "transaction"
        assert am["warnings"] == []

    def test_conversation_metadata_injected_into_dict_response(self):
        """Conversation routing metadata is correctly injected into dict response."""
        response = make_dict_response()
        metadata = {
            "target_model": "gpt-5.5",
            "routing_plugin": "conversation",
            "_routing_warnings": [],
        }

        BaseRouterCallback._inject_aegis_metadata(response, metadata)

        assert "aegis_metadata" in response
        am = response["aegis_metadata"]
        assert am["template"] == ""
        assert am["agent"] == ""
        assert am["assigned_model"] == "gpt-5.5"
        assert am["routing_plugin"] == "conversation"
        assert am["warnings"] == []

    def test_unknown_agent_warning_in_metadata(self):
        """UNKNOWN_AGENT warning appears in aegis_metadata.warnings."""
        response = make_dict_response()
        metadata = {
            "transaction_template": "resume_screening",
            "transaction_agent": "unknown_agent",
            "target_model": "deepseek-v3",
            "routing_plugin": "transaction",
            "_routing_warnings": ["UNKNOWN_AGENT"],
        }

        BaseRouterCallback._inject_aegis_metadata(response, metadata)

        am = response["aegis_metadata"]
        assert am["warnings"] == ["UNKNOWN_AGENT"]
        assert am["assigned_model"] == "deepseek-v3"

    def test_empty_metadata_produces_empty_fields(self):
        """Empty metadata produces empty string/list fields."""
        response = make_dict_response()
        metadata = {}

        BaseRouterCallback._inject_aegis_metadata(response, metadata)

        am = response["aegis_metadata"]
        assert am["template"] == ""
        assert am["agent"] == ""
        assert am["assigned_model"] == ""
        assert am["routing_plugin"] == ""
        assert am["warnings"] == []


# ---------------------------------------------------------------------------
# Test: _inject_aegis_metadata on object response
# ---------------------------------------------------------------------------


class TestInjectAegisMetadataObject:
    """Test aegis_metadata injection into object-based response objects."""

    def test_transaction_metadata_injected_into_object_response(self):
        """Transaction routing metadata is correctly injected into object response."""
        response = MockResponse()
        metadata = {
            "transaction_template": "code_review",
            "transaction_agent": "code_analyzer",
            "target_model": "codex-mini",
            "routing_plugin": "transaction",
            "_routing_warnings": [],
        }

        BaseRouterCallback._inject_aegis_metadata(response, metadata)

        assert hasattr(response, "aegis_metadata")
        am = response.aegis_metadata
        assert am["template"] == "code_review"
        assert am["agent"] == "code_analyzer"
        assert am["assigned_model"] == "codex-mini"
        assert am["routing_plugin"] == "transaction"
        assert am["warnings"] == []

    def test_conversation_metadata_injected_into_object_response(self):
        """Conversation routing metadata is correctly injected into object response."""
        response = MockResponse()
        metadata = {
            "target_model": "deepseek-v4-pro",
            "routing_plugin": "conversation",
            "_routing_warnings": [],
        }

        BaseRouterCallback._inject_aegis_metadata(response, metadata)

        am = response.aegis_metadata
        assert am["template"] == ""
        assert am["agent"] == ""
        assert am["assigned_model"] == "deepseek-v4-pro"
        assert am["routing_plugin"] == "conversation"
        assert am["warnings"] == []


# ---------------------------------------------------------------------------
# Test: Full pipeline — async_log_success_event injects aegis_metadata
# ---------------------------------------------------------------------------


class TestAsyncLogSuccessEventMetadata:
    """Integration tests: async_log_success_event injects aegis_metadata."""

    @pytest.mark.asyncio
    async def test_transaction_routing_injects_metadata_on_success(
        self, router, mock_pool
    ):
        """After successful LLM call, response contains aegis_metadata for transaction routing."""
        # First, route the request to populate metadata
        data = {
            "messages": [{"role": "user", "content": "Parse this resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "gemini-2.5-pro"

        # Now simulate a successful response
        response = make_dict_response("Parsed resume content")

        kwargs = {
            "model": "gemini-2.5-pro",
            "metadata": data["metadata"],
        }

        await router.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=None,
            end_time=None,
        )

        # Verify aegis_metadata is injected
        assert "aegis_metadata" in response
        am = response["aegis_metadata"]
        assert am["template"] == "resume_screening"
        assert am["agent"] == "resume_parser"
        assert am["assigned_model"] == "gemini-2.5-pro"
        assert am["routing_plugin"] == "transaction"
        assert am["warnings"] == []

    @pytest.mark.asyncio
    async def test_unknown_agent_metadata_contains_warning(
        self, router, mock_pool
    ):
        """UNKNOWN_AGENT scenario: aegis_metadata.warnings contains the warning."""
        data = {
            "messages": [{"role": "user", "content": "Test request."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "nonexistent_agent",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "deepseek-v3"

        # Simulate successful response
        response = make_dict_response("Fallback response")

        kwargs = {
            "model": "deepseek-v3",
            "metadata": data["metadata"],
        }

        await router.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=None,
            end_time=None,
        )

        assert "aegis_metadata" in response
        am = response["aegis_metadata"]
        assert am["template"] == "resume_screening"
        assert am["agent"] == "nonexistent_agent"
        assert am["assigned_model"] == "deepseek-v3"
        assert am["routing_plugin"] == "transaction"
        assert am["warnings"] == ["UNKNOWN_AGENT"]

    @pytest.mark.asyncio
    async def test_fallback_routing_metadata(self, router, mock_pool):
        """No transaction metadata → fallback, aegis_metadata still present."""
        data = {
            "messages": [{"role": "user", "content": "Hello world."}],
            "metadata": {},
        }

        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "deepseek-v3"

        # Simulate successful response
        response = make_dict_response("Hello!")

        kwargs = {
            "model": "deepseek-v3",
            "metadata": data["metadata"],
        }

        await router.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=None,
            end_time=None,
        )

        assert "aegis_metadata" in response
        am = response["aegis_metadata"]
        assert am["template"] == ""
        assert am["agent"] == ""
        assert am["assigned_model"] == "deepseek-v3"
        assert am["routing_plugin"] == "transaction"
        assert am["warnings"] == []

    @pytest.mark.asyncio
    async def test_object_response_gets_metadata(self, router, mock_pool):
        """Object-based response also gets aegis_metadata injected."""
        data = {
            "messages": [{"role": "user", "content": "Parse this resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "gpt-5.5"

        # Simulate a successful object-based response
        response = MockResponse("Skill matching result")

        kwargs = {
            "model": "gpt-5.5",
            "metadata": data["metadata"],
        }

        await router.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=None,
            end_time=None,
        )

        assert hasattr(response, "aegis_metadata")
        am = response.aegis_metadata
        assert am["template"] == "resume_screening"
        assert am["agent"] == "skill_matcher"
        assert am["assigned_model"] == "gpt-5.5"
        assert am["routing_plugin"] == "transaction"
        assert am["warnings"] == []

    @pytest.mark.asyncio
    async def test_no_request_id_skips_metadata_injection(self, router, mock_pool):
        """When no request_id in metadata, entire success event is skipped."""
        response = make_dict_response("Test")

        kwargs = {
            "model": "gpt-5.5",
            "metadata": {},  # No request_id
        }

        await router.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=None,
            end_time=None,
        )

        # Without request_id, the method returns early — no aegis_metadata
        assert "aegis_metadata" not in response


# ---------------------------------------------------------------------------
# Test: Conversation router sets routing_plugin correctly
# ---------------------------------------------------------------------------


class TestSmartRouterMetadata:
    """Test that smart_router sets routing_plugin='conversation' in metadata."""

    @pytest.mark.asyncio
    async def test_smart_router_sets_routing_plugin_conversation(self):
        """SmartRouterCallback sets routing_plugin='conversation' during routing."""
        from aegis_router.callbacks.smart_router import SmartRouterCallback

        mock_pool = MagicMock(spec=ClawVaultPool)
        mock_pool.max_connections = 10

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

        mock_pool.call = AsyncMock(side_effect=mock_call)

        router = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=None,
            classifier=None,
        )

        data = {
            "messages": [{"role": "user", "content": "Hello world."}],
            "metadata": {},
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        # Smart router should set routing_plugin = "conversation"
        assert data["metadata"]["routing_plugin"] == "conversation"
        assert data["metadata"]["_routing_warnings"] == []


# ---------------------------------------------------------------------------
# Test: V5-4 — aegis_metadata.assigned_model 字段正确
# ---------------------------------------------------------------------------


class TestV54AegisMetadataAssignedModel:
    """V5-4: 响应中 aegis_metadata.assigned_model 字段正确。"""

    @pytest.mark.asyncio
    async def test_v5_4_assigned_model_matches_routed_model(self, router, mock_pool):
        """V5-4: assigned_model in aegis_metadata matches the model routed to."""
        data = {
            "messages": [{"role": "user", "content": "Analyze code."}],
            "metadata": {
                "transaction": {
                    "template": "code_review",
                    "agent": "code_analyzer",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")
        assert data["model"] == "codex-mini"

        response = make_dict_response("Code analysis result")
        kwargs = {"model": "codex-mini", "metadata": data["metadata"]}

        await router.async_log_success_event(
            kwargs=kwargs,
            response_obj=response,
            start_time=None,
            end_time=None,
        )

        assert response["aegis_metadata"]["assigned_model"] == "codex-mini"
