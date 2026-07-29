"""Tests for V2-8: aegis_metadata 正确填充。

Verifies that after AgentWorkbuddyCallback._execute_routing() completes,
the metadata fields are correctly set such that BaseRouterCallback._inject_aegis_metadata()
produces the expected aegis_metadata structure in response objects.

Tests cover:
- Normal routing path: known agent → correct aegis_metadata
- Fallback paths: NO_AGENT, INVALID_AGENT, UNKNOWN_AGENT
- _inject_aegis_metadata works on dict-style response objects
- _inject_aegis_metadata works on object-style response objects (with __dict__)
- _inject_aegis_metadata works on pydantic-like models with _hidden_params

Requirements: FR-3.3, FR-8.3
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback
from aegis_router.callbacks.base_router import BaseRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.agent_plan_store import AgentPlanStore


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_store():
    """Create an AgentPlanStore with test data."""
    store = AgentPlanStore()
    store.set_model("intent_classifier", "deepseek-v4-pro")
    store.set_model("document_parser", "gpt-5.5")
    store.set_model("code_assistant", "codex-mini")
    return store


@pytest.fixture
def router(plan_store):
    """Create an AgentWorkbuddyCallback with test store."""
    with patch.object(ClawVaultPool, "__init__", return_value=None):
        return AgentWorkbuddyCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
        )


class DictStyleResponse(dict):
    """A dict-style response object (isinstance(obj, dict) == True)."""
    pass


class ObjectStyleResponse:
    """An object-style response that supports setattr (has __dict__)."""
    pass


class PydanticLikeResponse:
    """Simulates a pydantic model that rejects extra attributes via __setattr__."""

    def __init__(self):
        self._hidden_params = {}

    def __setattr__(self, name, value):
        if name == "_hidden_params":
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set {name} on frozen model")


class StrictPydanticResponse:
    """Simulates a strict pydantic model with model_extra fallback."""

    def __init__(self):
        object.__setattr__(self, "model_extra", {})

    def __setattr__(self, name, value):
        if name in ("model_extra",):
            object.__setattr__(self, name, value)
        else:
            raise ValueError(f"Cannot set {name}")


# ---------------------------------------------------------------------------
# Test: Normal routing path → aegis_metadata 正确填充
# ---------------------------------------------------------------------------


class TestAegisMetadataNormalRouting:
    """V2-8: Normal routing path — known agent produces correct aegis_metadata.

    When a known agent is found in the plan store, the metadata should
    contain all fields needed for _inject_aegis_metadata to produce:
    {
        "template": "",
        "agent": "<agent_name>",
        "assigned_model": "<model_from_store>",
        "routing_plugin": "agent_workbuddy",
        "warnings": []
    }
    """

    async def test_known_agent_metadata_fields_for_aegis_injection(self, router):
        """After routing known agent, metadata has correct fields for aegis injection."""
        data = {
            "messages": [
                {"role": "user", "content": "classify intent", "agent": "intent_classifier"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        metadata = data["metadata"]
        # Verify all fields that _inject_aegis_metadata reads
        assert metadata["transaction_template"] == ""
        assert metadata["transaction_agent"] == "intent_classifier"
        assert metadata["target_model"] == "deepseek-v4-pro"
        assert metadata["routing_plugin"] == "agent_workbuddy"
        assert metadata["_routing_warnings"] == []

    async def test_known_agent_inject_aegis_metadata_dict_response(self, router):
        """Full flow: routing + _inject_aegis_metadata on dict response object."""
        data = {
            "messages": [
                {"role": "user", "content": "classify intent", "agent": "intent_classifier"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        # Simulate response injection (as done in async_log_success_event)
        response_obj = {"choices": [{"message": {"content": "result"}}]}
        BaseRouterCallback._inject_aegis_metadata(response_obj, data["metadata"])

        assert "aegis_metadata" in response_obj
        aegis = response_obj["aegis_metadata"]
        assert aegis["template"] == ""
        assert aegis["agent"] == "intent_classifier"
        assert aegis["assigned_model"] == "deepseek-v4-pro"
        assert aegis["routing_plugin"] == "agent_workbuddy"
        assert aegis["warnings"] == []

    async def test_known_agent_inject_aegis_metadata_object_response(self, router):
        """Full flow: routing + _inject_aegis_metadata on object-style response."""
        data = {
            "messages": [
                {"role": "user", "content": "parse doc", "agent": "document_parser"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        response_obj = ObjectStyleResponse()
        BaseRouterCallback._inject_aegis_metadata(response_obj, data["metadata"])

        assert hasattr(response_obj, "aegis_metadata")
        aegis = response_obj.aegis_metadata
        assert aegis["template"] == ""
        assert aegis["agent"] == "document_parser"
        assert aegis["assigned_model"] == "gpt-5.5"
        assert aegis["routing_plugin"] == "agent_workbuddy"
        assert aegis["warnings"] == []

    async def test_different_known_agents_produce_different_aegis_metadata(self, router):
        """Each known agent produces aegis_metadata with its own agent name and model."""
        test_cases = [
            ("intent_classifier", "deepseek-v4-pro"),
            ("document_parser", "gpt-5.5"),
            ("code_assistant", "codex-mini"),
        ]

        for agent_name, expected_model in test_cases:
            data = {
                "messages": [
                    {"role": "user", "content": "request", "agent": agent_name}
                ],
                "metadata": {},
            }

            await router._execute_routing(data, "masked", "original", "hash123")

            response_obj = {}
            BaseRouterCallback._inject_aegis_metadata(response_obj, data["metadata"])

            aegis = response_obj["aegis_metadata"]
            assert aegis["agent"] == agent_name, f"Failed for agent: {agent_name}"
            assert aegis["assigned_model"] == expected_model, f"Failed for agent: {agent_name}"
            assert aegis["routing_plugin"] == "agent_workbuddy"
            assert aegis["template"] == ""
            assert aegis["warnings"] == []


# ---------------------------------------------------------------------------
# Test: Fallback path — NO_AGENT
# ---------------------------------------------------------------------------


class TestAegisMetadataNoAgent:
    """V2-8: NO_AGENT fallback path produces correct aegis_metadata."""

    async def test_no_agent_metadata_fields_for_aegis_injection(self, router):
        """After routing with no agent, metadata has correct fields."""
        data = {
            "messages": [
                {"role": "user", "content": "no agent here"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        metadata = data["metadata"]
        assert metadata["transaction_template"] == ""
        assert metadata["transaction_agent"] == ""
        assert metadata["target_model"] == "deepseek-v3"
        assert metadata["routing_plugin"] == "agent_workbuddy"
        assert metadata["_routing_warnings"] == ["NO_AGENT"]

    async def test_no_agent_inject_aegis_metadata_dict_response(self, router):
        """Full flow: no agent + inject produces correct aegis_metadata."""
        data = {
            "messages": [
                {"role": "user", "content": "no agent here"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        response_obj = {"choices": []}
        BaseRouterCallback._inject_aegis_metadata(response_obj, data["metadata"])

        aegis = response_obj["aegis_metadata"]
        assert aegis["template"] == ""
        assert aegis["agent"] == ""
        assert aegis["assigned_model"] == "deepseek-v3"
        assert aegis["routing_plugin"] == "agent_workbuddy"
        assert aegis["warnings"] == ["NO_AGENT"]


# ---------------------------------------------------------------------------
# Test: Fallback path — INVALID_AGENT
# ---------------------------------------------------------------------------


class TestAegisMetadataInvalidAgent:
    """V2-8: INVALID_AGENT fallback path produces correct aegis_metadata."""

    async def test_invalid_agent_metadata_fields_for_aegis_injection(self, router):
        """After routing with invalid agent, metadata has correct fields."""
        data = {
            "messages": [
                {"role": "user", "content": "request", "agent": "agent@invalid!"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        metadata = data["metadata"]
        assert metadata["transaction_template"] == ""
        assert metadata["transaction_agent"] == "agent@invalid!"
        assert metadata["target_model"] == "deepseek-v3"
        assert metadata["routing_plugin"] == "agent_workbuddy"
        assert metadata["_routing_warnings"] == ["INVALID_AGENT"]

    async def test_invalid_agent_inject_aegis_metadata_dict_response(self, router):
        """Full flow: invalid agent + inject produces correct aegis_metadata."""
        data = {
            "messages": [
                {"role": "user", "content": "request", "agent": "agent.with.dots"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        response_obj = {}
        BaseRouterCallback._inject_aegis_metadata(response_obj, data["metadata"])

        aegis = response_obj["aegis_metadata"]
        assert aegis["template"] == ""
        assert aegis["agent"] == "agent.with.dots"
        assert aegis["assigned_model"] == "deepseek-v3"
        assert aegis["routing_plugin"] == "agent_workbuddy"
        assert aegis["warnings"] == ["INVALID_AGENT"]


# ---------------------------------------------------------------------------
# Test: Fallback path — UNKNOWN_AGENT
# ---------------------------------------------------------------------------


class TestAegisMetadataUnknownAgent:
    """V2-8: UNKNOWN_AGENT fallback path produces correct aegis_metadata."""

    async def test_unknown_agent_metadata_fields_for_aegis_injection(self, router):
        """After routing with unknown agent, metadata has correct fields."""
        data = {
            "messages": [
                {"role": "user", "content": "request", "agent": "nonexistent_agent"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        metadata = data["metadata"]
        assert metadata["transaction_template"] == ""
        assert metadata["transaction_agent"] == "nonexistent_agent"
        assert metadata["target_model"] == "deepseek-v3"
        assert metadata["routing_plugin"] == "agent_workbuddy"
        assert metadata["_routing_warnings"] == ["UNKNOWN_AGENT"]

    async def test_unknown_agent_inject_aegis_metadata_dict_response(self, router):
        """Full flow: unknown agent + inject produces correct aegis_metadata."""
        data = {
            "messages": [
                {"role": "user", "content": "request", "agent": "mystery_bot"}
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "hash123")

        response_obj = {}
        BaseRouterCallback._inject_aegis_metadata(response_obj, data["metadata"])

        aegis = response_obj["aegis_metadata"]
        assert aegis["template"] == ""
        assert aegis["agent"] == "mystery_bot"
        assert aegis["assigned_model"] == "deepseek-v3"
        assert aegis["routing_plugin"] == "agent_workbuddy"
        assert aegis["warnings"] == ["UNKNOWN_AGENT"]


# ---------------------------------------------------------------------------
# Test: _inject_aegis_metadata on different response types
# ---------------------------------------------------------------------------


class TestInjectAegisMetadataResponseTypes:
    """V2-8: _inject_aegis_metadata works on both dict and object-style responses."""

    def _sample_metadata(self):
        """Create sample metadata as produced by _execute_routing."""
        return {
            "transaction_template": "",
            "transaction_agent": "intent_classifier",
            "target_model": "deepseek-v4-pro",
            "routing_plugin": "agent_workbuddy",
            "_routing_warnings": [],
        }

    def test_inject_into_plain_dict(self):
        """Dict response: aegis_metadata injected as key."""
        response = {}
        BaseRouterCallback._inject_aegis_metadata(response, self._sample_metadata())

        assert "aegis_metadata" in response
        assert response["aegis_metadata"]["agent"] == "intent_classifier"
        assert response["aegis_metadata"]["assigned_model"] == "deepseek-v4-pro"
        assert response["aegis_metadata"]["routing_plugin"] == "agent_workbuddy"
        assert response["aegis_metadata"]["template"] == ""
        assert response["aegis_metadata"]["warnings"] == []

    def test_inject_into_dict_subclass(self):
        """DictStyleResponse (dict subclass): aegis_metadata injected as key."""
        response = DictStyleResponse()
        BaseRouterCallback._inject_aegis_metadata(response, self._sample_metadata())

        # dict subclass: injected via dict path
        assert "aegis_metadata" in response
        assert response["aegis_metadata"]["agent"] == "intent_classifier"

    def test_inject_into_object_with_dict(self):
        """Object with __dict__: aegis_metadata set as attribute."""
        response = ObjectStyleResponse()
        BaseRouterCallback._inject_aegis_metadata(response, self._sample_metadata())

        assert hasattr(response, "aegis_metadata")
        assert response.aegis_metadata["agent"] == "intent_classifier"
        assert response.aegis_metadata["assigned_model"] == "deepseek-v4-pro"

    def test_inject_into_pydantic_like_with_hidden_params(self):
        """Pydantic-like model (rejects setattr) with _hidden_params: uses fallback."""
        response = PydanticLikeResponse()
        BaseRouterCallback._inject_aegis_metadata(response, self._sample_metadata())

        assert "aegis_metadata" in response._hidden_params
        aegis = response._hidden_params["aegis_metadata"]
        assert aegis["agent"] == "intent_classifier"
        assert aegis["assigned_model"] == "deepseek-v4-pro"
        assert aegis["routing_plugin"] == "agent_workbuddy"

    def test_inject_into_strict_pydantic_with_model_extra(self):
        """Strict pydantic model with model_extra: uses model_extra fallback."""
        response = StrictPydanticResponse()
        BaseRouterCallback._inject_aegis_metadata(response, self._sample_metadata())

        assert "aegis_metadata" in response.model_extra
        aegis = response.model_extra["aegis_metadata"]
        assert aegis["agent"] == "intent_classifier"
        assert aegis["assigned_model"] == "deepseek-v4-pro"
        assert aegis["routing_plugin"] == "agent_workbuddy"

    def test_inject_with_warnings_list(self):
        """Metadata with warnings correctly populates aegis_metadata.warnings."""
        metadata = {
            "transaction_template": "",
            "transaction_agent": "bad_agent",
            "target_model": "deepseek-v3",
            "routing_plugin": "agent_workbuddy",
            "_routing_warnings": ["UNKNOWN_AGENT"],
        }
        response = {}
        BaseRouterCallback._inject_aegis_metadata(response, metadata)

        assert response["aegis_metadata"]["warnings"] == ["UNKNOWN_AGENT"]

    def test_inject_with_empty_metadata_uses_defaults(self):
        """Empty metadata produces aegis_metadata with empty string defaults."""
        response = {}
        BaseRouterCallback._inject_aegis_metadata(response, {})

        aegis = response["aegis_metadata"]
        assert aegis["template"] == ""
        assert aegis["agent"] == ""
        assert aegis["assigned_model"] == ""
        assert aegis["routing_plugin"] == ""
        assert aegis["warnings"] == []

    def test_inject_preserves_existing_response_content(self):
        """Injecting aegis_metadata does not remove existing response fields."""
        response = {
            "choices": [{"message": {"content": "hello world"}}],
            "model": "deepseek-v4-pro",
        }
        BaseRouterCallback._inject_aegis_metadata(response, self._sample_metadata())

        # Original fields preserved
        assert response["choices"][0]["message"]["content"] == "hello world"
        assert response["model"] == "deepseek-v4-pro"
        # aegis_metadata added
        assert "aegis_metadata" in response
