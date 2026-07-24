"""Verification tests for V5-4: Every route_decision audit log entry
contains latency_mask_ms, latency_route_ms, and target_model fields.

Tests cover ALL routing paths:
- Rule engine match (trivial_chat)
- Classifier + resolver (normal path)
- Classifier timeout (fallback)
- Classifier error (fallback)
- No classifier (fallback)
- No resolver (fallback)
"""

from __future__ import annotations

import json
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.route_resolver import RouteResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"latency_mask_ms", "latency_route_ms", "target_model"}


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool that simulates normal responses."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock()
    pool.max_connections = 10
    return pool


@pytest.fixture
def routing_table():
    """Simple routing table with one tier."""
    return [
        {
            "name": "gpt-4o",
            "model": "gpt-4o",
            "computed_score": 80,
            "score_range": [0.0, 1.0],
            "cost_per_1m_input": 5.0,
            "overridden": False,
        },
    ]


@pytest.fixture
def mock_classifier():
    """Mock classifier returning score 0.5."""
    classifier = AsyncMock()
    result = MagicMock()
    result.score = 0.5
    classifier.aclassify = AsyncMock(return_value=result)
    return classifier


@pytest.fixture
def mock_rule_engine_match():
    """Mock rule engine that always matches (trivial chat)."""
    rule_engine = MagicMock()
    rule_result = MagicMock()
    rule_result.matched = True
    rule_result.target_model = "local-7b"
    rule_result.matched_pattern = "你好"
    rule_engine.check.return_value = rule_result
    return rule_engine


@pytest.fixture
def mock_rule_engine_no_match():
    """Mock rule engine that never matches."""
    rule_engine = MagicMock()
    rule_result = MagicMock()
    rule_result.matched = False
    rule_engine.check.return_value = rule_result
    return rule_engine


def build_callback(
    mock_pool,
    rule_engine,
    classifier=None,
    routing_table=None,
    strategy="lowest_cost",
    fallback_model="deepseek-v3",
):
    """Build a SmartRouterCallback with given components."""
    config_watcher = MagicMock()
    config_watcher.get_current_config.return_value = None

    cb = SmartRouterCallback(
        pool=mock_pool,
        enable_routing=True,
        rule_engine=rule_engine,
        classifier=classifier,
        config_watcher=config_watcher,
    )

    # Set route resolver if routing table provided
    if routing_table is not None:
        cb._route_resolver = RouteResolver(
            tiers=routing_table,
            strategy=strategy,
            fallback_model=fallback_model,
        )
    else:
        cb._route_resolver = None

    # Set routing config
    routing_config = MagicMock()
    routing_config.score_input = "masked"
    routing_config.fallback_model = fallback_model
    cb._routing_config = routing_config

    return cb


def make_request_data(text: str = "test prompt") -> dict:
    """Create a request data dict."""
    return {
        "messages": [{"role": "user", "content": text}],
        "model": "gpt-4o",
        "metadata": {
            "session_id": "sess-v54-test",
            "request_id": "req-v54-test",
            "latency_mask_ms": 5.5,
        },
    }


def extract_audit_entries(caplog) -> list[dict]:
    """Extract and parse all audit log entries from caplog."""
    audit_records = [
        r for r in caplog.records if r.name == "aegis_router.audit"
    ]
    entries = []
    for record in audit_records:
        parsed = json.loads(record.getMessage())
        if parsed.get("event") == "route_decision":
            entries.append(parsed)
    return entries


# ---------------------------------------------------------------------------
# Tests: Rule Engine Match (trivial_chat) path
# ---------------------------------------------------------------------------


class TestRuleEngineAuditFields:
    """V5-4: Rule engine match emits audit log with all 3 required fields."""

    @pytest.mark.asyncio
    async def test_rule_engine_match_has_required_fields(
        self, mock_pool, mock_rule_engine_match, caplog
    ):
        """Route via rule engine produces audit log with latency_mask_ms,
        latency_route_ms, and target_model."""
        cb = build_callback(mock_pool, mock_rule_engine_match)

        # ClawVault returns normal mask result
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "你好", "entities_found": []},
        ]

        data = make_request_data("你好")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        entries = extract_audit_entries(caplog)
        assert len(entries) == 1

        entry = entries[0]
        # All 3 required fields present
        for field in REQUIRED_FIELDS:
            assert field in entry, f"Missing field: {field}"

        assert entry["target_model"] == "local-7b"
        assert isinstance(entry["latency_mask_ms"], (int, float))
        assert isinstance(entry["latency_route_ms"], (int, float))
        assert entry["latency_route_ms"] >= 0
        assert entry["route_reason"] == "trivial_chat"


# ---------------------------------------------------------------------------
# Tests: Classifier + Resolver (normal path)
# ---------------------------------------------------------------------------


class TestClassifierResolverAuditFields:
    """V5-4: Normal classifier+resolver path emits audit log with all 3 fields."""

    @pytest.mark.asyncio
    async def test_classifier_resolver_has_required_fields(
        self, mock_pool, mock_rule_engine_no_match, mock_classifier, routing_table, caplog
    ):
        """Route via classifier+resolver produces audit log with latency_mask_ms,
        latency_route_ms, and target_model."""
        cb = build_callback(
            mock_pool, mock_rule_engine_no_match, mock_classifier, routing_table
        )

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "write a report", "entities_found": []},
        ]

        data = make_request_data("write a report")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        entries = extract_audit_entries(caplog)
        assert len(entries) == 1

        entry = entries[0]
        for field in REQUIRED_FIELDS:
            assert field in entry, f"Missing field: {field}"

        assert entry["target_model"] == "gpt-4o"
        assert isinstance(entry["latency_mask_ms"], (int, float))
        assert isinstance(entry["latency_route_ms"], (int, float))
        assert entry["route_score"] == 0.5


# ---------------------------------------------------------------------------
# Tests: Classifier Timeout (fallback)
# ---------------------------------------------------------------------------


class TestClassifierTimeoutAuditFields:
    """V5-4: Classifier timeout fallback emits audit log with all 3 fields."""

    @pytest.mark.asyncio
    async def test_classifier_timeout_has_required_fields(
        self, mock_pool, mock_rule_engine_no_match, routing_table, caplog
    ):
        """Route via classifier timeout produces audit log with latency_mask_ms,
        latency_route_ms, and target_model."""
        # Classifier that raises TimeoutError
        classifier = AsyncMock()
        classifier.aclassify = AsyncMock(side_effect=TimeoutError("timeout"))

        cb = build_callback(
            mock_pool, mock_rule_engine_no_match, classifier, routing_table
        )

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "complex question", "entities_found": []},
        ]

        data = make_request_data("complex question")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        entries = extract_audit_entries(caplog)
        assert len(entries) == 1

        entry = entries[0]
        for field in REQUIRED_FIELDS:
            assert field in entry, f"Missing field: {field}"

        assert entry["target_model"] == "deepseek-v3"
        assert entry["route_reason"] == "classifier_timeout"
        assert isinstance(entry["latency_mask_ms"], (int, float))
        assert isinstance(entry["latency_route_ms"], (int, float))


# ---------------------------------------------------------------------------
# Tests: Classifier Error (fallback)
# ---------------------------------------------------------------------------


class TestClassifierErrorAuditFields:
    """V5-4: Classifier error fallback emits audit log with all 3 fields."""

    @pytest.mark.asyncio
    async def test_classifier_error_has_required_fields(
        self, mock_pool, mock_rule_engine_no_match, routing_table, caplog
    ):
        """Route via classifier error produces audit log with latency_mask_ms,
        latency_route_ms, and target_model."""
        # Classifier that raises RuntimeError
        classifier = AsyncMock()
        classifier.aclassify = AsyncMock(side_effect=RuntimeError("model unavailable"))

        cb = build_callback(
            mock_pool, mock_rule_engine_no_match, classifier, routing_table
        )

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "some question", "entities_found": []},
        ]

        data = make_request_data("some question")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        entries = extract_audit_entries(caplog)
        assert len(entries) == 1

        entry = entries[0]
        for field in REQUIRED_FIELDS:
            assert field in entry, f"Missing field: {field}"

        assert entry["target_model"] == "deepseek-v3"
        assert entry["route_reason"] == "classifier_error"
        assert isinstance(entry["latency_mask_ms"], (int, float))
        assert isinstance(entry["latency_route_ms"], (int, float))


# ---------------------------------------------------------------------------
# Tests: No Classifier (fallback)
# ---------------------------------------------------------------------------


class TestNoClassifierAuditFields:
    """V5-4: No classifier path emits audit log with all 3 fields."""

    @pytest.mark.asyncio
    async def test_no_classifier_has_required_fields(
        self, mock_pool, mock_rule_engine_no_match, routing_table, caplog
    ):
        """Route with no classifier produces audit log with latency_mask_ms,
        latency_route_ms, and target_model."""
        # No classifier at all
        cb = build_callback(
            mock_pool, mock_rule_engine_no_match, None, routing_table
        )

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "hello world", "entities_found": []},
        ]

        data = make_request_data("hello world")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        entries = extract_audit_entries(caplog)
        assert len(entries) == 1

        entry = entries[0]
        for field in REQUIRED_FIELDS:
            assert field in entry, f"Missing field: {field}"

        assert entry["target_model"] == "deepseek-v3"
        assert entry["route_reason"] == "no_classifier"
        assert isinstance(entry["latency_mask_ms"], (int, float))
        assert isinstance(entry["latency_route_ms"], (int, float))


# ---------------------------------------------------------------------------
# Tests: No Resolver (fallback)
# ---------------------------------------------------------------------------


class TestNoResolverAuditFields:
    """V5-4: No resolver path emits audit log with all 3 fields."""

    @pytest.mark.asyncio
    async def test_no_resolver_has_required_fields(
        self, mock_pool, mock_rule_engine_no_match, mock_classifier, caplog
    ):
        """Route with no resolver produces audit log with latency_mask_ms,
        latency_route_ms, and target_model."""
        # Pass routing_table=None so resolver is None
        cb = build_callback(
            mock_pool, mock_rule_engine_no_match, mock_classifier, None
        )

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "any text", "entities_found": []},
        ]

        data = make_request_data("any text")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        entries = extract_audit_entries(caplog)
        assert len(entries) == 1

        entry = entries[0]
        for field in REQUIRED_FIELDS:
            assert field in entry, f"Missing field: {field}"

        assert entry["target_model"] == "deepseek-v3"
        assert entry["route_reason"] == "no_resolver"
        assert isinstance(entry["latency_mask_ms"], (int, float))
        assert isinstance(entry["latency_route_ms"], (int, float))


# ---------------------------------------------------------------------------
# Tests: Cross-cutting — every route_decision has all 3 fields
# ---------------------------------------------------------------------------


class TestAllPathsHaveRequiredFields:
    """V5-4: Comprehensive check that EVERY routing path emits all 3 fields."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scenario,classifier_behavior,has_resolver",
        [
            ("classifier_timeout", "timeout", True),
            ("classifier_error", "error", True),
            ("no_classifier", "none", True),
            ("no_resolver", "normal", False),
        ],
    )
    async def test_all_fallback_paths_have_three_fields(
        self,
        mock_pool,
        mock_rule_engine_no_match,
        routing_table,
        scenario,
        classifier_behavior,
        has_resolver,
        caplog,
    ):
        """Parametrized test: every fallback path includes latency_mask_ms,
        latency_route_ms, and target_model."""
        # Set up classifier
        if classifier_behavior == "timeout":
            classifier = AsyncMock()
            classifier.aclassify = AsyncMock(side_effect=TimeoutError())
        elif classifier_behavior == "error":
            classifier = AsyncMock()
            classifier.aclassify = AsyncMock(side_effect=RuntimeError("fail"))
        elif classifier_behavior == "none":
            classifier = None
        else:
            classifier = AsyncMock()
            result = MagicMock()
            result.score = 0.5
            classifier.aclassify = AsyncMock(return_value=result)

        rt = routing_table if has_resolver else None
        cb = build_callback(mock_pool, mock_rule_engine_no_match, classifier, rt)

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "test input", "entities_found": []},
        ]

        data = make_request_data("test input")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        entries = extract_audit_entries(caplog)
        assert len(entries) == 1, f"Expected 1 audit entry for {scenario}, got {len(entries)}"

        entry = entries[0]
        for field in REQUIRED_FIELDS:
            assert field in entry, (
                f"[{scenario}] Missing field '{field}' in route_decision audit log"
            )
