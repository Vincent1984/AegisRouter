"""Verification tests for V4-7: Audit log correctly records
prompt_hash, score, candidate model list, and final selected model.

Tests cover:
- Audit log emits valid JSON with required fields
- Single-match scenario (1 candidate)
- Overlap scenario (multiple candidates)
- Raw PII never appears in audit log output
- route_candidates stored in metadata
"""

from __future__ import annotations

import json
import logging
import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.observability.audit_logger import AuditLogger
from aegis_router.router.route_resolver import RouteResolver


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool that simulates normal responses."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock()
    pool.max_connections = 10
    return pool


@pytest.fixture
def single_match_routing_table():
    """Routing table where score 0.3 matches exactly one tier."""
    return [
        {
            "name": "local-7b",
            "model": "local-7b",
            "computed_score": 20,
            "score_range": [0.0, 0.4],
            "cost_per_1m_input": 0.0,
            "overridden": False,
        },
        {
            "name": "gpt-4o",
            "model": "gpt-4o",
            "computed_score": 80,
            "score_range": [0.6, 1.0],
            "cost_per_1m_input": 5.0,
            "overridden": False,
        },
    ]


@pytest.fixture
def overlap_routing_table():
    """Routing table where score 0.5 matches multiple tiers (overlap)."""
    return [
        {
            "name": "deepseek-chat",
            "model": "deepseek-chat",
            "computed_score": 40,
            "score_range": [0.3, 0.6],
            "cost_per_1m_input": 0.14,
            "overridden": False,
        },
        {
            "name": "gpt-4o",
            "model": "gpt-4o",
            "computed_score": 80,
            "score_range": [0.4, 0.8],
            "cost_per_1m_input": 5.0,
            "overridden": False,
        },
        {
            "name": "claude-3-opus",
            "model": "claude-3-opus",
            "computed_score": 90,
            "score_range": [0.45, 0.9],
            "cost_per_1m_input": 15.0,
            "overridden": False,
        },
    ]


@pytest.fixture
def mock_classifier_single():
    """Mock classifier that returns score 0.3 (single match in single_match table)."""
    classifier = AsyncMock()
    result = MagicMock()
    result.score = 0.3
    classifier.aclassify = AsyncMock(return_value=result)
    return classifier


@pytest.fixture
def mock_classifier_overlap():
    """Mock classifier that returns score 0.5 (overlap in overlap table)."""
    classifier = AsyncMock()
    result = MagicMock()
    result.score = 0.5
    classifier.aclassify = AsyncMock(return_value=result)
    return classifier


def build_callback(mock_pool, routing_table, classifier, strategy="lowest_cost"):
    """Build a SmartRouterCallback with given components."""
    # Create a mock config watcher
    config_watcher = MagicMock()
    config_watcher.get_current_config.return_value = None

    # Create a mock rule engine that never matches
    rule_engine = MagicMock()
    rule_result = MagicMock()
    rule_result.matched = False
    rule_engine.check.return_value = rule_result

    cb = SmartRouterCallback(
        pool=mock_pool,
        enable_routing=True,
        rule_engine=rule_engine,
        classifier=classifier,
        config_watcher=config_watcher,
    )

    # Manually set the route resolver with the provided table
    cb._route_resolver = RouteResolver(
        tiers=routing_table,
        strategy=strategy,
        fallback_model="deepseek-v3",
    )

    # Set a routing config mock for score_input
    routing_config = MagicMock()
    routing_config.score_input = "masked"
    routing_config.fallback_model = "deepseek-v3"
    cb._routing_config = routing_config

    return cb


def make_request_data(text: str) -> dict:
    """Create a request data dict with PII-laden text."""
    return {
        "messages": [{"role": "user", "content": text}],
        "model": "gpt-4o",
        "metadata": {
            "session_id": "sess-audit-test",
            "request_id": "req-audit-test",
        },
    }


# ---------------------------------------------------------------------------
# Tests: AuditLogger unit tests
# ---------------------------------------------------------------------------


class TestAuditLoggerUnit:
    """Unit tests for AuditLogger class."""

    def test_log_route_decision_returns_valid_entry(self, caplog):
        """AuditLogger.log_route_decision returns a dict with all required fields."""
        al = AuditLogger()

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_route_decision(
                request_id="req-1",
                session_id="sess-1",
                prompt_hash="a" * 64,
                prompt_length=100,
                route_score=0.45,
                candidates=["gpt-4o", "deepseek-chat"],
                target_model="deepseek-chat",
                route_reason="overlap_lowest_cost",
                latency_mask_ms=5.2,
                latency_route_ms=3.1,
                entities_detected=["PERSON", "PHONE"],
            )

        # Validate entry structure
        assert entry["event"] == "route_decision"
        assert entry["request_id"] == "req-1"
        assert entry["session_id"] == "sess-1"
        assert entry["prompt_hash"] == "a" * 64
        assert entry["prompt_length"] == 100
        assert entry["route_score"] == 0.45
        assert entry["candidates"] == ["gpt-4o", "deepseek-chat"]
        assert entry["target_model"] == "deepseek-chat"
        assert entry["route_reason"] == "overlap_lowest_cost"
        assert entry["latency_mask_ms"] == 5.2
        assert entry["latency_route_ms"] == 3.1
        assert entry["entities_detected"] == ["PERSON", "PHONE"]
        assert "ts" in entry

    def test_log_emits_valid_json(self, caplog):
        """Audit log emits single-line valid JSON."""
        al = AuditLogger()

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            al.log_route_decision(
                request_id="req-2",
                session_id="sess-2",
                prompt_hash="b" * 64,
                prompt_length=50,
                route_score=0.7,
                candidates=["gpt-4o"],
                target_model="gpt-4o",
                route_reason="single_match",
            )

        # Find the audit log record
        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1

        # Parse as JSON — must be valid
        parsed = json.loads(audit_records[0].getMessage())
        assert parsed["prompt_hash"] == "b" * 64
        assert parsed["route_score"] == 0.7
        assert parsed["candidates"] == ["gpt-4o"]
        assert parsed["target_model"] == "gpt-4o"

    def test_no_raw_pii_in_entry(self):
        """Audit log entry must never contain raw PII text."""
        al = AuditLogger()
        pii_text = "张三的手机号是13800138000"

        entry = al.log_route_decision(
            request_id="req-pii",
            session_id="sess-pii",
            prompt_hash="c" * 64,
            prompt_length=len(pii_text),
            route_score=0.5,
            candidates=["gpt-4o"],
            target_model="gpt-4o",
            route_reason="single_match",
        )

        # Serialize to string and check no PII appears
        entry_str = json.dumps(entry, ensure_ascii=False)
        assert "张三" not in entry_str
        assert "13800138000" not in entry_str


# ---------------------------------------------------------------------------
# Tests: Integration — single match scenario
# ---------------------------------------------------------------------------


class TestAuditLogSingleMatch:
    """Verify audit log in single-match routing scenario."""

    async def test_single_match_emits_audit_log(
        self, mock_pool, single_match_routing_table, mock_classifier_single, caplog
    ):
        """Single-match routing emits audit log with 1 candidate."""
        cb = build_callback(mock_pool, single_match_routing_table, mock_classifier_single)

        # Mock ClawVault responses
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "Hello, tell me about quantum physics", "entities_found": []},
        ]

        data = make_request_data("Hello, tell me about quantum physics")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        # Find audit log record
        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1

        log_entry = json.loads(audit_records[0].getMessage())

        # Verify required fields
        assert log_entry["prompt_hash"] is not None
        assert len(log_entry["prompt_hash"]) == 64  # SHA-256
        assert log_entry["route_score"] == 0.3
        assert log_entry["candidates"] == ["local-7b"]
        assert log_entry["target_model"] == "local-7b"
        assert log_entry["route_reason"] == "single_match"
        assert log_entry["event"] == "route_decision"

    async def test_single_match_stores_candidates_in_metadata(
        self, mock_pool, single_match_routing_table, mock_classifier_single
    ):
        """Single-match routing stores route_candidates list in metadata."""
        cb = build_callback(mock_pool, single_match_routing_table, mock_classifier_single)

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "Hello quantum", "entities_found": []},
        ]

        data = make_request_data("Hello quantum")
        await cb.async_pre_call_hook({}, None, data, "completion")

        assert data["metadata"]["route_candidates"] == ["local-7b"]


# ---------------------------------------------------------------------------
# Tests: Integration — overlap (multiple candidates) scenario
# ---------------------------------------------------------------------------


class TestAuditLogOverlap:
    """Verify audit log in overlap routing scenario (multiple candidates)."""

    async def test_overlap_emits_audit_log_with_multiple_candidates(
        self, mock_pool, overlap_routing_table, mock_classifier_overlap, caplog
    ):
        """Overlap routing emits audit log with multiple candidates in list."""
        cb = build_callback(mock_pool, overlap_routing_table, mock_classifier_overlap)

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "Explain the theory of relativity in detail", "entities_found": []},
        ]

        data = make_request_data("Explain the theory of relativity in detail")

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1

        log_entry = json.loads(audit_records[0].getMessage())

        # All 3 tiers overlap at score 0.5
        assert log_entry["route_score"] == 0.5
        assert len(log_entry["candidates"]) == 3
        assert "deepseek-chat" in log_entry["candidates"]
        assert "gpt-4o" in log_entry["candidates"]
        assert "claude-3-opus" in log_entry["candidates"]
        # lowest_cost strategy selects deepseek-chat
        assert log_entry["target_model"] == "deepseek-chat"
        assert "overlap" in log_entry["route_reason"]

    async def test_overlap_stores_candidates_in_metadata(
        self, mock_pool, overlap_routing_table, mock_classifier_overlap
    ):
        """Overlap routing stores full candidates list in metadata."""
        cb = build_callback(mock_pool, overlap_routing_table, mock_classifier_overlap)

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {"masked_text": "Complex question", "entities_found": []},
        ]

        data = make_request_data("Complex question")
        await cb.async_pre_call_hook({}, None, data, "completion")

        assert len(data["metadata"]["route_candidates"]) == 3
        assert "deepseek-chat" in data["metadata"]["route_candidates"]
        assert "gpt-4o" in data["metadata"]["route_candidates"]
        assert "claude-3-opus" in data["metadata"]["route_candidates"]


# ---------------------------------------------------------------------------
# Tests: PII safety — raw PII never in audit log
# ---------------------------------------------------------------------------


class TestAuditLogPIISafety:
    """Verify that raw PII never appears in audit log output."""

    async def test_pii_text_not_in_audit_log(
        self, mock_pool, single_match_routing_table, mock_classifier_single, caplog
    ):
        """Raw PII text must NOT appear in audit log — only prompt_hash."""
        cb = build_callback(mock_pool, single_match_routing_table, mock_classifier_single)

        pii_text = "我叫张三，手机号是13800138000，身份证号110101199003071234"
        masked_text = "我叫[PERSON_1]，手机号是[PHONE_1]，身份证号[ID_CARD_1]"

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": masked_text,
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.9},
                    {"type": "PHONE_NUMBER", "start": 9, "end": 20, "score": 0.95},
                    {"type": "ID_CARD", "start": 25, "end": 43, "score": 0.99},
                ],
            },
        ]

        data = make_request_data(pii_text)

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            await cb.async_pre_call_hook({}, None, data, "completion")

        # Get all audit log output as a single string
        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1

        full_log_text = audit_records[0].getMessage()

        # Assert no raw PII in audit log
        assert "张三" not in full_log_text
        assert "13800138000" not in full_log_text
        assert "110101199003071234" not in full_log_text

        # But prompt_hash IS present (SHA-256 of the original text)
        log_entry = json.loads(full_log_text)
        assert len(log_entry["prompt_hash"]) == 64


# ---------------------------------------------------------------------------
# Tests: RouteResolver returns candidates
# ---------------------------------------------------------------------------


class TestRouteResolverCandidates:
    """Verify RouteResolver.resolve() returns candidates list."""

    def test_single_match_returns_candidates_list(self, single_match_routing_table):
        """Single match returns candidates with one model."""
        resolver = RouteResolver(
            tiers=single_match_routing_table,
            strategy="lowest_cost",
            fallback_model="deepseek-v3",
        )

        result = resolver.resolve(0.3)
        assert result["candidates"] == ["local-7b"]
        assert result["model"] == "local-7b"

    def test_overlap_returns_all_candidates(self, overlap_routing_table):
        """Overlap returns all matching model names in candidates list."""
        resolver = RouteResolver(
            tiers=overlap_routing_table,
            strategy="lowest_cost",
            fallback_model="deepseek-v3",
        )

        result = resolver.resolve(0.5)
        assert len(result["candidates"]) == 3
        assert "deepseek-chat" in result["candidates"]
        assert "gpt-4o" in result["candidates"]
        assert "claude-3-opus" in result["candidates"]
        # lowest_cost should select deepseek-chat
        assert result["model"] == "deepseek-chat"

    def test_no_match_returns_empty_candidates(self, single_match_routing_table):
        """No match returns empty candidates list and fallback model."""
        resolver = RouteResolver(
            tiers=single_match_routing_table,
            strategy="lowest_cost",
            fallback_model="deepseek-v3",
        )

        # Score 0.5 doesn't match any tier in single_match_routing_table
        result = resolver.resolve(0.5)
        assert result["candidates"] == []
        assert result["model"] == "deepseek-v3"
        assert result["reason"] == "no_match_fallback"
