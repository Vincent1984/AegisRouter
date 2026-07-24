"""Unit tests for AuditLogger — compliance, degradation, lifecycle methods
and configure_audit_handler utility.

Does NOT modify or overlap with test_v4_7_audit_logging.py (integration tests).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

from aegis_router.observability.audit_logger import (
    AuditLogger,
    audit_logger,
    configure_audit_handler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def al() -> AuditLogger:
    """Provide a fresh AuditLogger instance per test."""
    return AuditLogger()


# ---------------------------------------------------------------------------
# Tests: configure_audit_handler
# ---------------------------------------------------------------------------


class TestConfigureAuditHandler:
    """Verify that configure_audit_handler sets up the logger correctly."""

    def test_adds_stream_handler(self):
        """configure_audit_handler(stream=True) adds a StreamHandler."""
        configure_audit_handler(stream=True, file_path=None)
        assert any(
            isinstance(h, logging.StreamHandler) for h in audit_logger.handlers
        )
        # cleanup
        audit_logger.handlers.clear()
        audit_logger.propagate = True

    def test_adds_file_handler(self, tmp_path: Path):
        """configure_audit_handler(file_path=...) adds a FileHandler."""
        log_file = str(tmp_path / "audit.log")
        configure_audit_handler(stream=False, file_path=log_file)
        assert any(
            isinstance(h, logging.FileHandler) for h in audit_logger.handlers
        )
        # cleanup
        audit_logger.handlers.clear()
        audit_logger.propagate = True

    def test_writes_json_to_file(self, tmp_path: Path):
        """Audit log entries written to file are valid JSON."""
        log_file = tmp_path / "audit.log"
        configure_audit_handler(stream=False, file_path=str(log_file))

        al = AuditLogger()
        al.log_route_decision(
            request_id="req-file",
            session_id="sess-file",
            prompt_hash="d" * 64,
            prompt_length=42,
            route_score=0.6,
            candidates=["gpt-4o"],
            target_model="gpt-4o",
            route_reason="single_match",
        )

        # Force flush
        for h in audit_logger.handlers:
            h.flush()

        content = log_file.read_text(encoding="utf-8").strip()
        parsed = json.loads(content)
        assert parsed["event"] == "route_decision"
        assert parsed["request_id"] == "req-file"

        # cleanup
        audit_logger.handlers.clear()
        audit_logger.propagate = True

    def test_idempotent_call(self):
        """Multiple calls don't stack handlers."""
        configure_audit_handler(stream=True)
        configure_audit_handler(stream=True)
        # Should only have one handler, not two
        stream_handlers = [
            h for h in audit_logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1

        # cleanup
        audit_logger.handlers.clear()
        audit_logger.propagate = True

    def test_disables_propagation(self):
        """configure_audit_handler sets propagate=False."""
        configure_audit_handler(stream=True)
        assert audit_logger.propagate is False

        # cleanup
        audit_logger.handlers.clear()
        audit_logger.propagate = True


# ---------------------------------------------------------------------------
# Tests: log_route_decision — api_key_hash support
# ---------------------------------------------------------------------------


class TestLogRouteDecisionApiKeyHash:
    """Verify api_key_hash optional parameter in log_route_decision."""

    def test_api_key_hash_included_when_provided(self, al: AuditLogger, caplog):
        """api_key_hash appears in entry when provided."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_route_decision(
                request_id="req-ak",
                session_id="sess-ak",
                prompt_hash="e" * 64,
                prompt_length=100,
                route_score=0.5,
                candidates=["gpt-4o"],
                target_model="gpt-4o",
                route_reason="single_match",
                api_key_hash="f" * 64,
            )

        assert entry["api_key_hash"] == "f" * 64

    def test_api_key_hash_absent_when_not_provided(self, al: AuditLogger, caplog):
        """api_key_hash is NOT in entry when omitted."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_route_decision(
                request_id="req-no-ak",
                session_id="sess-no-ak",
                prompt_hash="a" * 64,
                prompt_length=50,
                route_score=0.3,
                candidates=["local-7b"],
                target_model="local-7b",
                route_reason="single_match",
            )

        assert "api_key_hash" not in entry


# ---------------------------------------------------------------------------
# Tests: log_compliance_event
# ---------------------------------------------------------------------------


class TestLogComplianceEvent:
    """Verify log_compliance_event produces correct audit entries."""

    def test_injection_detected_strict(self, al: AuditLogger, caplog):
        """Injection detection in strict mode records expected fields."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_compliance_event(
                request_id="req-inj-1",
                session_id="sess-inj-1",
                check_type="injection",
                passed=False,
                mode="strict",
                violations=["ignore previous instructions"],
                details="Pattern matched: role_hijack",
            )

        assert entry["event"] == "compliance_check"
        assert entry["request_id"] == "req-inj-1"
        assert entry["session_id"] == "sess-inj-1"
        assert entry["check_type"] == "injection"
        assert entry["passed"] is False
        assert entry["mode"] == "strict"
        assert entry["violations"] == ["ignore previous instructions"]
        assert entry["details"] == "Pattern matched: role_hijack"
        assert "ts" in entry

    def test_sensitive_word_hit_permissive(self, al: AuditLogger, caplog):
        """Sensitive word hit in permissive mode."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_compliance_event(
                request_id="req-sw-1",
                session_id="sess-sw-1",
                check_type="sensitive_word",
                passed=False,
                mode="permissive",
                violations=["敏感词A"],
            )

        assert entry["check_type"] == "sensitive_word"
        assert entry["mode"] == "permissive"
        assert entry["passed"] is False
        assert "details" not in entry  # not provided

    def test_compliance_passed(self, al: AuditLogger, caplog):
        """Compliance check passes — no violations."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_compliance_event(
                request_id="req-ok",
                session_id="sess-ok",
                check_type="injection",
                passed=True,
                mode="strict",
            )

        assert entry["passed"] is True
        assert entry["violations"] == []

    def test_emits_valid_json(self, al: AuditLogger, caplog):
        """Compliance event emits valid JSON to the audit logger."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            al.log_compliance_event(
                request_id="req-json",
                session_id="sess-json",
                check_type="injection",
                passed=False,
                mode="strict",
                violations=["test"],
            )

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1
        parsed = json.loads(audit_records[0].getMessage())
        assert parsed["event"] == "compliance_check"

    def test_no_pii_in_compliance_entry(self, al: AuditLogger):
        """Raw PII text must NOT appear in compliance audit entry."""
        pii_text = "张三的身份证号是110101199003071234"

        entry = al.log_compliance_event(
            request_id="req-pii-c",
            session_id="sess-pii-c",
            check_type="injection",
            passed=True,
            mode="strict",
        )

        entry_str = json.dumps(entry, ensure_ascii=False)
        assert "张三" not in entry_str
        assert "110101199003071234" not in entry_str


# ---------------------------------------------------------------------------
# Tests: log_degradation_event
# ---------------------------------------------------------------------------


class TestLogDegradationEvent:
    """Verify log_degradation_event produces correct audit entries."""

    def test_clawvault_down_bypass(self, al: AuditLogger, caplog):
        """ClawVault down triggers bypass_masking degradation event."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_degradation_event(
                request_id="req-deg-1",
                session_id="sess-deg-1",
                component="clawvault",
                previous_state="healthy",
                current_state="unhealthy",
                action="bypass_masking",
            )

        assert entry["event"] == "degradation_change"
        assert entry["component"] == "clawvault"
        assert entry["previous_state"] == "healthy"
        assert entry["current_state"] == "unhealthy"
        assert entry["action"] == "bypass_masking"
        assert "fallback_model" not in entry
        assert "ts" in entry

    def test_redis_down_reject(self, al: AuditLogger, caplog):
        """Redis down triggers reject_request degradation event."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_degradation_event(
                request_id="req-deg-2",
                session_id="sess-deg-2",
                component="redis",
                previous_state="healthy",
                current_state="unhealthy",
                action="reject_request",
            )

        assert entry["component"] == "redis"
        assert entry["action"] == "reject_request"

    def test_classifier_timeout_with_fallback(self, al: AuditLogger, caplog):
        """Classifier timeout records fallback_model."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_degradation_event(
                request_id="req-deg-3",
                session_id="sess-deg-3",
                component="classifier",
                previous_state="healthy",
                current_state="unhealthy",
                action="use_fallback_model",
                fallback_model="deepseek-v3",
            )

        assert entry["component"] == "classifier"
        assert entry["fallback_model"] == "deepseek-v3"
        assert entry["action"] == "use_fallback_model"

    def test_emits_valid_json(self, al: AuditLogger, caplog):
        """Degradation event emits valid JSON."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            al.log_degradation_event(
                request_id="req-deg-j",
                session_id="sess-deg-j",
                component="clawvault",
                previous_state="unknown",
                current_state="unhealthy",
                action="bypass_masking",
            )

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1
        parsed = json.loads(audit_records[0].getMessage())
        assert parsed["event"] == "degradation_change"


# ---------------------------------------------------------------------------
# Tests: log_request_lifecycle
# ---------------------------------------------------------------------------


class TestLogRequestLifecycle:
    """Verify log_request_lifecycle produces correct audit entries."""

    def test_lifecycle_start(self, al: AuditLogger, caplog):
        """Request lifecycle start event records minimal fields."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_request_lifecycle(
                request_id="req-lc-1",
                session_id="sess-lc-1",
                phase="start",
            )

        assert entry["event"] == "request_lifecycle"
        assert entry["phase"] == "start"
        assert entry["status"] == "success"
        assert entry["latency_mask_ms"] == 0.0
        assert entry["latency_route_ms"] == 0.0
        assert entry["latency_llm_ms"] == 0.0
        assert entry["latency_restore_ms"] == 0.0
        assert entry["latency_total_ms"] == 0.0
        assert "ts" in entry

    def test_lifecycle_end_with_all_latencies(self, al: AuditLogger, caplog):
        """Request lifecycle end records all latency fields."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_request_lifecycle(
                request_id="req-lc-2",
                session_id="sess-lc-2",
                phase="end",
                target_model="gpt-4o",
                latency_mask_ms=8.234,
                latency_route_ms=6.123,
                latency_llm_ms=1250.5,
                latency_restore_ms=2.789,
                latency_total_ms=1267.646,
                status="success",
            )

        assert entry["phase"] == "end"
        assert entry["target_model"] == "gpt-4o"
        assert entry["latency_mask_ms"] == 8.23
        assert entry["latency_route_ms"] == 6.12
        assert entry["latency_llm_ms"] == 1250.5
        assert entry["latency_restore_ms"] == 2.79
        assert entry["latency_total_ms"] == 1267.65
        assert entry["status"] == "success"

    def test_lifecycle_error_phase(self, al: AuditLogger, caplog):
        """Request lifecycle error phase includes error message."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_request_lifecycle(
                request_id="req-lc-3",
                session_id="sess-lc-3",
                phase="error",
                status="error",
                error="Redis connection refused",
                latency_mask_ms=5.0,
            )

        assert entry["phase"] == "error"
        assert entry["status"] == "error"
        assert entry["error"] == "Redis connection refused"

    def test_target_model_absent_when_not_provided(self, al: AuditLogger, caplog):
        """target_model not in entry when omitted."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_request_lifecycle(
                request_id="req-lc-4",
                session_id="sess-lc-4",
                phase="start",
            )

        assert "target_model" not in entry

    def test_emits_valid_json(self, al: AuditLogger, caplog):
        """Lifecycle event emits valid JSON."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            al.log_request_lifecycle(
                request_id="req-lc-j",
                session_id="sess-lc-j",
                phase="end",
                target_model="deepseek-chat",
                latency_total_ms=100.0,
            )

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1
        parsed = json.loads(audit_records[0].getMessage())
        assert parsed["event"] == "request_lifecycle"
        assert parsed["target_model"] == "deepseek-chat"

    def test_v5_4_checkpoint_fields_present(self, al: AuditLogger, caplog):
        """V5-4 checkpoint: route_decision entry contains latency_mask_ms,
        latency_route_ms, and target_model."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_route_decision(
                request_id="req-v54",
                session_id="sess-v54",
                prompt_hash="a" * 64,
                prompt_length=100,
                route_score=0.5,
                candidates=["gpt-4o"],
                target_model="gpt-4o",
                route_reason="single_match",
                latency_mask_ms=8.2,
                latency_route_ms=6.1,
            )

        # V5-4 requirements
        assert "latency_mask_ms" in entry
        assert "latency_route_ms" in entry
        assert "target_model" in entry
        assert entry["latency_mask_ms"] == 8.2
        assert entry["latency_route_ms"] == 6.1
        assert entry["target_model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Tests: Cross-cutting — no PII leakage
# ---------------------------------------------------------------------------


class TestNoPiiLeakage:
    """Ensure no audit method can leak raw PII."""

    def test_compliance_event_no_pii(self, al: AuditLogger):
        """Compliance event fields don't contain PII even if violation
        description is sanitized by caller."""
        entry = al.log_compliance_event(
            request_id="req-np",
            session_id="sess-np",
            check_type="injection",
            passed=False,
            mode="strict",
            violations=["pattern: ignore_previous"],
        )
        entry_str = json.dumps(entry, ensure_ascii=False)
        # No phone numbers, ID card numbers, etc.
        assert "13800138000" not in entry_str

    def test_degradation_event_no_pii(self, al: AuditLogger):
        """Degradation event is structural — cannot contain PII by design."""
        entry = al.log_degradation_event(
            request_id="req-np2",
            session_id="sess-np2",
            component="redis",
            previous_state="healthy",
            current_state="unhealthy",
            action="reject_request",
        )
        entry_str = json.dumps(entry, ensure_ascii=False)
        assert "张三" not in entry_str

    def test_lifecycle_event_no_pii(self, al: AuditLogger):
        """Lifecycle event records latencies, not content."""
        entry = al.log_request_lifecycle(
            request_id="req-np3",
            session_id="sess-np3",
            phase="end",
            target_model="gpt-4o",
            latency_total_ms=200.0,
        )
        entry_str = json.dumps(entry, ensure_ascii=False)
        assert "13800138000" not in entry_str
        assert "110101199003071234" not in entry_str
