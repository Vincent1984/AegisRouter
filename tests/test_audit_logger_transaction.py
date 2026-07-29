"""Unit tests for AuditLogger transaction-related audit methods.

Tests cover:
- log_plan_generation_event (FR-8.1)
- log_dispatch_event (FR-8.2)
- log_config_change_event (FR-8.3)
"""

from __future__ import annotations

import json
import logging

import pytest

from aegis_router.observability.audit_logger import AuditLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def al() -> AuditLogger:
    """Provide a fresh AuditLogger instance per test."""
    return AuditLogger()


# ---------------------------------------------------------------------------
# Tests: log_plan_generation_event (FR-8.1)
# ---------------------------------------------------------------------------


class TestLogPlanGenerationEvent:
    """Verify log_plan_generation_event produces correct audit entries."""

    def test_basic_plan_generation(self, al: AuditLogger, caplog):
        """Plan generation event records all expected fields."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_plan_generation_event(
                trigger_reason="startup",
                template_name="resume_screening",
                assignments={
                    "intent_classifier": "local-7b",
                    "resume_parser": "gemini-2.5-pro",
                },
                total_agents=2,
            )

        assert entry["event"] == "plan_generation"
        assert entry["trigger_reason"] == "startup"
        assert entry["template_name"] == "resume_screening"
        assert entry["assignments"] == {
            "intent_classifier": "local-7b",
            "resume_parser": "gemini-2.5-pro",
        }
        assert entry["total_agents"] == 2
        assert "ts" in entry

    def test_config_change_trigger(self, al: AuditLogger, caplog):
        """Plan generation triggered by config change records correct reason."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_plan_generation_event(
                trigger_reason="models.yaml",
                template_name="code_review",
                assignments={"code_analyzer": "codex-mini"},
                total_agents=1,
            )

        assert entry["trigger_reason"] == "models.yaml"
        assert entry["template_name"] == "code_review"

    def test_emits_valid_json(self, al: AuditLogger, caplog):
        """Plan generation event emits valid JSON to the audit logger."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            al.log_plan_generation_event(
                trigger_reason="startup",
                template_name="test_tpl",
                assignments={"agent_a": "model_x"},
                total_agents=1,
            )

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1
        parsed = json.loads(audit_records[0].getMessage())
        assert parsed["event"] == "plan_generation"
        assert parsed["template_name"] == "test_tpl"

    def test_empty_assignments(self, al: AuditLogger, caplog):
        """Plan generation with empty assignments works correctly."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_plan_generation_event(
                trigger_reason="startup",
                template_name="empty_template",
                assignments={},
                total_agents=0,
            )

        assert entry["assignments"] == {}
        assert entry["total_agents"] == 0

    def test_returns_entry_dict(self, al: AuditLogger):
        """Method returns the entry dictionary."""
        entry = al.log_plan_generation_event(
            trigger_reason="startup",
            template_name="test",
            assignments={"a": "b"},
            total_agents=1,
        )
        assert isinstance(entry, dict)
        assert entry["event"] == "plan_generation"


# ---------------------------------------------------------------------------
# Tests: log_dispatch_event (FR-8.2)
# ---------------------------------------------------------------------------


class TestLogDispatchEvent:
    """Verify log_dispatch_event produces correct audit entries."""

    def test_plan_dispatch(self, al: AuditLogger, caplog):
        """Normal plan dispatch records all expected fields."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_dispatch_event(
                request_id="req-001",
                template="resume_screening",
                agent="resume_parser",
                assigned_model="gemini-2.5-pro",
                reason="plan",
            )

        assert entry["event"] == "transaction_dispatch"
        assert entry["request_id"] == "req-001"
        assert entry["template"] == "resume_screening"
        assert entry["agent"] == "resume_parser"
        assert entry["assigned_model"] == "gemini-2.5-pro"
        assert entry["reason"] == "plan"
        assert entry["warnings"] == []
        assert "ts" in entry

    def test_fallback_dispatch(self, al: AuditLogger, caplog):
        """Fallback dispatch records correct reason."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_dispatch_event(
                request_id="req-002",
                template="",
                agent="",
                assigned_model="deepseek-v3",
                reason="fallback",
            )

        assert entry["reason"] == "fallback"
        assert entry["assigned_model"] == "deepseek-v3"

    def test_unknown_agent_with_warning(self, al: AuditLogger, caplog):
        """Unknown agent dispatch includes UNKNOWN_AGENT warning."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_dispatch_event(
                request_id="req-003",
                template="code_review",
                agent="nonexistent_agent",
                assigned_model="deepseek-v3",
                reason="unknown",
                warnings=["UNKNOWN_AGENT"],
            )

        assert entry["reason"] == "unknown"
        assert entry["warnings"] == ["UNKNOWN_AGENT"]

    def test_failover_dispatch(self, al: AuditLogger, caplog):
        """Failover dispatch records correct reason."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_dispatch_event(
                request_id="req-004",
                template="resume_screening",
                agent="skill_matcher",
                assigned_model="gpt-5.2",
                reason="failover",
            )

        assert entry["reason"] == "failover"

    def test_emits_valid_json(self, al: AuditLogger, caplog):
        """Dispatch event emits valid JSON to the audit logger."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            al.log_dispatch_event(
                request_id="req-json",
                template="tpl",
                agent="ag",
                assigned_model="model-x",
                reason="plan",
            )

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1
        parsed = json.loads(audit_records[0].getMessage())
        assert parsed["event"] == "transaction_dispatch"

    def test_returns_entry_dict(self, al: AuditLogger):
        """Method returns the entry dictionary."""
        entry = al.log_dispatch_event(
            request_id="req-ret",
            template="t",
            agent="a",
            assigned_model="m",
            reason="plan",
        )
        assert isinstance(entry, dict)
        assert entry["event"] == "transaction_dispatch"


# ---------------------------------------------------------------------------
# Tests: log_config_change_event (FR-8.3)
# ---------------------------------------------------------------------------


class TestLogConfigChangeEvent:
    """Verify log_config_change_event produces correct audit entries."""

    def test_basic_config_change(self, al: AuditLogger, caplog):
        """Config change event records all expected fields."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_config_change_event(
                changed_files=["models.yaml"],
                trigger_reason="models.yaml",
                plan_diff_summary={
                    "added_templates": [],
                    "removed_templates": [],
                    "changed_assignments": [
                        {
                            "template": "resume_screening",
                            "agent": "skill_matcher",
                            "old_model": "gpt-5.5",
                            "new_model": "gpt-5.2",
                        }
                    ],
                },
                total_changes=1,
            )

        assert entry["event"] == "config_change"
        assert entry["changed_files"] == ["models.yaml"]
        assert entry["trigger_reason"] == "models.yaml"
        assert entry["total_changes"] == 1
        assert "ts" in entry
        assert len(entry["plan_diff_summary"]["changed_assignments"]) == 1

    def test_multiple_changed_files(self, al: AuditLogger, caplog):
        """Config change with multiple files records all of them."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_config_change_event(
                changed_files=["models.yaml", "capability_profiles.yaml"],
                trigger_reason="models.yaml, capability_profiles.yaml",
                plan_diff_summary={
                    "added_templates": ["new_pipeline"],
                    "removed_templates": [],
                    "changed_assignments": [],
                },
                total_changes=1,
            )

        assert len(entry["changed_files"]) == 2
        assert "models.yaml" in entry["changed_files"]
        assert "capability_profiles.yaml" in entry["changed_files"]
        assert entry["plan_diff_summary"]["added_templates"] == ["new_pipeline"]

    def test_template_removal(self, al: AuditLogger, caplog):
        """Config change recording template removal."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_config_change_event(
                changed_files=["transaction_templates.yaml"],
                trigger_reason="transaction_templates.yaml",
                plan_diff_summary={
                    "added_templates": [],
                    "removed_templates": ["old_pipeline"],
                    "changed_assignments": [],
                },
                total_changes=1,
            )

        assert entry["plan_diff_summary"]["removed_templates"] == ["old_pipeline"]
        assert entry["total_changes"] == 1

    def test_no_changes(self, al: AuditLogger, caplog):
        """Config change with zero actual changes records total_changes=0."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            entry = al.log_config_change_event(
                changed_files=["models.yaml"],
                trigger_reason="models.yaml",
                plan_diff_summary={
                    "added_templates": [],
                    "removed_templates": [],
                    "changed_assignments": [],
                },
                total_changes=0,
            )

        assert entry["total_changes"] == 0

    def test_emits_valid_json(self, al: AuditLogger, caplog):
        """Config change event emits valid JSON to the audit logger."""
        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            al.log_config_change_event(
                changed_files=["models.yaml"],
                trigger_reason="models.yaml",
                plan_diff_summary={
                    "added_templates": [],
                    "removed_templates": [],
                    "changed_assignments": [],
                },
                total_changes=0,
            )

        audit_records = [
            r for r in caplog.records if r.name == "aegis_router.audit"
        ]
        assert len(audit_records) == 1
        parsed = json.loads(audit_records[0].getMessage())
        assert parsed["event"] == "config_change"

    def test_returns_entry_dict(self, al: AuditLogger):
        """Method returns the entry dictionary."""
        entry = al.log_config_change_event(
            changed_files=["models.yaml"],
            trigger_reason="models.yaml",
            plan_diff_summary={
                "added_templates": [],
                "removed_templates": [],
                "changed_assignments": [],
            },
            total_changes=0,
        )
        assert isinstance(entry, dict)
        assert entry["event"] == "config_change"
