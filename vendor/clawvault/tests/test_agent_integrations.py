from __future__ import annotations

import pytest

from claw_vault.agents.registry import AgentIntegrationRegistry
from claw_vault.claude_code.integration import ClaudeCodeIntegration
from claw_vault.config import Settings
from claw_vault.openclaw.integration import OpenClawIntegration
from claw_vault.openclaw.service import OpenClawSessionRedactionService


class _DummyIntegration:
    key = "dummy"
    name = "Dummy"

    def status(self) -> dict:
        return {"key": self.key, "name": self.name, "enabled": True}

    def create_session_redaction_service(self, settings, global_detection_config=None):
        return None


def test_registry_registers_and_lists_integrations() -> None:
    registry = AgentIntegrationRegistry()
    integration = _DummyIntegration()

    registry.register(integration)

    assert registry.get("dummy") is integration
    assert registry.list() == [integration]
    assert registry.statuses() == [{"key": "dummy", "name": "Dummy", "enabled": True}]


def test_registry_rejects_duplicate_keys() -> None:
    registry = AgentIntegrationRegistry()
    registry.register(_DummyIntegration())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_DummyIntegration())


def test_registry_reports_missing_integration() -> None:
    registry = AgentIntegrationRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("missing")


def test_openclaw_integration_creates_existing_redaction_service() -> None:
    settings = Settings()
    integration = OpenClawIntegration()

    service = integration.create_session_redaction_service(settings)

    assert integration.status()["key"] == "openclaw"
    assert integration.status()["capabilities"]["session_redaction"] is True
    assert isinstance(service, OpenClawSessionRedactionService)
    assert service.sessions_root == settings.openclaw.session_redaction.sessions_root.expanduser()


def test_claude_code_integration_is_disabled_noop() -> None:
    integration = ClaudeCodeIntegration()

    status = integration.status()
    service = integration.create_session_redaction_service(Settings())

    assert status["key"] == "claude_code"
    assert status["name"] == "Claude Code"
    assert status["enabled"] is False
    assert status["capabilities"]["session_redaction"] is False
    assert service is None


def test_registry_creates_only_openclaw_session_redaction_service() -> None:
    registry = AgentIntegrationRegistry()
    registry.register(OpenClawIntegration())
    registry.register(ClaudeCodeIntegration())

    services = registry.create_session_redaction_services(Settings())

    assert set(services) == {"openclaw"}
    assert isinstance(services["openclaw"], OpenClawSessionRedactionService)


def test_registry_statuses_include_openclaw_and_claude_code_by_key() -> None:
    registry = AgentIntegrationRegistry()
    registry.register(OpenClawIntegration())
    registry.register(ClaudeCodeIntegration())

    statuses = {status["key"]: status for status in registry.statuses()}

    assert statuses["openclaw"]["enabled"] is True
    assert statuses["openclaw"]["capabilities"]["session_redaction"] is True
    assert statuses["claude_code"]["enabled"] is False
    assert statuses["claude_code"]["capabilities"]["session_redaction"] is False
