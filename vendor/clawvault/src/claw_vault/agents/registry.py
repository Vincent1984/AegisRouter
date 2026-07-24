"""Registry for agent integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claw_vault.agents.base import AgentIntegration, SessionRedactionService

if TYPE_CHECKING:
    from claw_vault.config import Settings


class AgentIntegrationRegistry:
    """Keep track of available agent integrations."""

    def __init__(self) -> None:
        self._integrations: dict[str, AgentIntegration] = {}

    def register(self, integration: AgentIntegration) -> None:
        """Register an integration by its unique key."""
        key = integration.key
        if not key:
            raise ValueError("Agent integration key cannot be empty")
        if key in self._integrations:
            raise ValueError(f"Agent integration already registered: {key}")
        self._integrations[key] = integration

    def get(self, key: str) -> AgentIntegration:
        """Return a registered integration by key."""
        try:
            return self._integrations[key]
        except KeyError as exc:
            raise KeyError(f"Agent integration not registered: {key}") from exc

    def list(self) -> list[AgentIntegration]:
        """Return all registered integrations in registration order."""
        return list(self._integrations.values())

    def statuses(self) -> list[dict]:
        """Return dashboard-friendly status payloads for all integrations."""
        return [integration.status() for integration in self.list()]

    def create_session_redaction_services(
        self,
        settings: Settings,
        global_detection_config: dict[str, bool] | None = None,
    ) -> dict[str, SessionRedactionService]:
        """Create supported session redaction services keyed by integration key."""
        services: dict[str, SessionRedactionService] = {}
        for integration in self.list():
            service = integration.create_session_redaction_service(
                settings,
                global_detection_config=global_detection_config,
            )
            if service is not None:
                services[integration.key] = service
        return services
