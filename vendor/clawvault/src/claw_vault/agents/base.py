"""Base contracts for agent integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from claw_vault.config import Settings


class SessionRedactionService(Protocol):
    """Runtime service that can redact an agent's persisted session data."""

    @property
    def running(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class AgentIntegration(Protocol):
    """Minimal contract implemented by supported agent integrations."""

    @property
    def key(self) -> str: ...

    @property
    def name(self) -> str: ...

    def status(self) -> dict[str, Any]: ...

    def create_session_redaction_service(
        self,
        settings: Settings,
        global_detection_config: dict[str, bool] | None = None,
    ) -> SessionRedactionService | None: ...
