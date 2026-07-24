"""OpenClaw agent integration wrapper."""

from __future__ import annotations

from typing import Any

from claw_vault.config import Settings
from claw_vault.openclaw.service import OpenClawSessionRedactionService


class OpenClawIntegration:
    """Expose existing OpenClaw support through the generic integration contract."""

    key = "openclaw"
    name = "OpenClaw"

    def status(self) -> dict[str, Any]:
        """Return static integration metadata for dashboard discovery."""
        return {
            "key": self.key,
            "name": self.name,
            "enabled": True,
            "capabilities": {
                "session_redaction": True,
            },
        }

    def create_session_redaction_service(
        self,
        settings: Settings,
        global_detection_config: dict[str, bool] | None = None,
    ) -> OpenClawSessionRedactionService:
        """Create the existing OpenClaw transcript redaction service."""
        return OpenClawSessionRedactionService(
            settings.openclaw.session_redaction,
            global_detection_config=global_detection_config,
        )
