"""Disabled Claude Code integration skeleton."""

from __future__ import annotations

from typing import Any

from claw_vault.agents.base import SessionRedactionService
from claw_vault.config import Settings


class ClaudeCodeIntegration:
    """Metadata-only placeholder for future Claude Code support."""

    key = "claude_code"
    name = "Claude Code"

    def status(self) -> dict[str, Any]:
        """Return disabled no-op metadata for dashboard discovery."""
        return {
            "key": self.key,
            "name": self.name,
            "enabled": False,
            "capabilities": {
                "session_redaction": False,
            },
        }

    def create_session_redaction_service(
        self,
        settings: Settings,
        global_detection_config: dict[str, bool] | None = None,
    ) -> SessionRedactionService | None:
        """Claude Code has no runtime service in this skeleton phase."""
        return None
