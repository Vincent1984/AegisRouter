"""V6-4 Verification: Kill ClawVault process → Supervisor auto-restart within 5s.

This test validates the supervisord.conf configuration ensures ClawVault will be
automatically restarted by Supervisor when the process dies unexpectedly.

Validates:
1. autorestart=true is set for [program:clawvault]
2. startretries >= 3 (sufficient retry attempts)
3. startsecs is configured (process must stay up for this duration)
4. priority is set (startup order)
5. The combination of settings ensures recovery within 5 seconds

Manual Docker Verification Steps:
---------------------------------
1. Build and start the container:
   $ docker build -t aegisrouter .
   $ docker run -d --name aegis aegisrouter

2. Verify ClawVault is running:
   $ docker exec aegis supervisorctl status clawvault
   # Expected: clawvault RUNNING pid <PID>, uptime <TIME>

3. Kill the ClawVault process:
   $ docker exec aegis pkill -f "aegis_router.clawvault.server"

4. Wait 2-3 seconds and check status again:
   $ docker exec aegis supervisorctl status clawvault
   # Expected: clawvault RUNNING pid <NEW_PID>, uptime 0:00:0x

5. Alternatively, run the automated script:
   $ docker exec aegis /bin/bash /app/scripts/verify_supervisor_restart.sh

The recovery time budget (5s) breaks down as:
- Process death detection: ~0s (Supervisor monitors child PIDs directly)
- startsecs=2: process must stay up 2s to be considered "started"
- Total expected: < 3s from kill to RUNNING state
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

import pytest

# Path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPERVISORD_CONF = PROJECT_ROOT / "supervisord.conf"


@pytest.fixture
def supervisor_config() -> configparser.ConfigParser:
    """Parse the supervisord.conf file."""
    assert SUPERVISORD_CONF.exists(), (
        f"supervisord.conf not found at {SUPERVISORD_CONF}"
    )
    config = configparser.ConfigParser()
    config.read(str(SUPERVISORD_CONF), encoding="utf-8")
    return config


@pytest.fixture
def clawvault_section(supervisor_config: configparser.ConfigParser) -> dict:
    """Extract [program:clawvault] section as a dict."""
    section_name = "program:clawvault"
    assert section_name in supervisor_config.sections(), (
        f"Section [{section_name}] not found in supervisord.conf. "
        f"Available sections: {supervisor_config.sections()}"
    )
    return dict(supervisor_config[section_name])


class TestSupervisorAutoRestart:
    """Verify supervisord.conf has correct auto-restart configuration for ClawVault."""

    def test_autorestart_enabled(self, clawvault_section: dict):
        """autorestart=true ensures Supervisor restarts ClawVault on unexpected exit."""
        assert "autorestart" in clawvault_section, (
            "autorestart not configured for [program:clawvault]"
        )
        assert clawvault_section["autorestart"].lower() == "true", (
            f"autorestart should be 'true', got '{clawvault_section['autorestart']}'"
        )

    def test_startretries_sufficient(self, clawvault_section: dict):
        """startretries >= 3 gives enough attempts for transient failures."""
        assert "startretries" in clawvault_section, (
            "startretries not configured for [program:clawvault]"
        )
        retries = int(clawvault_section["startretries"])
        assert retries >= 3, (
            f"startretries should be >= 3, got {retries}. "
            "Insufficient retries may cause permanent failure on transient errors."
        )

    def test_startsecs_configured(self, clawvault_section: dict):
        """startsecs defines how long process must stay up to be considered started."""
        assert "startsecs" in clawvault_section, (
            "startsecs not configured for [program:clawvault]. "
            "Without this, Supervisor cannot determine if the process started successfully."
        )
        startsecs = int(clawvault_section["startsecs"])
        assert startsecs > 0, (
            f"startsecs should be > 0, got {startsecs}"
        )
        # startsecs must be low enough to allow 5s total recovery
        assert startsecs <= 5, (
            f"startsecs={startsecs} is too high for 5s recovery requirement. "
            "Process startup confirmation must complete within the 5s budget."
        )

    def test_priority_configured(self, clawvault_section: dict):
        """priority ensures ClawVault starts before dependent services."""
        assert "priority" in clawvault_section, (
            "priority not configured for [program:clawvault]"
        )
        priority = int(clawvault_section["priority"])
        assert priority > 0, f"priority should be positive, got {priority}"

    def test_recovery_within_5s_budget(self, clawvault_section: dict):
        """Combined settings allow recovery within the 5-second budget.

        Recovery time = process detection (~0s) + startup time (startsecs).
        Supervisor detects child PID death immediately via SIGCHLD,
        so the restart begins without delay.
        """
        startsecs = int(clawvault_section.get("startsecs", "0"))
        # Supervisor restarts immediately on process death.
        # The process is considered "started" after startsecs.
        # Total time = ~0 (detection) + startsecs (startup validation)
        # Must be within 5s budget.
        estimated_recovery = startsecs + 0.5  # 0.5s buffer for process spawn
        assert estimated_recovery <= 5.0, (
            f"Estimated recovery time {estimated_recovery}s exceeds 5s budget. "
            f"startsecs={startsecs} is too high."
        )


class TestSupervisorGlobalConfig:
    """Verify global supervisor settings support the restart workflow."""

    def test_nodaemon_true(self, supervisor_config: configparser.ConfigParser):
        """supervisord runs in foreground (required for Docker containers)."""
        assert "supervisord" in supervisor_config.sections()
        nodaemon = supervisor_config["supervisord"].get("nodaemon", "false")
        assert nodaemon.lower() == "true", (
            "supervisord must run with nodaemon=true in Docker containers"
        )

    def test_clawvault_command_defined(self, clawvault_section: dict):
        """ClawVault command is properly defined."""
        assert "command" in clawvault_section, (
            "command not configured for [program:clawvault]"
        )
        command = clawvault_section["command"]
        assert "aegis_router.clawvault" in command, (
            f"Command does not reference aegis_router.clawvault: {command}"
        )

    def test_clawvault_directory_set(self, clawvault_section: dict):
        """Working directory is set for ClawVault."""
        assert "directory" in clawvault_section, (
            "directory not configured for [program:clawvault]"
        )


class TestSupervisorConfFileIntegrity:
    """Verify the supervisord.conf file is well-formed and complete."""

    def test_conf_file_exists(self):
        """supervisord.conf exists at project root."""
        assert SUPERVISORD_CONF.exists()

    def test_conf_file_not_empty(self):
        """supervisord.conf is not empty."""
        content = SUPERVISORD_CONF.read_text(encoding="utf-8")
        assert len(content.strip()) > 0

    def test_conf_has_required_sections(self, supervisor_config: configparser.ConfigParser):
        """supervisord.conf has all required Supervisor sections."""
        required = ["supervisord", "program:clawvault"]
        for section in required:
            assert section in supervisor_config.sections(), (
                f"Missing required section [{section}]"
            )

    def test_dockerfile_uses_supervisord(self):
        """Dockerfile CMD uses supervisord to start the container."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"
        content = dockerfile.read_text(encoding="utf-8")
        assert "supervisord" in content, (
            "Dockerfile does not reference supervisord in CMD"
        )
        # Verify it references the config file
        assert "supervisord.conf" in content or "/etc/supervisord.conf" in content, (
            "Dockerfile CMD should reference supervisord.conf"
        )
