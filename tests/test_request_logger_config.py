"""Unit tests for RequestLoggingConfig and load_request_logging_config.

Validates Requirements 6.1, 6.2, 6.3, 6.5:
- Config model fields and defaults
- Config loading from YAML
- Fallback to enabled=False on missing/invalid config
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from aegis_router.observability.request_logger import (
    RequestLoggingConfig,
    load_request_logging_config,
)


# ---------------------------------------------------------------------------
# Tests: RequestLoggingConfig model
# ---------------------------------------------------------------------------


class TestRequestLoggingConfig:
    """Verify RequestLoggingConfig Pydantic model fields and defaults."""

    def test_default_values(self):
        """Default config has expected field values."""
        cfg = RequestLoggingConfig()
        assert cfg.enabled is True
        assert cfg.output == "file"
        assert cfg.file_path == "./logs/request_log.jsonl"
        assert cfg.max_message_length == 4096
        assert cfg.retention_days == 30
        assert cfg.log_level == "INFO"

    def test_custom_values(self):
        """Config accepts custom field values."""
        cfg = RequestLoggingConfig(
            enabled=False,
            output="both",
            file_path="/var/log/aegis.jsonl",
            max_message_length=0,
            retention_days=7,
            log_level="DEBUG",
        )
        assert cfg.enabled is False
        assert cfg.output == "both"
        assert cfg.file_path == "/var/log/aegis.jsonl"
        assert cfg.max_message_length == 0
        assert cfg.retention_days == 7
        assert cfg.log_level == "DEBUG"

    def test_output_literal_validation(self):
        """Config rejects invalid output values."""
        with pytest.raises(Exception):
            RequestLoggingConfig(output="invalid")

    def test_output_accepts_stdout(self):
        """Config accepts 'stdout' output."""
        cfg = RequestLoggingConfig(output="stdout")
        assert cfg.output == "stdout"

    def test_output_accepts_both(self):
        """Config accepts 'both' output."""
        cfg = RequestLoggingConfig(output="both")
        assert cfg.output == "both"


# ---------------------------------------------------------------------------
# Tests: load_request_logging_config
# ---------------------------------------------------------------------------


class TestLoadRequestLoggingConfig:
    """Verify load_request_logging_config behavior."""

    def test_missing_config_dir(self, tmp_path):
        """Returns enabled=False when config directory doesn't exist."""
        cfg = load_request_logging_config(tmp_path / "nonexistent")
        assert cfg.enabled is False

    def test_missing_config_file(self, tmp_path):
        """Returns enabled=False when config.yaml doesn't exist."""
        cfg = load_request_logging_config(tmp_path)
        assert cfg.enabled is False

    def test_empty_yaml(self, tmp_path):
        """Returns enabled=False when YAML is empty."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("", encoding="utf-8")
        cfg = load_request_logging_config(tmp_path)
        assert cfg.enabled is False

    def test_no_request_logging_section(self, tmp_path):
        """Returns enabled=False when request_logging section is absent."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []\n", encoding="utf-8")
        cfg = load_request_logging_config(tmp_path)
        assert cfg.enabled is False

    def test_valid_section(self, tmp_path):
        """Loads config correctly from a valid request_logging section."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "request_logging:\n"
            "  enabled: true\n"
            '  output: "both"\n'
            '  file_path: "/tmp/test.jsonl"\n'
            "  max_message_length: 2048\n"
            "  retention_days: 14\n"
            '  log_level: "DEBUG"\n',
            encoding="utf-8",
        )
        cfg = load_request_logging_config(tmp_path)
        assert cfg.enabled is True
        assert cfg.output == "both"
        assert cfg.file_path == "/tmp/test.jsonl"
        assert cfg.max_message_length == 2048
        assert cfg.retention_days == 14
        assert cfg.log_level == "DEBUG"

    def test_partial_section_uses_defaults(self, tmp_path):
        """Missing fields in request_logging section use defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "request_logging:\n  enabled: true\n",
            encoding="utf-8",
        )
        cfg = load_request_logging_config(tmp_path)
        assert cfg.enabled is True
        assert cfg.output == "file"
        assert cfg.max_message_length == 4096
        assert cfg.retention_days == 30

    def test_invalid_output_value(self, tmp_path):
        """Returns enabled=False when output value is invalid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            'request_logging:\n  output: "invalid"\n',
            encoding="utf-8",
        )
        cfg = load_request_logging_config(tmp_path)
        assert cfg.enabled is False

    def test_malformed_yaml(self, tmp_path):
        """Returns enabled=False when YAML is malformed."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "request_logging:\n  enabled: [[[invalid yaml\n",
            encoding="utf-8",
        )
        cfg = load_request_logging_config(tmp_path)
        assert cfg.enabled is False

    def test_loads_from_real_config(self):
        """Loads config from the project's actual config directory."""
        cfg = load_request_logging_config(config_dir="./config")
        assert cfg.enabled is True
        assert cfg.output == "file"
        assert cfg.file_path == "./logs/request_log.jsonl"
        assert cfg.max_message_length == 4096
        assert cfg.retention_days == 30
