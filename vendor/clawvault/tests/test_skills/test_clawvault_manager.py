"""Tests for the standalone OpenClaw ClawVault manager."""
# ruff: noqa: S101

from __future__ import annotations

import importlib.util
import subprocess
import time
from pathlib import Path

import yaml

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "tophant-clawvault-installer"
    / "clawvault_manager.py"
)
spec = importlib.util.spec_from_file_location("standalone_clawvault_manager", MODULE_PATH)
assert spec is not None and spec.loader is not None
manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager_module)
ClawVaultManager = manager_module.ClawVaultManager


class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_start_services_does_not_restart_gateway_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    gateway_dir = tmp_path / ".config/systemd/user"
    gateway_dir.mkdir(parents=True)
    (gateway_dir / "openclaw-gateway.service").write_text("[Service]\n")

    manager = ClawVaultManager()
    bin_dir = manager.venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "clawvault").write_text("#!/bin/sh\n")

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    checks = iter(
        [
            {"proxy_running": False, "dashboard_running": False},
            {"proxy_running": True, "dashboard_running": True},
        ]
    )
    monkeypatch.setattr(manager, "_check_services", lambda: next(checks))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    result = manager._start_services()

    assert result["gateway_restart"]["skipped"] is True
    assert ["systemctl", "--user", "restart", "openclaw-gateway"] not in commands


def test_start_services_restarts_gateway_only_when_requested(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    gateway_dir = tmp_path / ".config/systemd/user"
    gateway_dir.mkdir(parents=True)
    (gateway_dir / "openclaw-gateway.service").write_text("[Service]\n")

    manager = ClawVaultManager()
    bin_dir = manager.venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "clawvault").write_text("#!/bin/sh\n")

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    checks = iter(
        [
            {"proxy_running": False, "dashboard_running": False},
            {"proxy_running": True, "dashboard_running": True},
        ]
    )
    monkeypatch.setattr(manager, "_check_services", lambda: next(checks))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    result = manager._start_services(restart_gateway=True)

    assert result["gateway_restart"]["restarted"] is True
    assert commands.count(["systemctl", "--user", "restart", "openclaw-gateway"]) == 1


def test_install_configures_gateway_proxy_by_default(monkeypatch):
    manager = ClawVaultManager()
    monkeypatch.setattr(manager, "is_installed", lambda: False)
    monkeypatch.setattr(manager, "_setup_venv", lambda: manager.venv_python)
    install_calls: list[tuple[str, ...]] = []

    def fake_pip_install(*args):
        install_calls.append(args)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(manager, "_pip_install", fake_pip_install)
    monkeypatch.setattr(manager, "get_version", lambda: "0.1.0")
    monkeypatch.setattr(
        manager,
        "initialize_config",
        lambda mode, config: {"success": True, "config_path": "config.yaml"},
    )
    monkeypatch.setattr(
        manager,
        "_start_services",
        lambda restart_gateway=False: {"running": True, "gateway_restart": {"skipped": True}},
    )
    monkeypatch.setattr(manager, "check_health", lambda: {"status": "healthy"})

    called = False

    def fake_integrate():
        nonlocal called
        called = True
        return {"success": True}

    monkeypatch.setattr(manager, "_integrate_openclaw_proxy", fake_integrate)

    result = manager.install()

    assert result["success"] is True
    assert called is True
    assert result["proxy_integration"] == {"success": True}
    assert result["install_source"] == "github_latest"
    assert result["install_spec"] == manager_module.CLAWVAULT_GITHUB_SPEC
    assert install_calls == [(manager_module.CLAWVAULT_GITHUB_SPEC,)]


def test_load_config_template_from_venv_helper(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manager = ClawVaultManager()
    manager.venv_python.parent.mkdir(parents=True)
    manager.venv_python.write_text("#!/usr/bin/env python\n")

    template = """
proxy:
  host: 127.0.0.1
  ssl_verify: true
detection:
  enabled: true
  api_keys: true
guard:
  mode: permissive
monitor: {}
audit: {}
dashboard: {}
cloud: {}
openclaw: {}
file_monitor:
  watch_project_sensitive: true
rules: []
agents: {}
vaults: {}
"""

    def fake_run(command, **kwargs):
        assert "claw_vault.config_template" in command[-1]
        return subprocess.CompletedProcess(command, 0, stdout=template, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    config = manager._load_config_template()

    assert config["file_monitor"]["watch_project_sensitive"] is True
    assert "audit" in config
    assert "openclaw" in config


def test_default_config_fallback_contains_full_sections():
    config = ClawVaultManager()._default_config()

    assert config["file_monitor"]["watch_project_sensitive"] is True
    assert "audit" in config
    assert "openclaw" in config
    assert "cloud" in config
    assert "vaults" in config


def test_initialize_config_writes_full_config(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manager = ClawVaultManager()

    result = manager.initialize_config("quick")

    assert result["success"] is True
    config = yaml.safe_load(Path(result["config_path"]).read_text())
    assert config["file_monitor"]["watch_project_sensitive"] is True
    assert config["proxy"]["ssl_verify"] is False
    assert config["guard"]["mode"] == "interactive"
    assert "audit" in config
    assert "openclaw" in config


def test_unconfigure_proxy_removes_proxy_env_without_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    gateway_dir = tmp_path / ".config/systemd/user"
    gateway_dir.mkdir(parents=True)
    service_file = gateway_dir / "openclaw-gateway.service"
    service_file.write_text(
        "[Service]\n"
        "Environment=HTTP_PROXY=http://127.0.0.1:8765\n"
        "Environment=HTTPS_PROXY=http://127.0.0.1:8765\n"
        "Environment=NO_PROXY=localhost,127.0.0.1\n"
        "Environment=NODE_TLS_REJECT_UNAUTHORIZED=0\n"
        "ExecStart=openclaw gateway\n"
    )

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ClawVaultManager().unconfigure_proxy()

    assert result["changed"] is True
    content = service_file.read_text()
    assert "HTTP_PROXY" not in content
    assert "HTTPS_PROXY" not in content
    assert "NODE_TLS_REJECT_UNAUTHORIZED" not in content
    assert "ExecStart=openclaw gateway" in content
    assert ["systemctl", "--user", "daemon-reload"] in commands
    assert ["systemctl", "--user", "restart", "openclaw-gateway"] not in commands
