from __future__ import annotations

from typer.testing import CliRunner

from claw_vault.cli import app


class _FakeResult:
    success = True
    message = "ok"
    warnings = []
    data = {"sanitized": "token=[API_KEY_1]"}


class _FakeRegistry:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, skill_name, tool_name, **kwargs):
        self.calls.append((skill_name, tool_name, kwargs))
        return _FakeResult()


def test_skill_invoke_stdin_passes_secret_outside_argv(monkeypatch) -> None:
    secret = "token=sk-proj-test-secret"
    registry = _FakeRegistry()
    monkeypatch.setattr("claw_vault.cli._get_registry", lambda: registry)

    result = CliRunner().invoke(
        app,
        ["skill", "invoke", "sanitize-restore", "sanitize_message", "--stdin"],
        input=secret,
    )

    assert result.exit_code == 0
    assert registry.calls == [
        ("sanitize-restore", "sanitize_message", {"text": secret})
    ]
    assert secret not in result.output
    assert "token=[API_KEY_1]" in result.output


def test_skill_invoke_stdin_json_outputs_data_only(monkeypatch) -> None:
    secret = "token=sk-proj-test-secret"
    registry = _FakeRegistry()
    monkeypatch.setattr("claw_vault.cli._get_registry", lambda: registry)

    result = CliRunner().invoke(
        app,
        [
            "skill",
            "invoke",
            "sanitize-restore",
            "sanitize_message",
            "--stdin",
            "--json",
        ],
        input=secret,
    )

    assert result.exit_code == 0
    assert registry.calls == [
        ("sanitize-restore", "sanitize_message", {"text": secret})
    ]
    assert secret not in result.output
    assert '"sanitized": "token=[API_KEY_1]"' in result.output
