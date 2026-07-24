from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "skills" / "tophant-clawvault-operator" / "clawvault_ops.py"
spec = importlib.util.spec_from_file_location("clawvault_ops", MODULE_PATH)
assert spec is not None and spec.loader is not None
clawvault_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clawvault_ops)
ClawVaultOps = clawvault_ops.ClawVaultOps

CN_SANITIZE_INFO = "\u8131\u654f\u4fe1\u606f"
CN_HELP_ME_SANITIZE = "\u5e2e\u6211\u8131\u654f"
CN_WHAT_IS_SANITIZE = "\u4ec0\u4e48\u662f\u8131\u654f\uff1f"
CN_MY_EMAIL_IS = "\u6211\u7684\u90ae\u7bb1\u662f"
CN_COLON = "\uff1a"


def test_parse_chinese_sanitize_intent() -> None:
    ops = ClawVaultOps()

    result = ops.parse_sanitize_intent(f"{CN_SANITIZE_INFO} {CN_MY_EMAIL_IS} alice@example.com")

    assert result == {"action": "sanitize", "text": f"{CN_MY_EMAIL_IS} alice@example.com"}


def test_parse_chinese_colon_sanitize_intent() -> None:
    ops = ClawVaultOps()

    result = ops.parse_sanitize_intent(f"@clawvault {CN_HELP_ME_SANITIZE}{CN_COLON}token=sk-test")

    assert result == {"action": "sanitize", "text": "token=sk-test"}


def test_parse_no_body_returns_usage() -> None:
    ops = ClawVaultOps()

    assert ops.parse_sanitize_intent(CN_SANITIZE_INFO) == {"action": "usage"}


def test_non_sanitize_question_does_not_trigger() -> None:
    ops = ClawVaultOps()

    assert ops.parse_sanitize_intent(CN_WHAT_IS_SANITIZE) == {"action": "none"}


def test_sanitize_uses_stdin_without_secret_in_argv(monkeypatch) -> None:
    ops = ClawVaultOps()
    secret = "token=sk-proj-test-secret"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='{"sanitized":"token=[API_KEY_1]"}', stderr="")

    monkeypatch.setattr(clawvault_ops.subprocess, "run", fake_run)
    monkeypatch.setattr(ops, "_sanitize_python_executable", lambda: "python")

    result = ops.sanitize_text(secret)

    assert result == {"success": True, "sanitized": "token=[API_KEY_1]"}
    argv, kwargs = calls[0]
    assert secret not in argv
    assert kwargs["input"] == secret
    assert kwargs["shell"] is False
    assert "--stdin" in argv
    assert "--json" in argv
    assert "--text" not in argv


def test_sanitize_python_prefers_cli_with_stdin_support(monkeypatch) -> None:
    ops = ClawVaultOps()

    monkeypatch.setattr(clawvault_ops.Path, "exists", lambda self: True)

    def fake_supports(python_executable: str) -> bool:
        return python_executable.endswith("/.venv/bin/python")

    monkeypatch.setattr(ops, "_clawvault_cli_supports_stdin", fake_supports)
    monkeypatch.setattr(ops, "_python_executable", lambda: "/home/user/.clawvault-env/bin/python3")

    assert ops._sanitize_python_executable().endswith("/.venv/bin/python")


def test_handle_no_body_does_not_call_subprocess(monkeypatch) -> None:
    ops = ClawVaultOps()

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess should not be called")

    monkeypatch.setattr(clawvault_ops.subprocess, "run", fail_run)

    result = ops.handle_sanitize_message(CN_SANITIZE_INFO)

    assert result is not None
    assert result["success"] is False
    assert result["error"] == "sanitize_text_required"


def test_non_sanitize_question_returns_none() -> None:
    ops = ClawVaultOps()

    assert ops.handle_sanitize_message(CN_WHAT_IS_SANITIZE) is None


def test_subprocess_failure_does_not_leak_original(monkeypatch) -> None:
    ops = ClawVaultOps()
    secret = "token=sk-proj-test-secret"

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr=f"failed {secret}")

    monkeypatch.setattr(clawvault_ops.subprocess, "run", fake_run)
    monkeypatch.setattr(ops, "_python_executable", lambda: "python")

    result = ops.sanitize_text(secret)

    assert result["success"] is False
    assert secret not in str(result)


def test_subprocess_timeout_does_not_leak_original(monkeypatch) -> None:
    ops = ClawVaultOps()
    secret = "token=sk-proj-test-secret"

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, timeout=20, output=secret, stderr=secret)

    monkeypatch.setattr(clawvault_ops.subprocess, "run", fake_run)
    monkeypatch.setattr(ops, "_python_executable", lambda: "python")

    result = ops.sanitize_text(secret)

    assert result["success"] is False
    assert secret not in str(result)


def test_sanitize_handler_is_local_and_does_not_call_provider(monkeypatch) -> None:
    ops = ClawVaultOps()
    secret = f"{CN_MY_EMAIL_IS} alice@example.com"
    called = {}

    def fake_sanitize(text: str):
        called["text"] = text
        return {"success": True, "sanitized": f"{CN_MY_EMAIL_IS} [EMAIL_1]"}

    monkeypatch.setattr(ops, "sanitize_text", fake_sanitize)

    result = ops.handle_sanitize_message(f"{CN_SANITIZE_INFO} {secret}")

    assert result == {"success": True, "sanitized": f"{CN_MY_EMAIL_IS} [EMAIL_1]"}
    assert called["text"] == secret
