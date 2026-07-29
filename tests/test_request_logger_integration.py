"""Integration test for RequestLoggerCallback — 日志输出验证 (Task 8.6).

Validates:
- 请求日志文件生成且包含 JSON 格式日志条目
- 日志包含 request_id、model、event_type 字段

This test creates a real RequestLoggerCallback with file output, simulates
request and success events, then verifies the generated log file content.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from aegis_router.observability.request_logger import (
    RequestLoggerCallback,
    RequestLoggingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeResponseObj:
    """Minimal mock of a LiteLLM response object with choices."""

    class _Choice:
        class _Message:
            def __init__(self, content: str):
                self.content = content

        def __init__(self, content: str):
            self.message = self._Message(content)

    def __init__(self, content: str = "This is a test response."):
        self.choices = [self._Choice(content)]


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------


class TestRequestLoggerIntegration:
    """Integration test: verify log file is created with valid JSON entries
    containing required fields (request_id, event_type, model info)."""

    @pytest.fixture
    def log_dir(self, tmp_path):
        """Create a temporary directory for log output."""
        log_path = tmp_path / "logs"
        log_path.mkdir()
        return log_path

    @pytest.fixture
    def log_file(self, log_dir):
        """Return the path to the log file within the temp directory."""
        return str(log_dir / "request_log.jsonl")

    @pytest.fixture
    def callback(self, log_file):
        """Create a RequestLoggerCallback configured to write to a temp file."""
        config = RequestLoggingConfig(
            enabled=True,
            output="file",
            file_path=log_file,
            max_message_length=4096,
            retention_days=7,
        )
        cb = RequestLoggerCallback(config=config)
        yield cb
        # Clean up logger handlers to avoid resource leaks
        cb._logger.handlers.clear()

    def _sample_request_data(self) -> dict:
        """Create realistic sample request data."""
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "model": "gpt-4",
            "metadata": {
                "request_id": "req-integration-001",
                "session_id": "sess-integration-001",
                "target_model": "gpt-4",
                "routing_plugin": "transaction",
                "route_reason": "plan_table_match",
                "route_score": 0.92,
            },
        }

    def _sample_success_kwargs(self) -> dict:
        """Create realistic kwargs for async_log_success_event."""
        return {
            "model": "gpt-4",
            "litellm_params": {
                "metadata": {
                    "request_id": "req-integration-001",
                    "session_id": "sess-integration-001",
                    "target_model": "gpt-4",
                    "routing_plugin": "transaction",
                },
            },
            "standard_logging_object": {
                "prompt_tokens": 25,
                "completion_tokens": 12,
                "total_tokens": 37,
                "response_time_ms": 450.5,
                "model": "gpt-4",
            },
        }

    def test_log_file_created_and_not_empty(self, callback, log_file):
        """Log file is created and contains data after processing a request."""
        data = self._sample_request_data()
        _run_async(callback.async_pre_call_hook({}, None, data, "completion"))

        # Flush handlers to ensure data is written
        for handler in callback._logger.handlers:
            handler.flush()

        assert os.path.exists(log_file), "Log file should be created"
        assert os.path.getsize(log_file) > 0, "Log file should not be empty"

    def test_each_line_is_valid_json(self, callback, log_file):
        """Every line in the log file is a valid JSON object."""
        data = self._sample_request_data()
        _run_async(callback.async_pre_call_hook({}, None, data, "completion"))

        response_obj = _FakeResponseObj("Paris is the capital of France.")
        kwargs = self._sample_success_kwargs()
        _run_async(
            callback.async_log_success_event(kwargs, response_obj, None, None)
        )

        # Flush handlers
        for handler in callback._logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) >= 2, "Should have at least 2 log entries (request + success)"

        for i, line in enumerate(lines):
            try:
                parsed = json.loads(line)
                assert isinstance(parsed, dict), f"Line {i} should be a JSON object"
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i} is not valid JSON: {e}\nContent: {line}")

    def test_entries_have_request_id(self, callback, log_file):
        """Each log entry contains a non-empty request_id field."""
        data = self._sample_request_data()
        _run_async(callback.async_pre_call_hook({}, None, data, "completion"))

        response_obj = _FakeResponseObj("Paris is the capital of France.")
        kwargs = self._sample_success_kwargs()
        _run_async(
            callback.async_log_success_event(kwargs, response_obj, None, None)
        )

        for handler in callback._logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert "request_id" in entry, f"Entry {i} missing 'request_id'"
            assert isinstance(entry["request_id"], str), (
                f"Entry {i} 'request_id' should be a string"
            )
            assert len(entry["request_id"]) > 0, (
                f"Entry {i} 'request_id' should not be empty"
            )

    def test_entries_have_event_type(self, callback, log_file):
        """Each log entry has event_type field with a valid value."""
        data = self._sample_request_data()
        _run_async(callback.async_pre_call_hook({}, None, data, "completion"))

        response_obj = _FakeResponseObj("Paris is the capital of France.")
        kwargs = self._sample_success_kwargs()
        _run_async(
            callback.async_log_success_event(kwargs, response_obj, None, None)
        )

        for handler in callback._logger.handlers:
            handler.flush()

        valid_event_types = {"request", "response_success", "response_failure"}

        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            entry = json.loads(line)
            assert "event_type" in entry, f"Entry {i} missing 'event_type'"
            assert entry["event_type"] in valid_event_types, (
                f"Entry {i} has invalid event_type: {entry['event_type']}"
            )

    def test_request_entry_contains_model_info(self, callback, log_file):
        """Request entries contain model information."""
        data = self._sample_request_data()
        _run_async(callback.async_pre_call_hook({}, None, data, "completion"))

        for handler in callback._logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Find the request entry
        request_entries = []
        for line in lines:
            entry = json.loads(line)
            if entry.get("event_type") == "request":
                request_entries.append(entry)

        assert len(request_entries) >= 1, "Should have at least one request entry"

        for entry in request_entries:
            # Model info is in model_requested or routing_decision.target_model
            has_model_requested = (
                "model_requested" in entry and entry["model_requested"] is not None
            )
            has_target_model = (
                "routing_decision" in entry
                and isinstance(entry["routing_decision"], dict)
                and entry["routing_decision"].get("target_model") is not None
            )
            assert has_model_requested or has_target_model, (
                f"Request entry missing model info. "
                f"model_requested={entry.get('model_requested')}, "
                f"routing_decision={entry.get('routing_decision')}"
            )

    def test_success_entry_contains_model_used(self, callback, log_file):
        """Success response entries contain model_used field."""
        data = self._sample_request_data()
        _run_async(callback.async_pre_call_hook({}, None, data, "completion"))

        response_obj = _FakeResponseObj("Paris is the capital of France.")
        kwargs = self._sample_success_kwargs()
        _run_async(
            callback.async_log_success_event(kwargs, response_obj, None, None)
        )

        for handler in callback._logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        success_entries = []
        for line in lines:
            entry = json.loads(line)
            if entry.get("event_type") == "response_success":
                success_entries.append(entry)

        assert len(success_entries) >= 1, "Should have at least one success entry"

        for entry in success_entries:
            assert "model_used" in entry, "Success entry missing 'model_used'"
            assert entry["model_used"] is not None, "model_used should not be None"

    def test_full_lifecycle_produces_correct_event_sequence(
        self, callback, log_file
    ):
        """A full request → response lifecycle produces entries in correct order."""
        data = self._sample_request_data()
        _run_async(callback.async_pre_call_hook({}, None, data, "completion"))

        response_obj = _FakeResponseObj("Paris is the capital of France.")
        kwargs = self._sample_success_kwargs()
        _run_async(
            callback.async_log_success_event(kwargs, response_obj, None, None)
        )

        for handler in callback._logger.handlers:
            handler.flush()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 2, f"Expected 2 entries, got {len(lines)}"

        entry_0 = json.loads(lines[0])
        entry_1 = json.loads(lines[1])

        assert entry_0["event_type"] == "request"
        assert entry_1["event_type"] == "response_success"

        # Both entries share the same request_id
        assert entry_0["request_id"] == entry_1["request_id"]
        assert entry_0["request_id"] == "req-integration-001"
