"""Unit tests for RequestLoggerCallback._truncate_messages.

Validates Requirement 6.4:
- max_message_length > 0 时，超出长度的内容截断为 max_message_length 字符 + " [truncated]"
- max_message_length <= 0 时不截断
"""

from __future__ import annotations

import pytest

from aegis_router.observability.request_logger import (
    RequestLoggerCallback,
    RequestLoggingConfig,
)


@pytest.fixture
def callback():
    """Create a RequestLoggerCallback with stdout output for testing."""
    config = RequestLoggingConfig(
        enabled=True,
        output="stdout",
        max_message_length=4096,
    )
    return RequestLoggerCallback(config=config)


class TestTruncateMessages:
    """Verify _truncate_messages truncation logic."""

    def test_no_truncation_when_max_length_zero(self, callback):
        """max_message_length=0 means no truncation."""
        messages = [{"role": "user", "content": "a" * 10000}]
        result = callback._truncate_messages(messages, max_length=0)
        assert result == messages

    def test_no_truncation_when_max_length_negative(self, callback):
        """max_message_length < 0 means no truncation."""
        messages = [{"role": "user", "content": "a" * 10000}]
        result = callback._truncate_messages(messages, max_length=-1)
        assert result == messages

    def test_no_truncation_when_content_within_limit(self, callback):
        """Content shorter than max_length is not modified."""
        messages = [{"role": "user", "content": "short message"}]
        result = callback._truncate_messages(messages, max_length=100)
        assert result[0]["content"] == "short message"

    def test_no_truncation_when_content_equals_limit(self, callback):
        """Content exactly at max_length is not truncated."""
        content = "a" * 50
        messages = [{"role": "user", "content": content}]
        result = callback._truncate_messages(messages, max_length=50)
        assert result[0]["content"] == content

    def test_truncation_when_content_exceeds_limit(self, callback):
        """Content exceeding max_length is truncated with suffix."""
        content = "a" * 100
        messages = [{"role": "user", "content": content}]
        result = callback._truncate_messages(messages, max_length=50)
        assert result[0]["content"] == "a" * 50 + " [truncated]"

    def test_truncated_content_length(self, callback):
        """Truncated content is max_length chars + ' [truncated]' suffix."""
        content = "x" * 200
        max_length = 80
        messages = [{"role": "user", "content": content}]
        result = callback._truncate_messages(messages, max_length=max_length)
        expected = "x" * max_length + " [truncated]"
        assert result[0]["content"] == expected
        assert len(result[0]["content"]) == max_length + len(" [truncated]")

    def test_multiple_messages_truncation(self, callback):
        """Multiple messages are each truncated independently."""
        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 30},
            {"role": "user", "content": "c" * 100},
        ]
        result = callback._truncate_messages(messages, max_length=50)
        assert result[0]["content"] == "a" * 50 + " [truncated]"
        assert result[1]["content"] == "b" * 30  # within limit, unchanged
        assert result[2]["content"] == "c" * 50 + " [truncated]"

    def test_non_string_content_not_truncated(self, callback):
        """Non-string content (e.g., list for multimodal) is not truncated."""
        multimodal_content = [{"type": "text", "text": "a" * 10000}]
        messages = [{"role": "user", "content": multimodal_content}]
        result = callback._truncate_messages(messages, max_length=50)
        assert result[0]["content"] == multimodal_content

    def test_missing_content_field(self, callback):
        """Messages without content field get empty string as content."""
        messages = [{"role": "system"}]
        result = callback._truncate_messages(messages, max_length=50)
        assert result[0]["content"] == ""
        assert result[0]["role"] == "system"

    def test_preserves_other_message_fields(self, callback):
        """Other message fields (role, name, etc.) are preserved."""
        messages = [
            {"role": "user", "content": "hello", "name": "Alice"}
        ]
        result = callback._truncate_messages(messages, max_length=50)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"
        assert result[0]["name"] == "Alice"

    def test_does_not_mutate_original_messages(self, callback):
        """Original messages list is not modified."""
        original_content = "a" * 100
        messages = [{"role": "user", "content": original_content}]
        callback._truncate_messages(messages, max_length=50)
        assert messages[0]["content"] == original_content

    def test_empty_messages_list(self, callback):
        """Empty messages list returns empty list."""
        result = callback._truncate_messages([], max_length=50)
        assert result == []
