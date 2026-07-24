# ruff: noqa: S101

from __future__ import annotations

import json

from claw_vault.proxy.openai_sanitize_rewrite import rewrite_openai_chat_sanitize_command

CN_SANITIZE_INFO = "\u8131\u654f\u4fe1\u606f"
CN_WHAT_IS_SANITIZE = "\u4ec0\u4e48\u662f\u8131\u654f\uff1f"
CN_HOW_CONFIGURE_SANITIZE = "\u600e\u4e48\u914d\u7f6e\u8131\u654f\uff1f"
CN_MY_EMAIL_IS = "\u6211\u7684\u90ae\u7bb1\u662f"


def _body(content: object, *, stream: bool = False) -> str:
    return json.dumps(
        {
            "model": "gpt-4o",
            "stream": stream,
            "messages": [
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "previous message"},
                {"role": "user", "content": content},
            ],
        }
    )


def test_rewrites_english_sanitize_command_without_control_prefix() -> None:
    result = rewrite_openai_chat_sanitize_command(
        _body("@clawvault sanitize email=alice@example.com")
    )

    assert result.action == "rewrite"
    payload = json.loads(result.body)
    latest = payload["messages"][-1]["content"]
    assert latest == "email=[EMAIL_1]"
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "alice@example.com" not in serialized
    assert "@clawvault" not in latest
    assert result.safe_original_body == result.body


def test_rewrites_chinese_sanitize_command() -> None:
    result = rewrite_openai_chat_sanitize_command(
        _body(f"@clawvault {CN_SANITIZE_INFO} {CN_MY_EMAIL_IS} alice@example.com")
    )

    assert result.action == "rewrite"
    payload = json.loads(result.body)
    latest = payload["messages"][-1]["content"]
    assert latest == f"{CN_MY_EMAIL_IS} [EMAIL_1]"
    assert "alice@example.com" not in json.dumps(payload, ensure_ascii=False)


def test_rewrites_openclaw_timestamp_prefixed_sanitize_command() -> None:
    result = rewrite_openai_chat_sanitize_command(
        _body(f"[Wed 2026-06-17 12:55 GMT+8] @clawvault {CN_SANITIZE_INFO} email=alice@example.com")
    )

    assert result.action == "rewrite"
    payload = json.loads(result.body)
    latest = payload["messages"][-1]["content"]
    assert latest == "email=[EMAIL_1]"
    assert "alice@example.com" not in json.dumps(payload, ensure_ascii=False)


def test_aliases_rewrite_to_sanitized_body() -> None:
    for command in ("redact", "mask"):
        result = rewrite_openai_chat_sanitize_command(
            _body(f"@clawvault {command} email=alice@example.com")
        )
        assert result.action == "rewrite"
        assert json.loads(result.body)["messages"][-1]["content"] == "email=[EMAIL_1]"


def test_non_trigger_questions_are_noop() -> None:
    for prompt in (
        f"@clawvault {CN_WHAT_IS_SANITIZE}",
        f"@clawvault {CN_HOW_CONFIGURE_SANITIZE}",
        "@clawvault what is sanitize?",
    ):
        result = rewrite_openai_chat_sanitize_command(_body(prompt))
        assert result.action == "none"
        assert result.body == _body(prompt)


def test_empty_body_returns_usage_with_suppressed_original() -> None:
    result = rewrite_openai_chat_sanitize_command(_body("@clawvault sanitize"))

    assert result.action == "usage"
    assert "alice@example.com" not in result.safe_original_body
    assert "[CLAWVAULT_SANITIZE_ORIGINAL_SUPPRESSED]" in result.safe_original_body


def test_stream_request_rewrites_same_as_non_stream() -> None:
    result = rewrite_openai_chat_sanitize_command(
        _body("@clawvault sanitize email=alice@example.com", stream=True)
    )

    assert result.action == "rewrite"
    payload = json.loads(result.body)
    assert payload["stream"] is True
    assert payload["messages"][-1]["content"] == "email=[EMAIL_1]"


def test_multimodal_text_part_is_rewritten_and_other_parts_preserved() -> None:
    content = [
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        {"type": "text", "text": "@clawvault sanitize email=alice@example.com"},
    ]

    result = rewrite_openai_chat_sanitize_command(_body(content))

    assert result.action == "rewrite"
    rewritten = json.loads(result.body)["messages"][-1]["content"]
    assert rewritten[0] == content[0]
    assert rewritten[1]["text"] == "email=[EMAIL_1]"
    assert "alice@example.com" not in json.dumps(rewritten, ensure_ascii=False)
