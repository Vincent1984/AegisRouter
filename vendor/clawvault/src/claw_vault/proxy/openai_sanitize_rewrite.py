"""OpenAI-compatible chat request rewrite for ClawVault sanitize commands."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from claw_vault.detector.engine import DetectionEngine, ScanResult
from claw_vault.sanitizer.replacer import Sanitizer

RewriteAction = Literal["none", "rewrite", "usage", "error"]


@dataclass(frozen=True)
class SanitizeRewriteResult:
    action: RewriteAction
    body: str
    sanitized_text: str = ""
    safe_original_body: str = ""
    scan: ScanResult | None = None
    error: str = ""

    @property
    def should_forward(self) -> bool:
        return self.action in {"none", "rewrite"}

    @property
    def no_restore(self) -> bool:
        return self.action == "rewrite"


_SANITIZE_PATTERNS = (
    re.compile(
        "^(?:\u8bf7)?(?:\u5e2e\u6211)?(?:\u8131\u654f\u4fe1\u606f|\u654f\u611f\u4fe1\u606f\u8131\u654f|\u4fe1\u606f\u8131\u654f|\u8131\u654f)[\uff1a:\\s]+(?P<text>.+)$",
        re.DOTALL,
    ),
    re.compile("^(?:sanitize|redact|mask)[\uff1a:\\s]+(?P<text>.+)$", re.IGNORECASE | re.DOTALL),
)
_SANITIZE_USAGE_TERMS = {
    "\u8131\u654f",
    "\u8131\u654f\u4fe1\u606f",
    "\u654f\u611f\u4fe1\u606f\u8131\u654f",
    "\u4fe1\u606f\u8131\u654f",
    "sanitize",
    "redact",
    "mask",
}
_SANITIZE_QUESTION_RE = re.compile(
    "(?:\u4ec0\u4e48\u662f|\u662f\u4ec0\u4e48\u610f\u601d|"
    "\u600e\u4e48\u914d\u7f6e|what is|explain|how\\s+to\\s+configure|"
    "configure|\u4ecb\u7ecd).*(?:\u8131\u654f|sanitize|redact|mask)",
    re.IGNORECASE,
)
_METADATA_RE = re.compile(
    r"^Sender\s*\(.*?\):\s*```json\s*\{[^}]*\}\s*```\s*(?:\[.*?\]\s*\.{3}\s*)?",
    re.DOTALL,
)
_OPENCLAW_TIMESTAMP_RE = re.compile(r"^\[[^\]\n]*\]\s+(?=@clawvault\b)", re.IGNORECASE)


def rewrite_openai_chat_sanitize_command(body: str) -> SanitizeRewriteResult:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return SanitizeRewriteResult(action="none", body=body)
    if not isinstance(data, dict):
        return SanitizeRewriteResult(action="none", body=body)

    target = _find_latest_user_text(data)
    if target is None:
        return SanitizeRewriteResult(action="none", body=body)
    message_index, part_index, prompt = target
    intent = parse_sanitize_intent(prompt)
    if intent["action"] == "none":
        return SanitizeRewriteResult(action="none", body=body)
    if intent["action"] == "usage":
        return SanitizeRewriteResult(action="usage", body="", safe_original_body=_safe_body(data))

    payload = intent.get("text", "")
    try:
        scan = DetectionEngine().scan_full(payload)
        sanitized = Sanitizer().sanitize(payload, scan.sensitive) if scan.sensitive else payload
    except Exception:
        return SanitizeRewriteResult(action="error", body="", safe_original_body=_safe_body(data))

    rewritten = copy.deepcopy(data)
    _replace_latest_user_text(rewritten, message_index, part_index, sanitized)
    rewritten_body = json.dumps(rewritten, ensure_ascii=False)
    return SanitizeRewriteResult(
        action="rewrite",
        body=rewritten_body,
        sanitized_text=sanitized,
        safe_original_body=rewritten_body,
        scan=scan,
    )


def parse_sanitize_intent(message: str) -> dict[str, str]:
    text = strip_openclaw_metadata(message).strip()
    if not text:
        return {"action": "none"}
    if not text.lower().startswith("@clawvault"):
        return {"action": "none"}
    normalized = text[len("@clawvault") :].strip()
    if not normalized:
        return {"action": "none"}
    if _SANITIZE_QUESTION_RE.search(normalized):
        return {"action": "none"}
    if normalized.lower() in _SANITIZE_USAGE_TERMS:
        return {"action": "usage"}
    for pattern in _SANITIZE_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue
        payload = match.group("text").strip()
        return {"action": "sanitize", "text": payload} if payload else {"action": "usage"}
    return {"action": "none"}


def strip_openclaw_metadata(content: str) -> str:
    stripped = content.strip()
    if _METADATA_RE.search(stripped):
        stripped = _METADATA_RE.sub("", stripped, count=1).strip()
    return _OPENCLAW_TIMESTAMP_RE.sub("", stripped, count=1).strip()


def _find_latest_user_text(data: dict[str, Any]) -> tuple[int, int | None, str] | None:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return message_index, None, strip_openclaw_metadata(content)
        if isinstance(content, list):
            for index in range(len(content) - 1, -1, -1):
                item = content[index]
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    return message_index, index, strip_openclaw_metadata(item["text"])
    return None


def _replace_latest_user_text(
    data: dict[str, Any], message_index: int, part_index: int | None, text: str
) -> None:
    messages = data.get("messages")
    if not isinstance(messages, list) or not 0 <= message_index < len(messages):
        return
    message = messages[message_index]
    if not isinstance(message, dict):
        return
    if part_index is None:
        message["content"] = text
        return
    content = message.get("content")
    if isinstance(content, list) and 0 <= part_index < len(content):
        item = content[part_index]
        if isinstance(item, dict):
            item["text"] = text


def _safe_body(data: dict[str, Any]) -> str:
    safe = copy.deepcopy(data)
    messages = safe.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if (
                isinstance(content, str)
                and strip_openclaw_metadata(content).lower().lstrip().startswith("@clawvault")
            ):
                message["content"] = "[CLAWVAULT_SANITIZE_ORIGINAL_SUPPRESSED]"
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        text = item["text"]
                        if (
                            strip_openclaw_metadata(text)
                            .lower()
                            .lstrip()
                            .startswith("@clawvault")
                        ):
                            item["text"] = "[CLAWVAULT_SANITIZE_ORIGINAL_SUPPRESSED]"
    return json.dumps(safe, ensure_ascii=False)
