"""OpenAI-compatible provider adapter for ClawVault."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urljoin

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from claw_vault.audit.models import AuditRecord
from claw_vault.config import ProviderAdapterConfig
from claw_vault.detector.engine import DetectionEngine, ScanResult
from claw_vault.guard.action import Action, ActionResult
from claw_vault.guard.rule_engine import RuleEngine
from claw_vault.proxy.openai_sanitize_rewrite import (
    rewrite_openai_chat_sanitize_command,
    strip_openclaw_metadata,
)
from claw_vault.sanitizer import Sanitizer

logger = structlog.get_logger()
router = APIRouter(tags=["provider-adapter"])
_config = ProviderAdapterConfig()
_engine = DetectionEngine()
_fallback_rules = RuleEngine()
_sanitizer = Sanitizer()

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def configure_provider_adapter(config: ProviderAdapterConfig | None = None) -> None:
    global _config
    _config = config or ProviderAdapterConfig()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    if not _config.enabled:
        return _adapter_error("provider_adapter_disabled", status_code=403)

    raw_body = await request.body()
    body = raw_body.decode("utf-8", errors="replace")
    rewrite = rewrite_openai_chat_sanitize_command(body)
    if rewrite.action == "usage":
        return _adapter_error("Usage: @clawvault sanitize <text>", status_code=400)
    if rewrite.action == "error":
        return _adapter_error("ClawVault could not sanitize this text locally.", status_code=502)

    forward_body = rewrite.body
    scan_text = _extract_user_content(forward_body)
    agent_config = _get_agent_config()
    if agent_config.get("enabled", True):
        scan = _engine.scan_full(scan_text, detection_config=agent_config.get("detection"))
        action_result = _get_rule_engine().evaluate(
            scan,
            guard_mode=agent_config.get("guard_mode"),
            auto_sanitize=agent_config.get("auto_sanitize"),
        )
        if rewrite.action == "rewrite":
            _record_adapter_event("sanitize", scan, forward_body, scan_text)
        elif action_result.action == Action.BLOCK:
            safe_body = _mask_body_with_scan(forward_body, scan)
            safe_text = _mask_text_with_scan(scan_text, scan)
            _record_adapter_event("block", scan, safe_body, safe_text)
            block_message = _format_block_message(scan, action_result)
            if _is_stream_request(forward_body):
                return _chat_completion_sse_response(forward_body, block_message)
            return _chat_completion_response(forward_body, block_message)
        elif action_result.action == Action.ASK_USER:
            safe_body = _mask_body_with_scan(forward_body, scan)
            safe_text = _mask_text_with_scan(scan_text, scan)
            _record_adapter_event("ask_user", scan, safe_body, safe_text)
            warning_message = _format_block_message(scan, action_result)
            if _is_stream_request(forward_body):
                return _chat_completion_sse_response(forward_body, warning_message)
            return _chat_completion_response(forward_body, warning_message)
        elif action_result.action == Action.SANITIZE and scan.sensitive:
            forward_body = _sanitizer.sanitize_by_value(forward_body, scan.sensitive)
            safe_text = _mask_text_with_scan(scan_text, scan)
            _record_adapter_event("sanitize", scan, forward_body, safe_text)
        else:
            _record_adapter_event(action_result.action.value, scan, forward_body, scan_text)

    target_url = urljoin(_config.upstream_base_url.rstrip("/") + "/", "chat/completions")
    headers = _forward_headers(request.headers)
    timeout = httpx.Timeout(_config.request_timeout_seconds)

    if _is_stream_request(forward_body):
        return StreamingResponse(
            _stream_upstream(target_url, headers, forward_body, timeout),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream = await client.post(target_url, content=forward_body, headers=headers)
    except httpx.HTTPError:
        logger.warning("provider_adapter_upstream_failed")
        return _adapter_error(
            "ClawVault provider adapter could not reach upstream.", status_code=502
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type", "application/json"),
    )


async def _stream_upstream(
    target_url: str, headers: dict[str, str], body: str, timeout: httpx.Timeout
) -> AsyncIterator[bytes]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", target_url, content=body, headers=headers) as upstream:
                async for chunk in upstream.aiter_bytes():
                    if chunk:
                        yield chunk
    except httpx.HTTPError:
        logger.warning("provider_adapter_stream_upstream_failed")
        error_chunk = (
            b'data: {"error":{"message":"ClawVault provider adapter could not reach '
            b'upstream."}}\n\n'
        )
        yield error_chunk
        yield b"data: [DONE]\n\n"


def _adapter_error(message: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "claw_vault_provider_adapter"}},
    )


def _chat_completion_response(request_body: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "id": f"clawvault-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": _extract_model(request_body),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    )


def _chat_completion_sse_response(request_body: str, message: str) -> Response:
    response_id = f"clawvault-{uuid.uuid4().hex[:8]}"
    model = _extract_model(request_body)
    first_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": message}, "finish_reason": None}],
    }
    stop_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    sse_body = (
        f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"
        f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    )
    return Response(
        content=sse_body,
        status_code=200,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def _forward_headers(headers: Any) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in _HOP_BY_HOP_HEADERS:
            continue
        forwarded[key] = value
    forwarded.setdefault("Content-Type", "application/json")
    return forwarded


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    returned: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        returned[key] = value
    return returned


def _is_stream_request(body: str) -> bool:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, dict) and data.get("stream") is True


def _extract_model(body: str) -> str:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return "clawvault"
    if isinstance(data, dict) and isinstance(data.get("model"), str):
        return data["model"]
    return "clawvault"


def _extract_user_content(body: str) -> str:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body

    if not isinstance(data, dict):
        return body

    messages = data.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return strip_openclaw_metadata(content)
            if isinstance(content, list):
                parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if parts:
                    return "\n".join(parts)

    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt

    return body


def _format_block_message(scan: ScanResult, action_result: ActionResult) -> str:
    details = _format_detection_details(scan, action_result)
    if details:
        return f"[ClawVault] {action_result.reason}\n\n{details}"
    return f"[ClawVault] {action_result.reason}"


def _format_detection_details(scan: ScanResult, action_result: ActionResult) -> str:
    lines: list[str] = []
    if scan.sensitive:
        lines.append("Sensitive data detected:")
        for item in scan.sensitive:
            lines.append(f"  - {item.description}: {item.masked_value}")
    if scan.commands:
        lines.append("Dangerous commands detected:")
        for item in scan.commands:
            lines.append(f"  - {item.reason}: {item.command[:50]}")
    if scan.injections:
        lines.append("Injection attacks detected:")
        for item in scan.injections:
            lines.append(f"  - {item.description}")
    joined = "\n".join(lines)
    for detail in action_result.details:
        safe_detail = detail
        for item in scan.sensitive:
            if item.value:
                safe_detail = safe_detail.replace(item.value, item.masked_value)
        if safe_detail not in joined:
            lines.append(f"  - {safe_detail}")
    return "\n".join(lines)


def _mask_text_with_scan(text: str, scan: ScanResult) -> str:
    masked = text
    for item in sorted(scan.sensitive, key=lambda result: result.start, reverse=True):
        masked = masked[: item.start] + item.masked_value + masked[item.end :]
    return masked


def _mask_body_with_scan(body: str, scan: ScanResult) -> str:
    masked = body
    for item in scan.sensitive:
        if item.value:
            masked = masked.replace(item.value, item.masked_value)
    return masked


def _get_agent_config() -> dict[str, Any]:
    try:
        from claw_vault.dashboard.api import get_agent_config

        return get_agent_config(None)
    except Exception:
        return {
            "enabled": True,
            "guard_mode": "permissive",
            "detection": None,
            "auto_sanitize": True,
        }


def _get_rule_engine() -> RuleEngine:
    try:
        from claw_vault.dashboard import api as dashboard_api

        rule_engine = getattr(dashboard_api, "_rule_engine", None)
        if isinstance(rule_engine, RuleEngine):
            return rule_engine
    except Exception:
        logger.debug("provider_adapter_rule_engine_lookup_failed")
    return _fallback_rules


def _record_adapter_event(
    action: str,
    scan: ScanResult,
    request_body: str,
    user_content: str | None,
) -> None:
    try:
        from claw_vault.dashboard.api import push_proxy_event

        record = AuditRecord(
            agent_id="provider-adapter",
            agent_name="Provider Adapter",
            direction="request",
            api_endpoint="/v1/chat/completions",
            method="POST",
            risk_level=scan.threat_level.value,
            risk_score=scan.max_risk_score,
            action_taken=action,
            detections=[
                *[f"sensitive:{item.pattern_type}" for item in scan.sensitive],
                *[f"command:{item.command[:30]}" for item in scan.commands],
                *[f"injection:{item.injection_type}" for item in scan.injections],
            ],
            user_content=user_content,
        )
        push_proxy_event(record, scan, request_body)
    except Exception:
        logger.debug("provider_adapter_dashboard_event_failed")
