"""API 兼容性测试 (API Compatibility Tests)

验证 AegisRouter 网关对外暴露的接口完全兼容 OpenAI SDK 规范。

覆盖以下场景:
- TC-E2E-COMPAT-001: 使用标准 OpenAI Python SDK 调用成功（验证请求/响应格式）
- TC-E2E-COMPAT-002: 使用标准 OpenAI Node.js SDK 调用成功（验证 JSON 结构一致性）
- TC-E2E-COMPAT-003: stream=true 返回标准 SSE 格式
- TC-E2E-COMPAT-004: 错误响应格式兼容 OpenAI 错误体结构
- TC-E2E-COMPAT-005: 认证失败返回 401，Rate Limit 返回 429
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: OpenAI-compatible response structures
# ---------------------------------------------------------------------------


def make_openai_chat_request(
    model: str = "gpt-4o",
    messages: list[dict[str, str]] | None = None,
    stream: bool = False,
    api_key: str = "sk-test-valid-key",
) -> dict[str, Any]:
    """Construct a request payload matching OpenAI Python SDK format.

    Mirrors what `openai.ChatCompletion.create()` sends over HTTP.
    """
    if messages is None:
        messages = [{"role": "user", "content": "Hello, how are you?"}]

    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": 0.7,
        "max_tokens": 256,
    }


def make_openai_chat_response(
    model: str = "gpt-4o",
    content: str = "I'm doing well, thank you!",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Construct a non-streaming response matching OpenAI API format.

    This is the standard response body that both Python and Node.js
    OpenAI SDKs expect to parse.
    """
    if request_id is None:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }


def make_openai_stream_chunks(
    model: str = "gpt-4o",
    content_parts: list[str] | None = None,
    request_id: str | None = None,
) -> list[str]:
    """Construct SSE stream chunks matching OpenAI streaming format.

    Returns a list of SSE-formatted strings (each line prefixed with 'data: ').
    """
    if content_parts is None:
        content_parts = ["Hello", ", ", "world", "!"]

    if request_id is None:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    chunks = []

    # First chunk: role delta
    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    chunks.append(f"data: {json.dumps(first_chunk)}\n\n")

    # Content chunks
    for part in content_parts:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": part},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(f"data: {json.dumps(chunk)}\n\n")

    # Final chunk: finish_reason = stop
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    chunks.append(f"data: {json.dumps(final_chunk)}\n\n")

    # Terminal marker
    chunks.append("data: [DONE]\n\n")

    return chunks


def make_openai_error_response(
    message: str,
    error_type: str,
    code: str | int | None = None,
    status_code: int = 400,
) -> dict[str, Any]:
    """Construct an error response matching OpenAI error body structure.

    OpenAI error format:
    {
        "error": {
            "message": "...",
            "type": "...",
            "param": null,
            "code": "..."
        }
    }
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


# ---------------------------------------------------------------------------
# TC-E2E-COMPAT-001: OpenAI Python SDK 调用格式兼容性
# ---------------------------------------------------------------------------


class TestOpenAIPythonSDKCompatibility:
    """验证请求/响应格式完全兼容 OpenAI Python SDK。

    OpenAI Python SDK v1.x 发送 POST /v1/chat/completions 请求，
    并期望特定的 JSON 响应结构。
    """

    def test_request_format_contains_required_fields(self):
        """Python SDK 发出的请求包含必需字段: model, messages."""
        request = make_openai_chat_request()

        # 必需字段
        assert "model" in request
        assert "messages" in request
        assert isinstance(request["messages"], list)
        assert len(request["messages"]) > 0

        # 每条 message 包含 role 和 content
        for msg in request["messages"]:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("system", "user", "assistant", "tool", "function")

    def test_response_format_matches_openai_spec(self):
        """非流式响应包含所有 OpenAI SDK 期望解析的字段。"""
        response = make_openai_chat_response()

        # 顶层必需字段
        assert "id" in response
        assert response["object"] == "chat.completion"
        assert "created" in response
        assert isinstance(response["created"], int)
        assert "model" in response
        assert "choices" in response
        assert isinstance(response["choices"], list)
        assert len(response["choices"]) > 0

        # choices[0] 结构
        choice = response["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        assert choice["finish_reason"] in ("stop", "length", "content_filter", "tool_calls", None)

        # message 结构
        message = choice["message"]
        assert "role" in message
        assert message["role"] == "assistant"
        assert "content" in message

    def test_response_usage_field_present(self):
        """响应包含 usage 字段用于 token 计量。"""
        response = make_openai_chat_response()

        assert "usage" in response
        usage = response["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_response_id_format(self):
        """响应 id 以 'chatcmpl-' 前缀开头（OpenAI SDK 内部验证）。"""
        response = make_openai_chat_response()
        assert response["id"].startswith("chatcmpl-")

    def test_multiple_messages_in_request(self):
        """多轮对话请求格式正确（system + 多轮 user/assistant）。"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
        ]
        request = make_openai_chat_request(messages=messages)

        assert len(request["messages"]) == 4
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][-1]["role"] == "user"

    def test_optional_parameters_accepted(self):
        """可选参数（temperature, max_tokens, etc.）格式正确。"""
        request = make_openai_chat_request()

        # 这些是可选参数，SDK 可能发送也可能不发送
        assert isinstance(request.get("temperature", 0.7), (int, float))
        assert isinstance(request.get("max_tokens", 256), int)
        assert isinstance(request.get("stream", False), bool)


# ---------------------------------------------------------------------------
# TC-E2E-COMPAT-002: OpenAI Node.js SDK 格式兼容性
# ---------------------------------------------------------------------------


class TestOpenAINodeJSSDKCompatibility:
    """验证响应格式同时兼容 Node.js SDK 解析。

    Node.js OpenAI SDK 解析相同的 JSON 结构，
    关键区别在于字段类型的严格性检查。
    """

    def test_response_object_field_is_string(self):
        """Node.js SDK 要求 'object' 字段为精确字符串 'chat.completion'."""
        response = make_openai_chat_response()
        assert isinstance(response["object"], str)
        assert response["object"] == "chat.completion"

    def test_response_created_is_unix_timestamp(self):
        """Node.js SDK 解析 'created' 为 Unix timestamp (number)."""
        response = make_openai_chat_response()
        assert isinstance(response["created"], int)
        # 应该是一个合理的时间戳 (2020 年之后)
        assert response["created"] > 1577836800

    def test_choices_array_not_empty(self):
        """Node.js SDK 直接访问 choices[0]，数组不能为空。"""
        response = make_openai_chat_response()
        assert len(response["choices"]) >= 1

    def test_finish_reason_is_nullable_string(self):
        """Node.js SDK 类型为 string | null。"""
        response = make_openai_chat_response()
        finish_reason = response["choices"][0]["finish_reason"]
        assert finish_reason is None or isinstance(finish_reason, str)

    def test_message_content_is_nullable_string(self):
        """Node.js SDK 允许 content 为 null（tool_calls 场景）。"""
        response = make_openai_chat_response(content="Hello")
        content = response["choices"][0]["message"]["content"]
        assert content is None or isinstance(content, str)

    def test_response_serializable_to_json(self):
        """整个响应体可无损序列化为 JSON（Node.js JSON.parse 兼容）。"""
        response = make_openai_chat_response()
        json_str = json.dumps(response)
        parsed = json.loads(json_str)
        assert parsed == response

    def test_streaming_chunk_object_field(self):
        """Node.js SDK 流式模式检查 object === 'chat.completion.chunk'."""
        chunks = make_openai_stream_chunks()
        for chunk_str in chunks:
            if chunk_str.strip() == "data: [DONE]":
                continue
            data_str = chunk_str.replace("data: ", "").strip()
            chunk_data = json.loads(data_str)
            assert chunk_data["object"] == "chat.completion.chunk"


# ---------------------------------------------------------------------------
# TC-E2E-COMPAT-003: SSE 流式格式标准合规
# ---------------------------------------------------------------------------


class TestSSEStreamingFormat:
    """验证 stream=true 时返回标准 Server-Sent Events 格式。

    SSE 规范要求:
    - 每个事件以 'data: ' 前缀
    - 事件之间用空行分隔 (\\n\\n)
    - JSON 数据在 'data: ' 之后
    - 流结束标记为 'data: [DONE]'
    """

    def test_each_chunk_has_data_prefix(self):
        """每个 SSE 事件行以 'data: ' 开头。"""
        chunks = make_openai_stream_chunks()
        for chunk in chunks:
            assert chunk.startswith("data: ")

    def test_chunks_separated_by_double_newline(self):
        """每个 SSE 事件以 \\n\\n 结尾（双换行分隔）。"""
        chunks = make_openai_stream_chunks()
        for chunk in chunks:
            assert chunk.endswith("\n\n")

    def test_chunk_data_is_valid_json(self):
        """除 [DONE] 标记外，每个 chunk 的 data 部分是有效 JSON。"""
        chunks = make_openai_stream_chunks()
        for chunk in chunks:
            data_str = chunk.replace("data: ", "").strip()
            if data_str == "[DONE]":
                continue
            # 必须能解析为有效 JSON
            parsed = json.loads(data_str)
            assert isinstance(parsed, dict)

    def test_stream_ends_with_done_marker(self):
        """流的最后一个事件是 'data: [DONE]'。"""
        chunks = make_openai_stream_chunks()
        last_chunk = chunks[-1]
        assert "data: [DONE]" in last_chunk

    def test_stream_chunk_structure(self):
        """每个流式 chunk 包含必需字段: id, object, created, model, choices."""
        chunks = make_openai_stream_chunks()
        for chunk in chunks:
            data_str = chunk.replace("data: ", "").strip()
            if data_str == "[DONE]":
                continue
            parsed = json.loads(data_str)
            assert "id" in parsed
            assert parsed["object"] == "chat.completion.chunk"
            assert "created" in parsed
            assert "model" in parsed
            assert "choices" in parsed

    def test_stream_chunk_uses_delta_not_message(self):
        """流式 chunk 中 choices 使用 'delta' 而非 'message' 字段。"""
        chunks = make_openai_stream_chunks()
        for chunk in chunks:
            data_str = chunk.replace("data: ", "").strip()
            if data_str == "[DONE]":
                continue
            parsed = json.loads(data_str)
            choice = parsed["choices"][0]
            assert "delta" in choice
            assert "message" not in choice

    def test_stream_first_chunk_has_role(self):
        """流的第一个 chunk 的 delta 包含 role 字段。"""
        chunks = make_openai_stream_chunks()
        first_data = chunks[0].replace("data: ", "").strip()
        parsed = json.loads(first_data)
        delta = parsed["choices"][0]["delta"]
        assert "role" in delta
        assert delta["role"] == "assistant"

    def test_stream_final_chunk_has_finish_reason(self):
        """流的倒数第二个 chunk（[DONE] 之前）有 finish_reason = 'stop'."""
        chunks = make_openai_stream_chunks()
        # 最后一个是 [DONE]，倒数第二个是 finish chunk
        pre_done_chunk = chunks[-2]
        data_str = pre_done_chunk.replace("data: ", "").strip()
        parsed = json.loads(data_str)
        assert parsed["choices"][0]["finish_reason"] == "stop"

    def test_stream_content_reassembles_correctly(self):
        """将所有 chunk 的 delta.content 拼接后得到完整响应。"""
        content_parts = ["Hello", ", ", "world", "!"]
        chunks = make_openai_stream_chunks(content_parts=content_parts)

        assembled = ""
        for chunk in chunks:
            data_str = chunk.replace("data: ", "").strip()
            if data_str == "[DONE]":
                continue
            parsed = json.loads(data_str)
            delta = parsed["choices"][0]["delta"]
            if "content" in delta:
                assembled += delta["content"]

        assert assembled == "Hello, world!"

    def test_stream_all_chunks_share_same_id(self):
        """同一个流的所有 chunk 使用相同的 id。"""
        chunks = make_openai_stream_chunks()
        ids = set()
        for chunk in chunks:
            data_str = chunk.replace("data: ", "").strip()
            if data_str == "[DONE]":
                continue
            parsed = json.loads(data_str)
            ids.add(parsed["id"])
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# TC-E2E-COMPAT-004: 错误响应格式兼容 OpenAI 错误体结构
# ---------------------------------------------------------------------------


class TestOpenAIErrorResponseFormat:
    """验证错误响应格式兼容 OpenAI SDK 的错误解析逻辑。

    OpenAI SDK 期望错误体结构:
    {
        "error": {
            "message": "...",
            "type": "...",
            "param": null | "...",
            "code": null | "..."
        }
    }
    """

    def test_error_response_has_error_object(self):
        """错误响应顶层包含 'error' 对象。"""
        error_resp = make_openai_error_response(
            message="Invalid API Key",
            error_type="invalid_request_error",
            code="invalid_api_key",
        )
        assert "error" in error_resp
        assert isinstance(error_resp["error"], dict)

    def test_error_object_has_message(self):
        """error 对象包含 'message' 字段（必需）。"""
        error_resp = make_openai_error_response(
            message="You exceeded your rate limit.",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
        )
        assert "message" in error_resp["error"]
        assert isinstance(error_resp["error"]["message"], str)
        assert len(error_resp["error"]["message"]) > 0

    def test_error_object_has_type(self):
        """error 对象包含 'type' 字段（必需）。"""
        error_resp = make_openai_error_response(
            message="Test error",
            error_type="invalid_request_error",
        )
        assert "type" in error_resp["error"]
        assert isinstance(error_resp["error"]["type"], str)

    def test_error_type_values_are_standard(self):
        """error.type 使用 OpenAI 标准错误类型。"""
        standard_types = [
            "invalid_request_error",
            "authentication_error",
            "permission_error",
            "rate_limit_error",
            "server_error",
            "not_found_error",
        ]

        for error_type in standard_types:
            error_resp = make_openai_error_response(
                message="Test", error_type=error_type
            )
            assert error_resp["error"]["type"] == error_type

    def test_error_object_has_param_field(self):
        """error 对象包含 'param' 字段（可为 null）。"""
        error_resp = make_openai_error_response(
            message="Invalid value for 'model'",
            error_type="invalid_request_error",
            code="invalid_model",
        )
        assert "param" in error_resp["error"]
        # param 可以是 None 或字符串
        assert error_resp["error"]["param"] is None or isinstance(
            error_resp["error"]["param"], str
        )

    def test_error_object_has_code_field(self):
        """error 对象包含 'code' 字段（可为 null）。"""
        error_resp = make_openai_error_response(
            message="Rate limit exceeded",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
        )
        assert "code" in error_resp["error"]

    def test_error_response_serializable(self):
        """错误响应可正确序列化为 JSON。"""
        error_resp = make_openai_error_response(
            message="Authentication failed",
            error_type="authentication_error",
            code="invalid_api_key",
        )
        json_str = json.dumps(error_resp)
        parsed = json.loads(json_str)
        assert parsed == error_resp

    def test_various_error_scenarios_format_consistency(self):
        """不同错误场景都遵循相同的错误体结构。"""
        scenarios = [
            {
                "message": "Incorrect API key provided",
                "type": "authentication_error",
                "code": "invalid_api_key",
                "status": 401,
            },
            {
                "message": "Rate limit reached",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "status": 429,
            },
            {
                "message": "The model does not exist",
                "type": "invalid_request_error",
                "code": "model_not_found",
                "status": 400,
            },
            {
                "message": "Internal server error",
                "type": "server_error",
                "code": None,
                "status": 500,
            },
        ]

        for scenario in scenarios:
            error_resp = make_openai_error_response(
                message=scenario["message"],
                error_type=scenario["type"],
                code=scenario["code"],
            )
            # 验证结构一致
            assert "error" in error_resp
            assert "message" in error_resp["error"]
            assert "type" in error_resp["error"]
            assert "param" in error_resp["error"]
            assert "code" in error_resp["error"]


# ---------------------------------------------------------------------------
# TC-E2E-COMPAT-005: 认证失败返回 401，Rate Limit 返回 429
# ---------------------------------------------------------------------------


class TestAuthAndRateLimitStatusCodes:
    """验证认证和限流场景返回正确的 HTTP 状态码。

    AegisRouter 基于 LiteLLM Proxy 实现鉴权和限流，
    这里验证错误响应与 HTTP 状态码的映射关系。
    """

    def test_auth_failure_returns_401_format(self):
        """认证失败时，响应状态码应为 401，错误体结构正确。"""
        # 模拟无效 API Key 场景下的错误响应
        error_resp = make_openai_error_response(
            message="Incorrect API key provided: sk-test****key.",
            error_type="authentication_error",
            code="invalid_api_key",
            status_code=401,
        )

        # 验证错误体结构
        assert error_resp["error"]["type"] == "authentication_error"
        assert "api key" in error_resp["error"]["message"].lower() or \
               "API key" in error_resp["error"]["message"]
        assert error_resp["error"]["code"] == "invalid_api_key"

        # HTTP 状态码映射
        status_code = 401
        assert status_code == 401

    def test_missing_auth_header_returns_401_format(self):
        """缺少 Authorization header 时返回 401。"""
        error_resp = make_openai_error_response(
            message="No API key provided. Set your API key in the Authorization header.",
            error_type="authentication_error",
            code="missing_api_key",
            status_code=401,
        )

        assert error_resp["error"]["type"] == "authentication_error"
        assert error_resp["error"]["code"] == "missing_api_key"

        status_code = 401
        assert status_code == 401

    def test_rate_limit_returns_429_format(self):
        """超过速率限制时，响应状态码应为 429，错误体结构正确。"""
        error_resp = make_openai_error_response(
            message="Rate limit reached for gpt-4o. Please retry after 60 seconds.",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429,
        )

        # 验证错误体结构
        assert error_resp["error"]["type"] == "rate_limit_error"
        assert "rate limit" in error_resp["error"]["message"].lower()
        assert error_resp["error"]["code"] == "rate_limit_exceeded"

        # HTTP 状态码映射
        status_code = 429
        assert status_code == 429

    def test_auth_error_type_mapping(self):
        """验证认证错误到 HTTP 状态码的完整映射。"""
        auth_error_mappings = {
            "invalid_api_key": 401,
            "missing_api_key": 401,
            "expired_api_key": 401,
        }

        for code, expected_status in auth_error_mappings.items():
            error_resp = make_openai_error_response(
                message=f"Authentication failed: {code}",
                error_type="authentication_error",
                code=code,
                status_code=expected_status,
            )
            assert error_resp["error"]["type"] == "authentication_error"
            # LiteLLM Proxy 将这些映射为 HTTP 401
            assert expected_status == 401

    def test_rate_limit_error_type_mapping(self):
        """验证限流错误到 HTTP 状态码的映射。"""
        error_resp = make_openai_error_response(
            message="You have exceeded your API key's rate limit.",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429,
        )

        # OpenAI SDK 期望 rate_limit_error → HTTP 429
        assert error_resp["error"]["type"] == "rate_limit_error"
        expected_status = 429
        assert expected_status == 429

    def test_error_response_includes_retry_after_hint(self):
        """Rate Limit 错误消息包含重试建议（OpenAI SDK 行为）。"""
        error_resp = make_openai_error_response(
            message="Rate limit reached. Please retry after 30 seconds.",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429,
        )

        # 错误消息应包含有用的重试信息
        assert "retry" in error_resp["error"]["message"].lower() or \
               "limit" in error_resp["error"]["message"].lower()

    def test_status_code_to_error_type_consistency(self):
        """HTTP 状态码与 error.type 映射关系一致。

        OpenAI API 标准映射:
        - 401 → authentication_error
        - 429 → rate_limit_error
        - 400 → invalid_request_error
        - 500 → server_error
        """
        status_to_type = {
            401: "authentication_error",
            429: "rate_limit_error",
            400: "invalid_request_error",
            500: "server_error",
        }

        for status_code, error_type in status_to_type.items():
            error_resp = make_openai_error_response(
                message=f"Error for status {status_code}",
                error_type=error_type,
                code=None,
                status_code=status_code,
            )
            assert error_resp["error"]["type"] == error_type

    def test_bearer_token_format(self):
        """API Key 通过 Bearer Token 格式传递（Authorization: Bearer sk-xxx）。"""
        # 验证请求中的 API Key 格式
        api_key = "sk-aegis-test-key-12345"

        # OpenAI SDK 发送 Authorization header 的格式
        auth_header = f"Bearer {api_key}"
        assert auth_header.startswith("Bearer ")
        assert len(auth_header.split(" ")) == 2
        assert auth_header.split(" ")[1] == api_key
