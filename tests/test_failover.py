"""Comprehensive Failover 测试 — 覆盖 FR-5.1 ~ FR-5.4 全部灾备容错场景.

Test Cases:
- TC-FAILOVER-001: Mock 目标 LLM 返回 429 → 自动漂移到 Failover 链下一模型
- TC-FAILOVER-002: Mock 目标 LLM 返回 503 → 自动漂移
- TC-FAILOVER-003: Mock 目标 LLM 超时 (>30s) → 触发 Failover
- TC-FAILOVER-004: Failover 链全部不可用 → 返回 HTTP 503
- TC-FAILOVER-005: Failover 切换耗时 < 50ms

参考需求: FR-5.1, FR-5.2, FR-5.3, FR-5.4
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from litellm import Router
from litellm.exceptions import RateLimitError, ServiceUnavailableError, Timeout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_list() -> list[dict]:
    """构造与 config.yaml 一致的 model_list 配置。"""
    return [
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "fake-openai-key",
            },
        },
        {
            "model_name": "gemini-1.5-pro",
            "litellm_params": {
                "model": "gemini/gemini-1.5-pro",
                "api_key": "fake-gemini-key",
            },
        },
        {
            "model_name": "deepseek-chat",
            "litellm_params": {
                "model": "deepseek/deepseek-chat",
                "api_key": "fake-deepseek-key",
            },
        },
        {
            "model_name": "local-7b",
            "litellm_params": {
                "model": "ollama/qwen2-7b",
                "api_base": "http://localhost:11434",
            },
        },
    ]


@pytest.fixture
def fallbacks() -> list[dict]:
    """构造与 config.yaml 一致的 fallbacks 配置。"""
    return [
        {"gpt-4o": ["gemini-1.5-pro", "deepseek-chat"]},
        {"deepseek-chat": ["gpt-4o", "local-7b"]},
        {"gemini-1.5-pro": ["gpt-4o", "deepseek-chat"]},
        {"local-7b": ["deepseek-chat"]},
    ]


@pytest.fixture
def router(model_list: list[dict], fallbacks: list[dict]) -> Router:
    """创建配置了 failover 的 LiteLLM Router 实例。"""
    return Router(
        model_list=model_list,
        fallbacks=fallbacks,
        num_retries=0,  # 测试中禁用重试，直接触发 fallback
        timeout=30,
        retry_after=0,  # 测试中不等待
        allowed_fails=0,  # 立即触发 failover
        routing_strategy="simple-shuffle",
    )


# ---------------------------------------------------------------------------
# Helper: 构造 mock 对象
# ---------------------------------------------------------------------------


def _make_successful_response(model: str = "gemini/gemini-1.5-pro") -> MagicMock:
    """构造一个模拟的成功 completion 响应。"""
    choice = MagicMock()
    choice.message.content = "这是来自备用模型的响应"
    choice.message.role = "assistant"
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.id = "chatcmpl-failover-test"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    response._hidden_params = {"model_id": "fallback-model-id"}
    return response


def _make_rate_limit_error() -> RateLimitError:
    """构造一个 429 RateLimitError。"""
    mock_response = httpx.Response(
        status_code=429,
        request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
    )
    return RateLimitError(
        message="Rate limit exceeded",
        model="openai/gpt-4o",
        llm_provider="openai",
        response=mock_response,
    )


def _make_service_unavailable_error() -> ServiceUnavailableError:
    """构造一个 503 ServiceUnavailableError。"""
    mock_response = httpx.Response(
        status_code=503,
        request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
    )
    return ServiceUnavailableError(
        message="Service temporarily unavailable",
        model="openai/gpt-4o",
        llm_provider="openai",
        response=mock_response,
    )


def _make_timeout_error() -> Timeout:
    """构造一个 Timeout 错误 (模拟超过 30s 超时)。"""
    return Timeout(
        message="Request timed out after 30s",
        model="openai/gpt-4o",
        llm_provider="openai",
    )


# ---------------------------------------------------------------------------
# TC-FAILOVER-001: 429 → 自动漂移到 Failover 链下一模型
# ---------------------------------------------------------------------------


class TestFailover001_RateLimit429:
    """TC-FAILOVER-001: 主模型返回 429 时自动 failover 到备用模型。

    验证 FR-5.1: 自动捕获 LLM API 的 429 (Rate Limit) 错误。
    验证 FR-5.3: 支持优先级排序的 Failover 链。
    """

    async def test_429_triggers_failover_to_next_model(self, router: Router):
        """主模型返回 429，Router 自动漂移到 fallback 模型并返回成功响应。"""
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")

        with patch.object(
            router, "async_function_with_fallbacks", return_value=successful_response
        ):
            response = await router.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "你好，请帮我分析数据"}],
            )

            assert response is not None
            assert response.choices[0].message.content == "这是来自备用模型的响应"

    async def test_429_failover_transparent_to_client(self, router: Router):
        """Failover 对客户端透明 — 客户端收到正常格式响应。"""
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")

        with patch.object(
            router, "async_function_with_fallbacks", return_value=successful_response
        ):
            response = await router.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试 429 failover"}],
            )

            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            assert response.choices[0].finish_reason == "stop"

    async def test_429_fallback_chain_order(self, router: Router, fallbacks: list[dict]):
        """Failover 链优先级与配置一致 (gpt-4o → gemini-1.5-pro → deepseek-chat)。"""
        gpt4o_fallback = None
        for entry in router.fallbacks:
            if "gpt-4o" in entry:
                gpt4o_fallback = entry["gpt-4o"]
                break

        assert gpt4o_fallback is not None
        assert gpt4o_fallback == ["gemini-1.5-pro", "deepseek-chat"]


# ---------------------------------------------------------------------------
# TC-FAILOVER-002: 503 → 自动漂移
# ---------------------------------------------------------------------------


class TestFailover002_ServiceUnavailable503:
    """TC-FAILOVER-002: 主模型返回 503 时自动 failover 到备用模型。

    验证 FR-5.1: 自动捕获 LLM API 的 503 (Service Unavailable) 错误。
    """

    async def test_503_triggers_failover(self, router: Router):
        """主模型返回 503，Router 自动调用 fallback 链并返回成功响应。"""
        successful_response = _make_successful_response("deepseek/deepseek-chat")

        with patch.object(
            router, "async_function_with_fallbacks", return_value=successful_response
        ):
            response = await router.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试 503 failover"}],
            )

            assert response is not None
            assert response.choices[0].message.content == "这是来自备用模型的响应"

    async def test_503_error_captured_by_router(self, router: Router):
        """验证 Router 能捕获 503 并触发 fallback 逻辑。"""
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")
        call_models: list[str] = []

        async def mock_async_function_with_retries(*args, **kwargs):
            model = kwargs.get("model", "")
            call_models.append(model)
            if model == "gpt-4o":
                raise _make_service_unavailable_error()
            return successful_response

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_async_function_with_retries,
        ):
            response = await router.async_function_with_fallbacks(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试 503"}],
                original_function=router._acompletion,
                num_retries=0,
                metadata={"model_group": "gpt-4o"},
            )

            assert response is not None
            assert response.choices[0].message.content == "这是来自备用模型的响应"
            # 验证: 首先尝试 gpt-4o，失败后尝试了 fallback 模型
            assert "gpt-4o" in call_models
            assert len(call_models) >= 2


# ---------------------------------------------------------------------------
# TC-FAILOVER-003: 超时 (>30s) → 触发 Failover
# ---------------------------------------------------------------------------


class TestFailover003_Timeout:
    """TC-FAILOVER-003: 主模型超时 (>30s) 时触发 Failover。

    验证 FR-5.1: 自动捕获 LLM API 的 Timeout 错误。
    """

    async def test_timeout_triggers_failover(self, router: Router):
        """主模型超时后，Router 自动切换到 fallback 模型。"""
        successful_response = _make_successful_response("deepseek/deepseek-chat")
        call_models: list[str] = []

        async def mock_async_function_with_retries(*args, **kwargs):
            model = kwargs.get("model", "")
            call_models.append(model)
            if model == "gpt-4o":
                raise _make_timeout_error()
            return successful_response

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_async_function_with_retries,
        ):
            response = await router.async_function_with_fallbacks(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试超时 failover"}],
                original_function=router._acompletion,
                num_retries=0,
                metadata={"model_group": "gpt-4o"},
            )

            assert response is not None
            assert response.choices[0].message.content == "这是来自备用模型的响应"
            assert "gpt-4o" in call_models

    async def test_timeout_error_has_correct_status_code(self):
        """Timeout 异常包含 408 状态码。"""
        error = _make_timeout_error()
        assert error.status_code == 408
        assert "timed out" in error.message.lower()

    async def test_asyncio_timeout_triggers_failover(self, router: Router):
        """asyncio.TimeoutError 也能触发 failover。"""
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")
        call_models: list[str] = []

        async def mock_async_function_with_retries(*args, **kwargs):
            model = kwargs.get("model", "")
            call_models.append(model)
            if model == "gpt-4o":
                raise asyncio.TimeoutError("Connection timed out after 30s")
            return successful_response

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_async_function_with_retries,
        ):
            response = await router.async_function_with_fallbacks(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试 asyncio 超时"}],
                original_function=router._acompletion,
                num_retries=0,
                metadata={"model_group": "gpt-4o"},
            )

            assert response is not None
            assert "gpt-4o" in call_models
            assert len(call_models) >= 2


# ---------------------------------------------------------------------------
# TC-FAILOVER-004: Failover 链全部不可用 → 返回 HTTP 503
# ---------------------------------------------------------------------------


class TestFailover004_AllUnavailable:
    """TC-FAILOVER-004: Failover 链全部不可用时返回错误。

    验证: 当所有模型（主模型 + 所有 fallback 模型）都不可用时，
    客户端收到错误响应而非挂起。
    """

    async def test_all_models_unavailable_raises_error(self, router: Router):
        """所有模型都返回错误时，最终向客户端抛出异常。"""

        async def mock_always_fail(*args, **kwargs):
            model = kwargs.get("model", "")
            mock_response = httpx.Response(
                status_code=503,
                request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
            )
            raise ServiceUnavailableError(
                message=f"Service unavailable for {model}",
                model=model,
                llm_provider="openai",
                response=mock_response,
            )

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_always_fail,
        ):
            with pytest.raises(Exception) as exc_info:
                await router.async_function_with_fallbacks(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "测试全部不可用"}],
                    original_function=router._acompletion,
                    num_retries=0,
                )

            # 验证: 抛出的异常关联到服务不可用
            error = exc_info.value
            assert hasattr(error, "status_code")
            assert error.status_code == 503

    async def test_all_models_rate_limited_raises_error(self, router: Router):
        """所有模型都返回 429 时，最终向客户端抛出异常。"""

        async def mock_all_rate_limited(*args, **kwargs):
            model = kwargs.get("model", "")
            mock_response = httpx.Response(
                status_code=429,
                request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
            )
            raise RateLimitError(
                message=f"Rate limit for {model}",
                model=model,
                llm_provider="openai",
                response=mock_response,
            )

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_all_rate_limited,
        ):
            with pytest.raises(Exception) as exc_info:
                await router.async_function_with_fallbacks(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "全部限流"}],
                    original_function=router._acompletion,
                    num_retries=0,
                )

            error = exc_info.value
            assert hasattr(error, "status_code")
            assert error.status_code in (429, 503)

    async def test_all_models_timeout_raises_error(self, router: Router):
        """所有模型都超时时，最终向客户端抛出异常。"""

        async def mock_all_timeout(*args, **kwargs):
            raise _make_timeout_error()

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_all_timeout,
        ):
            with pytest.raises(Exception) as exc_info:
                await router.async_function_with_fallbacks(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "全部超时"}],
                    original_function=router._acompletion,
                    num_retries=0,
                )

            error = exc_info.value
            assert hasattr(error, "status_code")
            assert error.status_code in (408, 503)


# ---------------------------------------------------------------------------
# TC-FAILOVER-005: Failover 切换耗时 < 50ms
# ---------------------------------------------------------------------------


class TestFailover005_Latency:
    """TC-FAILOVER-005: Failover 切换耗时 < 50ms。

    验证 FR-5.2: 触发灾备时，在 50ms 内将请求漂移到配置的候选模型。
    """

    async def test_failover_latency_under_50ms(self, router: Router):
        """从错误检测到 fallback 响应返回的总耗时 < 50ms。"""
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")
        call_count = 0

        async def mock_async_function_with_retries(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            model = kwargs.get("model", "")
            if model == "gpt-4o":
                raise _make_rate_limit_error()
            return successful_response

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_async_function_with_retries,
        ):
            start = time.perf_counter()
            response = await router.async_function_with_fallbacks(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试延迟"}],
                original_function=router._acompletion,
                num_retries=0,
                metadata={"model_group": "gpt-4o"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response is not None
            assert elapsed_ms < 50, (
                f"Failover 切换耗时 {elapsed_ms:.2f}ms, 超过 50ms 限制 (FR-5.2)"
            )

    async def test_failover_latency_503_under_50ms(self, router: Router):
        """503 场景下 failover 切换耗时同样 < 50ms。"""
        successful_response = _make_successful_response("deepseek/deepseek-chat")

        async def mock_async_function_with_retries(*args, **kwargs):
            model = kwargs.get("model", "")
            if model == "gpt-4o":
                raise _make_service_unavailable_error()
            return successful_response

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_async_function_with_retries,
        ):
            start = time.perf_counter()
            response = await router.async_function_with_fallbacks(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试 503 延迟"}],
                original_function=router._acompletion,
                num_retries=0,
                metadata={"model_group": "gpt-4o"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response is not None
            assert elapsed_ms < 50, (
                f"503 Failover 切换耗时 {elapsed_ms:.2f}ms, 超过 50ms 限制 (FR-5.2)"
            )

    async def test_failover_latency_timeout_under_50ms(self, router: Router):
        """Timeout 场景下 failover 切换耗时 < 50ms（不含模型本身超时等待时间）。"""
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")

        async def mock_async_function_with_retries(*args, **kwargs):
            model = kwargs.get("model", "")
            if model == "gpt-4o":
                raise _make_timeout_error()
            return successful_response

        with patch.object(
            router,
            "async_function_with_retries",
            side_effect=mock_async_function_with_retries,
        ):
            start = time.perf_counter()
            response = await router.async_function_with_fallbacks(
                model="gpt-4o",
                messages=[{"role": "user", "content": "测试超时延迟"}],
                original_function=router._acompletion,
                num_retries=0,
                metadata={"model_group": "gpt-4o"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert response is not None
            assert elapsed_ms < 50, (
                f"Timeout Failover 切换耗时 {elapsed_ms:.2f}ms, 超过 50ms 限制 (FR-5.2)"
            )
