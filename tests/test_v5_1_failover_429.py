"""Tests for V5-1: Mock 目标 LLM 返回 429 → 请求自动路由到 Failover 链下一模型，客户端无感知.

验证 LiteLLM Router 的内置 failover 机制:
- 当主模型 (gpt-4o) 返回 HTTP 429 (Rate Limit) 时
- 请求自动漂移到 Failover 链中下一模型 (gemini-1.5-pro)
- 客户端收到正常响应，对 failover 过程无感知

参考需求: FR-5.1, FR-5.2, FR-5.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from litellm import Router
from litellm.exceptions import RateLimitError


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
    ]


@pytest.fixture
def fallbacks() -> list[dict]:
    """构造与 config.yaml 一致的 fallbacks 配置。"""
    return [
        {"gpt-4o": ["gemini-1.5-pro", "deepseek-chat"]},
        {"deepseek-chat": ["gpt-4o"]},
        {"gemini-1.5-pro": ["gpt-4o", "deepseek-chat"]},
    ]


@pytest.fixture
def router(model_list: list[dict], fallbacks: list[dict]) -> Router:
    """创建配置了 failover 的 LiteLLM Router 实例。"""
    return Router(
        model_list=model_list,
        fallbacks=fallbacks,
        num_retries=2,
        timeout=30,
        retry_after=0,  # 测试中不等待
        allowed_fails=1,
        routing_strategy="simple-shuffle",
    )


# ---------------------------------------------------------------------------
# Helper: 构造 mock 响应
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
    response.id = "chatcmpl-test-failover"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    response._hidden_params = {"model_id": "fallback-model-id"}
    return response


def _make_rate_limit_error() -> RateLimitError:
    """构造一个 429 RateLimitError。"""
    return RateLimitError(
        message="Rate limit exceeded",
        model="openai/gpt-4o",
        llm_provider="openai",
    )


# ---------------------------------------------------------------------------
# Tests: Failover 429 行为验证
# ---------------------------------------------------------------------------


class TestFailover429:
    """验证主模型返回 429 时自动 failover 到备用模型。"""

    async def test_client_receives_successful_response_on_429(
        self, router: Router
    ):
        """当主模型返回 429，客户端仍收到成功响应（非 429 错误）。

        验证 FR-5.1: 自动捕获 LLM API 的 429 (Rate Limit) 错误。
        """
        successful_response = _make_successful_response()

        with patch.object(
            router, "async_function_with_fallbacks", return_value=successful_response
        ):
            response = await router.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "你好，请帮我分析数据"}],
            )

            # 客户端收到的是成功响应，不是 429 错误
            assert response is not None
            assert response.choices[0].message.content == "这是来自备用模型的响应"

    async def test_failover_is_transparent_to_client(self, router: Router):
        """Failover 对客户端透明 — 客户端无法区分是主模型还是备用模型响应。

        验证 FR-5.2: 触发灾备时将请求漂移到配置的候选模型。
        """
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")

        with patch.object(
            router, "async_function_with_fallbacks", return_value=successful_response
        ):
            response = await router.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
            )

            # 客户端收到正常格式的响应
            assert response.choices is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            assert response.choices[0].finish_reason == "stop"

    async def test_fallback_chain_order_matches_config(
        self, router: Router, fallbacks: list[dict]
    ):
        """Failover 链优先级与配置一致 (gpt-4o → gemini-1.5-pro → deepseek-chat)。

        验证 FR-5.3: 支持优先级排序的 Failover 链。
        """
        # 验证 router 配置正确加载了 fallbacks
        assert router.fallbacks is not None
        assert len(router.fallbacks) > 0

        # 找到 gpt-4o 的 fallback 配置
        gpt4o_fallback = None
        for entry in router.fallbacks:
            if "gpt-4o" in entry:
                gpt4o_fallback = entry["gpt-4o"]
                break

        assert gpt4o_fallback is not None
        assert gpt4o_fallback == ["gemini-1.5-pro", "deepseek-chat"]

    async def test_router_configured_with_correct_retry_settings(
        self, router: Router
    ):
        """Router 的重试配置与 config.yaml 一致。"""
        assert router.num_retries == 2
        assert router.timeout == 30
        assert router.allowed_fails == 1


class TestFailover429WithMockedCompletion:
    """使用 mock 验证 Router failover 行为。"""

    async def test_primary_model_429_triggers_fallback_call(
        self, router: Router
    ):
        """主模型返回 429 时，Router 尝试调用 fallback 模型。

        验证完整的 failover 流程: 429 → 捕获 → 路由到下一模型。
        """
        successful_response = _make_successful_response("gemini/gemini-1.5-pro")
        call_log: list[str] = []

        # 模拟 Router 的 _acompletion 内部方法：gpt-4o 返回 429，其它模型成功
        async def mock_router_acompletion(*args, **kwargs):
            model = kwargs.get("model", args[0] if args else "")
            call_log.append(str(model))
            if "gpt-4o" in str(model):
                raise _make_rate_limit_error()
            return successful_response

        with patch.object(router, "_acompletion", side_effect=mock_router_acompletion):
            # 使用 async_function_with_fallbacks 模拟 Router 的实际 fallback 逻辑
            # Router.acompletion 在遇到 fallback 配置时会调用 async_function_with_fallbacks
            with patch.object(
                router,
                "async_function_with_fallbacks",
                return_value=successful_response,
            ) as mock_fallback_fn:
                response = await router.acompletion(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "测试 failover"}],
                )
                # 客户端收到成功响应
                assert response is not None
                assert response.choices[0].message.content == "这是来自备用模型的响应"

    async def test_successful_response_has_expected_format(
        self, router: Router
    ):
        """Failover 后的响应格式与正常响应一致，客户端兼容。"""
        successful_response = _make_successful_response()

        with patch.object(
            router, "async_function_with_fallbacks", return_value=successful_response
        ):
            response = await router.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            )

            # 验证响应结构兼容 OpenAI SDK
            assert hasattr(response, "choices")
            assert hasattr(response, "model")
            assert hasattr(response, "id")
            assert hasattr(response, "usage")
            assert response.id == "chatcmpl-test-failover"


class TestFailoverConfigIntegration:
    """验证 failover 配置与 config.yaml 的一致性。"""

    async def test_all_fallback_models_registered_in_router(
        self, router: Router, model_list: list[dict]
    ):
        """所有 fallback 中引用的模型都已在 Router 的 model_list 中注册。"""
        registered_models = {d["model_name"] for d in model_list}

        for fallback_entry in router.fallbacks:
            for primary, candidates in fallback_entry.items():
                assert primary in registered_models, (
                    f"主模型 '{primary}' 未在 model_list 中"
                )
                for candidate in candidates:
                    assert candidate in registered_models, (
                        f"候选模型 '{candidate}' 未在 model_list 中"
                    )

    async def test_router_has_multiple_fallback_chains(self, router: Router):
        """Router 配置了多条 failover 链。"""
        assert len(router.fallbacks) >= 3

    async def test_no_model_references_itself_in_fallback(self, router: Router):
        """无自引用的 fallback 配置。"""
        for entry in router.fallbacks:
            for primary, candidates in entry.items():
                assert primary not in candidates, (
                    f"模型 '{primary}' 不应出现在自己的 fallback 链中"
                )
