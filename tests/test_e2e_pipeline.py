"""E2E Integration Tests — Complete pipeline with Mock LLM.

Tests cover:
- TC-E2E-001: Non-streaming full pipeline (PII masking → routing → response restore)
- TC-E2E-002: Streaming full pipeline (chunk placeholder restoration)
- TC-E2E-003: Multi-turn session consistency (3 rounds, same placeholders)
- TC-E2E-004: Concurrent 50 requests with isolated PII mappings
"""

from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.config import (
    ClassifierConfig,
    RoutingConfig,
    TrivialConfig,
)
from aegis_router.router.model_classifier import ClassifierResult, ModelClassifier
from aegis_router.router.route_resolver import RouteResolver
from aegis_router.router.rule_engine import RuleEngine, RuleEngineResult


# ---------------------------------------------------------------------------
# Mock Data Classes
# ---------------------------------------------------------------------------


@dataclass
class MockMessage:
    """Mock LiteLLM message object."""
    content: str
    role: str = "assistant"


@dataclass
class MockChoice:
    """Mock LiteLLM choice object."""
    message: MockMessage
    index: int = 0


@dataclass
class MockResponse:
    """Mock LiteLLM ModelResponse object."""
    choices: list


def make_response(content: str) -> MockResponse:
    """Create a mock LiteLLM response with given content."""
    return MockResponse(choices=[MockChoice(message=MockMessage(content=content))])


@dataclass
class MockDelta:
    """Mock LiteLLM streaming delta object."""
    content: Optional[str] = None
    role: Optional[str] = None


@dataclass
class MockStreamChoice:
    """Mock LiteLLM streaming choice object."""
    delta: MockDelta
    index: int = 0


@dataclass
class MockStreamChunk:
    """Mock LiteLLM streaming chunk object."""
    choices: list = field(default_factory=list)


def make_stream_chunk(content: Optional[str] = None, role: Optional[str] = None) -> MockStreamChunk:
    """Create a mock streaming chunk with given content."""
    delta = MockDelta(content=content, role=role)
    choice = MockStreamChoice(delta=delta)
    return MockStreamChunk(choices=[choice])


async def async_iter_chunks(chunks: list):
    """Create an async iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


async def collect_stream_content(async_gen) -> str:
    """Collect all text content from a streaming async generator."""
    texts = []
    async for chunk in async_gen:
        content = None
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                content = choice.delta.content
        if content:
            texts.append(content)
    return "".join(texts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock()
    pool.max_connections = 10
    return pool


@pytest.fixture
def mock_rule_engine():
    """Create a mock RuleEngine that does NOT match (non-trivial prompt)."""
    engine = MagicMock(spec=RuleEngine)
    engine.check.return_value = RuleEngineResult(matched=False)
    return engine


@pytest.fixture
def mock_classifier():
    """Create a mock ModelClassifier returning a mid-range score."""
    classifier = MagicMock(spec=ModelClassifier)
    classifier.aclassify = AsyncMock(
        return_value=ClassifierResult(score=0.5, classifier_type="mf", latency_ms=5.0)
    )
    return classifier


@pytest.fixture
def mock_resolver():
    """Create a mock RouteResolver that routes to deepseek/deepseek-chat."""
    resolver = MagicMock(spec=RouteResolver)
    resolver.resolve.return_value = {
        "model": "deepseek/deepseek-chat",
        "reason": "single_match",
    }
    return resolver


@pytest.fixture
def routing_config():
    """Create a routing config for E2E tests."""
    return RoutingConfig(
        score_input="masked",
        trivial=TrivialConfig(enabled=True, max_length=30, target_model="local-7b"),
        classifier=ClassifierConfig(type="mf"),
        overlap_strategy="lowest_cost",
        fallback_model="deepseek-v3",
    )


@pytest.fixture
def e2e_callback(mock_pool, mock_rule_engine, mock_classifier, mock_resolver, routing_config):
    """Create a SmartRouterCallback wired for E2E testing."""
    cb = SmartRouterCallback(
        pool=mock_pool,
        enable_routing=True,
        rule_engine=mock_rule_engine,
        classifier=mock_classifier,
    )
    cb._route_resolver = mock_resolver
    cb._routing_config = routing_config
    return cb


# ---------------------------------------------------------------------------
# TC-E2E-001: Non-streaming full pipeline
# ---------------------------------------------------------------------------


class TestE2E001_NonStreamingFullPipeline:
    """TC-E2E-001: 客户端发送含 PII 的 prompt → 脱敏 → 路由到正确模型
    → Mock 响应含占位符 → 还原 → 客户端收到完整原文。"""

    async def test_full_pipeline_mask_route_restore(
        self, e2e_callback, mock_pool, mock_resolver
    ):
        """完整非流式管道: PII 脱敏 → 路由 → LLM 响应 → 还原。"""
        # === Phase 1: pre_call_hook ===
        request_data = {
            "messages": [
                {"role": "user", "content": "我叫张三，手机号是13800138000，帮我写一份报告"},
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-e2e-001",
                "request_id": "req-e2e-001",
            },
        }

        # Pool mock: compliance passes, then mask replaces PII
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "我叫[PERSON_1]，手机号是[PHONE_1]，帮我写一份报告",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 9, "end": 20, "score": 0.99},
                ],
            },
        ]

        await e2e_callback.async_pre_call_hook({}, None, request_data, "completion")

        # Verify: message is masked
        assert request_data["messages"][0]["content"] == (
            "我叫[PERSON_1]，手机号是[PHONE_1]，帮我写一份报告"
        )
        # Verify: model is routed
        assert request_data["model"] == "deepseek/deepseek-chat"
        assert request_data["metadata"]["target_model"] == "deepseek/deepseek-chat"

        # === Phase 2: Mock LLM response with placeholders ===
        llm_response = make_response(
            "[PERSON_1] 您好！以下是为您撰写的报告。联系方式: [PHONE_1]。"
        )

        # === Phase 3: async_log_success_event (restore) ===
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "restored_text": "张三 您好！以下是为您撰写的报告。联系方式: 13800138000。"
        }

        kwargs = {
            "metadata": request_data["metadata"],
            "model": "deepseek/deepseek-chat",
        }
        await e2e_callback.async_log_success_event(kwargs, llm_response, None, None)

        # === Verify: client receives fully restored text ===
        client_response = llm_response.choices[0].message.content
        assert client_response == "张三 您好！以下是为您撰写的报告。联系方式: 13800138000。"
        assert "[PERSON_1]" not in client_response
        assert "[PHONE_1]" not in client_response

        # Verify restore was called correctly
        mock_pool.call.assert_called_once_with(
            "restore",
            {
                "text": "[PERSON_1] 您好！以下是为您撰写的报告。联系方式: [PHONE_1]。",
                "request_id": "req-e2e-001",
                "session_id": "sess-e2e-001",
            },
        )


# ---------------------------------------------------------------------------
# TC-E2E-002: Streaming full pipeline
# ---------------------------------------------------------------------------


class TestE2E002_StreamingFullPipeline:
    """TC-E2E-002: 相同测试 streaming 模式 → chunk 中占位符正确还原。"""

    async def test_streaming_pipeline_with_split_placeholders(
        self, e2e_callback, mock_pool, mock_resolver
    ):
        """流式管道: PII 脱敏 → 路由 → 流式响应含分割占位符 → 还原拼接正确。"""
        # === Phase 1: pre_call_hook (same as non-streaming) ===
        request_data = {
            "messages": [
                {"role": "user", "content": "我叫李四，电话13912345678，查一下我的订单"},
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-e2e-002",
                "request_id": "req-e2e-002",
            },
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "我叫[PERSON_1]，电话[PHONE_1]，查一下我的订单",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 6, "end": 17, "score": 0.99},
                ],
            },
        ]

        await e2e_callback.async_pre_call_hook({}, None, request_data, "completion")

        # Verify masking and routing
        assert "[PERSON_1]" in request_data["messages"][0]["content"]
        assert "[PHONE_1]" in request_data["messages"][0]["content"]
        assert request_data["model"] == "deepseek/deepseek-chat"

        # === Phase 2: Streaming response with split placeholders ===
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "李四", "[PHONE_1]": "13912345678"}
        }

        # Simulate LLM streaming chunks with split placeholders
        chunks = [
            make_stream_chunk(content="[PER"),
            make_stream_chunk(content="SON_1] 您好！您的订单已发货，"),
            make_stream_chunk(content="联系电话 [PHO"),
            make_stream_chunk(content="NE_1] 将收到短信通知。"),
        ]

        stream = e2e_callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)

        # === Verify: fully restored text ===
        assert result == "李四 您好！您的订单已发货，联系电话 13912345678 将收到短信通知。"
        assert "[PERSON_1]" not in result
        assert "[PHONE_1]" not in result


# ---------------------------------------------------------------------------
# TC-E2E-003: Multi-turn session consistency
# ---------------------------------------------------------------------------


class TestE2E003_MultiTurnSessionConsistency:
    """TC-E2E-003: 多轮对话（3轮），session 内占位符一致性验证。"""

    async def test_three_round_session_placeholder_consistency(
        self, e2e_callback, mock_pool, mock_resolver
    ):
        """3轮对话同 session，同一 PII 始终映射到相同占位符，还原正确。"""
        session_id = "sess-e2e-003-multi"

        # Define 3 rounds of conversation
        rounds = [
            {
                "user_msg": "我是王五，手机号18611112222，帮我查余额",
                "masked_msg": "我是[PERSON_1]，手机号[PHONE_1]，帮我查余额",
                "llm_response": "[PERSON_1] 的账户余额为 1000 元。",
                "restored_response": "王五 的账户余额为 1000 元。",
            },
            {
                "user_msg": "王五想转账500元",
                "masked_msg": "[PERSON_1]想转账500元",
                "llm_response": "[PERSON_1] 的转账已处理，余额剩余 500 元。",
                "restored_response": "王五 的转账已处理，余额剩余 500 元。",
            },
            {
                "user_msg": "再发一条通知到18611112222",
                "masked_msg": "再发一条通知到[PHONE_1]",
                "llm_response": "通知已发送至 [PHONE_1]。",
                "restored_response": "通知已发送至 18611112222。",
            },
        ]

        for i, round_data in enumerate(rounds, start=1):
            request_id = f"req-e2e-003-round{i}"

            # --- pre_call_hook ---
            request_data = {
                "messages": [
                    {"role": "user", "content": round_data["user_msg"]},
                ],
                "model": "gpt-4o",
                "metadata": {
                    "session_id": session_id,
                    "request_id": request_id,
                },
            }

            mock_pool.call.reset_mock()
            mock_pool.call.side_effect = [
                {"passed": True, "violations": [], "mode": "strict"},
                {
                    "masked_text": round_data["masked_msg"],
                    "entities_found": [],
                },
            ]

            await e2e_callback.async_pre_call_hook(
                {}, None, request_data, "completion"
            )

            # Verify masking consistency
            assert request_data["messages"][0]["content"] == round_data["masked_msg"]

            # --- Mock LLM response ---
            llm_response = make_response(round_data["llm_response"])

            # --- async_log_success_event (restore) ---
            mock_pool.call.reset_mock()
            mock_pool.call.side_effect = None
            mock_pool.call.return_value = {
                "restored_text": round_data["restored_response"]
            }

            kwargs = {
                "metadata": request_data["metadata"],
                "model": "deepseek/deepseek-chat",
            }
            await e2e_callback.async_log_success_event(
                kwargs, llm_response, None, None
            )

            # Verify restored text
            client_text = llm_response.choices[0].message.content
            assert client_text == round_data["restored_response"], (
                f"Round {i}: expected '{round_data['restored_response']}', "
                f"got '{client_text}'"
            )
            assert "[PERSON_1]" not in client_text
            assert "[PHONE_1]" not in client_text

    async def test_same_pii_uses_same_placeholder_across_rounds(
        self, e2e_callback, mock_pool, mock_resolver
    ):
        """验证 ClawVault mask 被调用时始终传递相同的 session_id，
        确保同一 PII 映射到同一占位符。"""
        session_id = "sess-e2e-003-verify"
        user_messages = [
            "张三的手机号是13800138000",
            "请通知张三",
            "张三确认收到",
        ]

        for i, msg in enumerate(user_messages, start=1):
            request_data = {
                "messages": [{"role": "user", "content": msg}],
                "model": "gpt-4o",
                "metadata": {
                    "session_id": session_id,
                    "request_id": f"req-verify-{i}",
                },
            }

            mock_pool.call.reset_mock()
            mock_pool.call.side_effect = [
                {"passed": True, "violations": [], "mode": "strict"},
                {
                    "masked_text": msg.replace("张三", "[PERSON_1]").replace(
                        "13800138000", "[PHONE_1]"
                    ),
                    "entities_found": [],
                },
            ]

            await e2e_callback.async_pre_call_hook(
                {}, None, request_data, "completion"
            )

            # Verify the mask call always includes the same session_id
            mask_call = mock_pool.call.call_args_list[1]
            assert mask_call[0][1]["session_id"] == session_id


# ---------------------------------------------------------------------------
# TC-E2E-004: Concurrent 50 requests with isolated PII mappings
# ---------------------------------------------------------------------------


class TestE2E004_ConcurrentRequestIsolation:
    """TC-E2E-004: 并发 50 个请求，各自的 PII 映射互不干扰。"""

    async def test_50_concurrent_requests_pii_isolation(
        self, mock_pool, mock_rule_engine, mock_classifier, mock_resolver, routing_config
    ):
        """50 个并发请求各自有不同 PII，验证还原后互不干扰。"""
        NUM_REQUESTS = 50

        # Create a callback per test (shared is fine since pool is mocked per-call)
        cb = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=mock_rule_engine,
            classifier=mock_classifier,
        )
        cb._route_resolver = mock_resolver
        cb._routing_config = routing_config

        # Generate unique PII for each request
        requests_data = []
        for i in range(NUM_REQUESTS):
            name = f"用户{i:03d}"
            phone = f"138{i:08d}"
            requests_data.append({
                "name": name,
                "phone": phone,
                "session_id": f"sess-concurrent-{i:03d}",
                "request_id": f"req-concurrent-{i:03d}",
            })

        # Track results for verification
        results = [None] * NUM_REQUESTS

        async def run_single_request(idx: int):
            """Run a single request through the full pipeline."""
            rd = requests_data[idx]
            name = rd["name"]
            phone = rd["phone"]

            # Build request
            data = {
                "messages": [
                    {"role": "user", "content": f"我叫{name}，电话{phone}，查询"},
                ],
                "model": "gpt-4o",
                "metadata": {
                    "session_id": rd["session_id"],
                    "request_id": rd["request_id"],
                },
            }

            # Create a per-request pool mock to avoid side_effect conflicts
            local_pool = MagicMock(spec=ClawVaultPool)
            local_pool.call = AsyncMock()
            local_pool.max_connections = 10

            local_cb = SmartRouterCallback(
                pool=local_pool,
                enable_routing=True,
                rule_engine=mock_rule_engine,
                classifier=mock_classifier,
            )
            local_cb._route_resolver = mock_resolver
            local_cb._routing_config = routing_config

            # pre_call mock
            local_pool.call.side_effect = [
                {"passed": True, "violations": [], "mode": "strict"},
                {
                    "masked_text": f"我叫[PERSON_1]，电话[PHONE_1]，查询",
                    "entities_found": [
                        {"type": "PERSON", "start": 2, "end": 2 + len(name), "score": 0.95},
                        {"type": "PHONE_NUMBER", "start": 5, "end": 5 + len(phone), "score": 0.99},
                    ],
                },
            ]

            await local_cb.async_pre_call_hook({}, None, data, "completion")

            # Verify masking
            assert data["messages"][0]["content"] == "我叫[PERSON_1]，电话[PHONE_1]，查询"

            # Mock LLM response
            llm_response = make_response(
                f"[PERSON_1] 的查询结果: 电话 [PHONE_1] 已验证。"
            )

            # Restore mock
            local_pool.call.reset_mock()
            local_pool.call.side_effect = None
            local_pool.call.return_value = {
                "restored_text": f"{name} 的查询结果: 电话 {phone} 已验证。"
            }

            kwargs = {"metadata": data["metadata"], "model": "deepseek/deepseek-chat"}
            await local_cb.async_log_success_event(kwargs, llm_response, None, None)

            results[idx] = llm_response.choices[0].message.content

        # Run all 50 requests concurrently
        await asyncio.gather(
            *[run_single_request(i) for i in range(NUM_REQUESTS)]
        )

        # === Verify: each request got its OWN PII back ===
        for i in range(NUM_REQUESTS):
            rd = requests_data[i]
            expected = f"{rd['name']} 的查询结果: 电话 {rd['phone']} 已验证。"
            assert results[i] == expected, (
                f"Request {i}: expected '{expected}', got '{results[i]}'"
            )

        # Verify no cross-contamination: each result contains only its own PII
        for i in range(NUM_REQUESTS):
            for j in range(NUM_REQUESTS):
                if i == j:
                    continue
                # Other request's unique name should NOT appear in this result
                other_name = requests_data[j]["name"]
                assert other_name not in results[i], (
                    f"Cross-contamination: request {j}'s PII '{other_name}' "
                    f"found in request {i}'s result"
                )
