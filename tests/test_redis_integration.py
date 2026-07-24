"""Integration-style tests: PIIMasker → RedisClient → Redis 端到端映射存储验证

验证从 masker.mask() 开始，经过 RedisClient.store_mapping / update_session_mapping，
到最终写入 Redis 时的 key 格式和 TTL 值是否符合设计规范。

Validates:
- FR-2.4: 占位符与真实值的映射关系存入 Redis，关联到 request_id
- NFR-2.1: Redis 隐私映射表 TTL = 30 分钟，到期物理擦除
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aegis_router.clawvault.masker import PIIMasker
from aegis_router.storage.redis_client import (
    RedisClient,
    _DEFAULT_REQUEST_TTL,
    _DEFAULT_SESSION_TTL,
    _KEY_PII_REQUEST,
    _KEY_PII_SESSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_raw_redis():
    """模拟底层 async Redis 连接 (redis.asyncio.Redis 实例)。"""
    mock = AsyncMock()
    mock.set = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.ping = AsyncMock(return_value=True)
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def redis_client(mock_raw_redis) -> RedisClient:
    """创建 RedisClient 实例，注入底层 mock。"""
    rc = RedisClient(url="redis://localhost:6379/0")
    rc._client = mock_raw_redis
    return rc


@pytest.fixture
def masker_with_real_redis_client(redis_client):
    """创建 PIIMasker，注入真实 RedisClient（底层 Redis 连接仍为 mock）。

    这样可以验证 masker → RedisClient → Redis 的完整调用链。
    """
    return PIIMasker(
        redis_client=redis_client,
        language="en",
        nlp_model="en_core_web_sm",
        score_threshold=0.4,
    )


# ---------------------------------------------------------------------------
# Tests: End-to-end key format and TTL verification
# ---------------------------------------------------------------------------


class TestMaskerToRedisKeyAndTTL:
    """验证 PIIMasker.mask() 端到端写入 Redis 时的 key 格式和 TTL。

    这些测试使用真实的 RedisClient 实例（仅底层 Redis 连接被 mock），
    从而验证完整的调用链：masker.mask() → redis_client.store_mapping()
    → redis_client._execute_with_retry("set", key, value, ex=ttl)。
    """

    async def test_request_mapping_key_format(
        self, masker_with_real_redis_client, mock_raw_redis
    ):
        """V2-7: 请求映射 key 格式为 aegis:pii:{session_id}:{request_id}。"""
        text = "Please email me at alice@example.com for details."
        session_id = "sess-abc-123"
        request_id = "req-xyz-789"

        await masker_with_real_redis_client.mask(
            text, session_id=session_id, request_id=request_id
        )

        # store_mapping 应该触发一次 set 调用 (request-level)
        # update_session_mapping 会触发 get + set (session-level)
        set_calls = mock_raw_redis.set.call_args_list

        # 至少有 2 次 set 调用: request mapping + session mapping
        assert len(set_calls) >= 2

        # 第一次 set 调用应该是 request mapping
        request_key = set_calls[0][0][0]
        expected_request_key = f"aegis:pii:{session_id}:{request_id}"
        assert request_key == expected_request_key

    async def test_request_mapping_ttl_is_1800(
        self, masker_with_real_redis_client, mock_raw_redis
    ):
        """V2-7: 请求映射 TTL = 1800 秒 (30 分钟)。"""
        text = "My IP address is 10.20.30.40 for the record."
        session_id = "sess-ttl-test"
        request_id = "req-ttl-001"

        await masker_with_real_redis_client.mask(
            text, session_id=session_id, request_id=request_id
        )

        set_calls = mock_raw_redis.set.call_args_list
        assert len(set_calls) >= 1

        # 验证 request mapping 的 TTL
        request_set_call = set_calls[0]
        assert request_set_call[1]["ex"] == 1800
        assert request_set_call[1]["ex"] == _DEFAULT_REQUEST_TTL

    async def test_session_mapping_key_format(
        self, masker_with_real_redis_client, mock_raw_redis
    ):
        """V2-7: 会话映射 key 格式为 aegis:pii:session:{session_id}。"""
        text = "Card number 4111111111111111 is on file."
        session_id = "sess-session-key"
        request_id = "req-001"

        await masker_with_real_redis_client.mask(
            text, session_id=session_id, request_id=request_id
        )

        set_calls = mock_raw_redis.set.call_args_list
        assert len(set_calls) >= 2

        # 第二次 set 调用应该是 session mapping
        session_key = set_calls[1][0][0]
        expected_session_key = f"aegis:pii:session:{session_id}"
        assert session_key == expected_session_key

    async def test_session_mapping_ttl_is_3600(
        self, masker_with_real_redis_client, mock_raw_redis
    ):
        """V2-7: 会话映射 TTL = 3600 秒 (1 小时)。"""
        text = "Reach me at user@domain.org anytime."
        session_id = "sess-session-ttl"
        request_id = "req-002"

        await masker_with_real_redis_client.mask(
            text, session_id=session_id, request_id=request_id
        )

        set_calls = mock_raw_redis.set.call_args_list
        assert len(set_calls) >= 2

        # 验证 session mapping 的 TTL
        session_set_call = set_calls[1]
        assert session_set_call[1]["ex"] == 3600
        assert session_set_call[1]["ex"] == _DEFAULT_SESSION_TTL

    async def test_mapping_data_stored_as_json(
        self, masker_with_real_redis_client, mock_raw_redis
    ):
        """V2-7: 映射数据以 JSON (orjson) 格式序列化存储。"""
        import orjson

        text = "Contact support@company.io for help."
        session_id = "sess-json"
        request_id = "req-json"

        result = await masker_with_real_redis_client.mask(
            text, session_id=session_id, request_id=request_id
        )

        set_calls = mock_raw_redis.set.call_args_list
        assert len(set_calls) >= 1

        # 验证 request mapping 的 value 是有效的 JSON
        stored_value = set_calls[0][0][1]
        deserialized = orjson.loads(stored_value)

        # 反序列化后应与 masker 返回的 mapping 一致
        assert deserialized == result["mapping"]
        # mapping 中应包含邮箱占位符
        assert any("EMAIL" in k for k in deserialized.keys())

    async def test_no_redis_writes_when_no_pii_detected(
        self, masker_with_real_redis_client, mock_raw_redis
    ):
        """当无 PII 时，不应有任何 Redis 写入操作。"""
        text = "The weather is cloudy today with a chance of rain."
        session_id = "sess-no-pii"
        request_id = "req-no-pii"

        result = await masker_with_real_redis_client.mask(
            text, session_id=session_id, request_id=request_id
        )

        assert result["mapping"] == {}
        mock_raw_redis.set.assert_not_called()

    async def test_key_pattern_constants_match_design(self):
        """验证 key 模式常量与设计文档一致。"""
        assert _KEY_PII_REQUEST == "aegis:pii:{session}:{request}"
        assert _KEY_PII_SESSION == "aegis:pii:session:{session}"
        assert _DEFAULT_REQUEST_TTL == 1800
        assert _DEFAULT_SESSION_TTL == 3600

    async def test_full_flow_multiple_pii_entities(
        self, masker_with_real_redis_client, mock_raw_redis
    ):
        """完整流程: 多个 PII 实体 → 正确的 key 和 TTL。"""
        import orjson

        text = (
            "Please contact admin@corp.com or visit 192.168.1.1 for support."
        )
        session_id = "sess-multi"
        request_id = "req-multi"

        result = await masker_with_real_redis_client.mask(
            text, session_id=session_id, request_id=request_id
        )

        # 应检测到至少两种 PII
        assert len(result["mapping"]) >= 2

        set_calls = mock_raw_redis.set.call_args_list
        assert len(set_calls) >= 2

        # 验证 request mapping
        req_call = set_calls[0]
        assert req_call[0][0] == f"aegis:pii:{session_id}:{request_id}"
        assert req_call[1]["ex"] == 1800

        # 验证存储的映射包含所有检测到的 PII
        stored_mapping = orjson.loads(req_call[0][1])
        assert stored_mapping == result["mapping"]

        # 验证 session mapping
        sess_call = set_calls[1]
        assert sess_call[0][0] == f"aegis:pii:session:{session_id}"
        assert sess_call[1]["ex"] == 3600
