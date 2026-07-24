"""Tests for aegis_router.storage.redis_client module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegis_router.storage.redis_client import (
    RedisClient,
    get_redis_client,
    reset_redis_client,
    _KEY_PII_REQUEST,
    _KEY_PII_SESSION,
    _DEFAULT_REQUEST_TTL,
    _DEFAULT_SESSION_TTL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """每个测试前后重置单例。"""
    reset_redis_client()
    yield
    reset_redis_client()


@pytest.fixture
def mock_redis():
    """创建一个模拟的 async Redis 客户端。"""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def client(mock_redis) -> RedisClient:
    """创建使用模拟 Redis 的 RedisClient 实例。"""
    rc = RedisClient(url="redis://localhost:6379/0")
    rc._client = mock_redis
    return rc


# ---------------------------------------------------------------------------
# Tests: Import
# ---------------------------------------------------------------------------


class TestImport:
    """验证模块可以正确导入。"""

    def test_import_redis_client_class(self):
        from aegis_router.storage.redis_client import RedisClient
        assert RedisClient is not None

    def test_import_factory_function(self):
        from aegis_router.storage.redis_client import get_redis_client
        assert callable(get_redis_client)


# ---------------------------------------------------------------------------
# Tests: Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """验证 RedisClient 可以正常实例化。"""

    def test_standalone_mode(self):
        rc = RedisClient(url="redis://localhost:6379/0")
        assert rc._mode == "standalone"
        assert rc._client is not None

    def test_sentinel_mode_requires_params(self):
        with pytest.raises(ValueError, match="sentinel_master"):
            RedisClient(mode="sentinel")

    def test_sentinel_mode_with_params(self):
        with patch("aegis_router.storage.redis_client.Sentinel") as mock_sentinel:
            mock_sentinel.return_value.master_for.return_value = MagicMock()
            rc = RedisClient(
                mode="sentinel",
                sentinel_master="aegis-master",
                sentinel_nodes=[("localhost", 26379)],
                password="secret",
            )
            assert rc._mode == "sentinel"
            mock_sentinel.assert_called_once()

    def test_default_url(self):
        rc = RedisClient()
        assert rc._mode == "standalone"
        assert rc._client is not None

    def test_max_retries_default(self):
        rc = RedisClient()
        assert rc._max_retries == 3


# ---------------------------------------------------------------------------
# Tests: store_mapping
# ---------------------------------------------------------------------------


class TestStoreMapping:
    """验证 store_mapping 正确存储映射。"""

    async def test_store_mapping_basic(self, client: RedisClient, mock_redis):
        mapping = {"[PERSON_1]": "张三", "[PHONE_1]": "13800138000"}
        await client.store_mapping("sess-1", "req-1", mapping)

        expected_key = "aegis:pii:sess-1:req-1"
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == expected_key
        assert call_args[1]["ex"] == _DEFAULT_REQUEST_TTL

    async def test_store_mapping_custom_ttl(self, client: RedisClient, mock_redis):
        await client.store_mapping("sess-1", "req-1", {"[PERSON_1]": "李四"}, ttl=600)

        call_args = mock_redis.set.call_args
        assert call_args[1]["ex"] == 600

    async def test_store_mapping_serializes_json(self, client: RedisClient, mock_redis):
        import orjson

        mapping = {"[EMAIL_1]": "test@example.com"}
        await client.store_mapping("s1", "r1", mapping)

        call_args = mock_redis.set.call_args
        stored_value = call_args[0][1]
        assert orjson.loads(stored_value) == mapping


# ---------------------------------------------------------------------------
# Tests: get_mapping
# ---------------------------------------------------------------------------


class TestGetMapping:
    """验证 get_mapping 正确检索映射。"""

    async def test_get_mapping_request_level(self, client: RedisClient, mock_redis):
        import orjson

        mapping = {"[PERSON_1]": "张三"}
        mock_redis.get = AsyncMock(return_value=orjson.dumps(mapping))

        result = await client.get_mapping("req-1", session_id="sess-1")
        assert result == mapping

        expected_key = "aegis:pii:sess-1:req-1"
        mock_redis.get.assert_called_once_with(expected_key)

    async def test_get_mapping_fallback_to_session(self, client: RedisClient, mock_redis):
        import orjson

        session_mapping = {"[PERSON_1]": "王五"}

        # 第一次调用 (request-level) 返回 None，第二次 (session-level) 返回数据
        mock_redis.get = AsyncMock(
            side_effect=[None, orjson.dumps(session_mapping)]
        )

        result = await client.get_mapping("req-99", session_id="sess-1")
        assert result == session_mapping

        # 验证第二次调用使用了 session key
        calls = mock_redis.get.call_args_list
        assert calls[0][0][0] == "aegis:pii:sess-1:req-99"
        assert calls[1][0][0] == "aegis:pii:session:sess-1"

    async def test_get_mapping_returns_empty_when_not_found(
        self, client: RedisClient, mock_redis
    ):
        mock_redis.get = AsyncMock(return_value=None)
        result = await client.get_mapping("req-404", session_id="sess-1")
        assert result == {}

    async def test_get_mapping_no_session_id(self, client: RedisClient, mock_redis):
        """没有 session_id 时直接返回空字典。"""
        result = await client.get_mapping("req-1", session_id=None)
        assert result == {}
        mock_redis.get.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: update_session_mapping
# ---------------------------------------------------------------------------


class TestUpdateSessionMapping:
    """验证 update_session_mapping 正确合并映射。"""

    async def test_merge_into_empty_session(self, client: RedisClient, mock_redis):
        import orjson

        mock_redis.get = AsyncMock(return_value=None)
        new_mapping = {"[PERSON_1]": "赵六"}

        await client.update_session_mapping("sess-1", new_mapping)

        # 验证写入
        call_args = mock_redis.set.call_args
        key = call_args[0][0]
        value = orjson.loads(call_args[0][1])
        assert key == "aegis:pii:session:sess-1"
        assert value == new_mapping
        assert call_args[1]["ex"] == _DEFAULT_SESSION_TTL

    async def test_merge_into_existing_session(self, client: RedisClient, mock_redis):
        import orjson

        existing = {"[PERSON_1]": "张三", "[PHONE_1]": "13800138000"}
        mock_redis.get = AsyncMock(return_value=orjson.dumps(existing))

        new_mapping = {"[PERSON_2]": "李四", "[PHONE_1]": "13900139000"}
        await client.update_session_mapping("sess-1", new_mapping)

        # 验证合并结果: 新映射覆盖旧值
        call_args = mock_redis.set.call_args
        merged = orjson.loads(call_args[0][1])
        assert merged == {
            "[PERSON_1]": "张三",
            "[PHONE_1]": "13900139000",  # 被覆盖
            "[PERSON_2]": "李四",        # 新增
        }

    async def test_custom_ttl(self, client: RedisClient, mock_redis):
        mock_redis.get = AsyncMock(return_value=None)
        await client.update_session_mapping("sess-1", {"[X_1]": "val"}, ttl=7200)

        call_args = mock_redis.set.call_args
        assert call_args[1]["ex"] == 7200


# ---------------------------------------------------------------------------
# Tests: TTL
# ---------------------------------------------------------------------------


class TestTTL:
    """验证默认 TTL 值符合设计规范。"""

    def test_request_ttl_is_1800(self):
        assert _DEFAULT_REQUEST_TTL == 1800

    def test_session_ttl_is_3600(self):
        assert _DEFAULT_SESSION_TTL == 3600


# ---------------------------------------------------------------------------
# Tests: health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """验证 health_check 返回正确状态。"""

    async def test_healthy(self, client: RedisClient, mock_redis):
        mock_redis.ping = AsyncMock(return_value=True)
        result = await client.health_check()
        assert result == {"status": "healthy"}

    async def test_unhealthy_on_exception(self, client: RedisClient, mock_redis):
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("refused"))
        result = await client.health_check()
        assert result["status"] == "unhealthy"
        assert "refused" in result["error"]

    async def test_unhealthy_on_false_ping(self, client: RedisClient, mock_redis):
        mock_redis.ping = AsyncMock(return_value=False)
        result = await client.health_check()
        assert result["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# Tests: Graceful degradation (retry logic)
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """验证 Redis 不可用时的重试和降级行为。"""

    async def test_retry_on_connection_error(self, client: RedisClient, mock_redis):
        import orjson
        import redis.asyncio as aioredis

        mapping = {"[PERSON_1]": "测试"}
        # 前两次失败，第三次成功
        mock_redis.get = AsyncMock(
            side_effect=[
                aioredis.ConnectionError("connection reset"),
                aioredis.ConnectionError("connection reset"),
                orjson.dumps(mapping),
            ]
        )

        result = await client.get_mapping("r1", session_id="s1")
        assert result == mapping
        assert mock_redis.get.call_count == 3

    async def test_raises_after_max_retries(self, client: RedisClient, mock_redis):
        import redis.asyncio as aioredis

        mock_redis.set = AsyncMock(
            side_effect=aioredis.ConnectionError("refused")
        )

        with pytest.raises(aioredis.ConnectionError):
            await client.store_mapping("s1", "r1", {"[X]": "v"})

        assert mock_redis.set.call_count == 3  # max_retries = 3

    async def test_retry_on_timeout(self, client: RedisClient, mock_redis):
        import orjson
        import redis.asyncio as aioredis

        # 第一次超时，第二次成功
        mock_redis.get = AsyncMock(
            side_effect=[
                aioredis.TimeoutError("timed out"),
                orjson.dumps({"[A]": "b"}),
            ]
        )

        result = await client.get_mapping("r1", session_id="s1")
        assert result == {"[A]": "b"}


# ---------------------------------------------------------------------------
# Tests: close
# ---------------------------------------------------------------------------


class TestClose:
    """验证关闭连接的行为。"""

    async def test_close_calls_aclose(self, client: RedisClient, mock_redis):
        await client.close()
        mock_redis.aclose.assert_called_once()
        assert client._client is None

    async def test_close_idempotent(self, client: RedisClient, mock_redis):
        await client.close()
        await client.close()  # 第二次应该不报错
        mock_redis.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Singleton factory
# ---------------------------------------------------------------------------


class TestSingletonFactory:
    """验证 get_redis_client 的单例行为。"""

    def test_returns_same_instance(self):
        c1 = get_redis_client(url="redis://localhost:6379/0")
        c2 = get_redis_client()
        assert c1 is c2

    def test_reset_clears_instance(self):
        c1 = get_redis_client(url="redis://localhost:6379/0")
        reset_redis_client()
        c2 = get_redis_client(url="redis://localhost:6379/1")
        assert c1 is not c2
