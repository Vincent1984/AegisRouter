"""异步 Redis 操作封装

提供 PII 映射存储、会话映射管理、健康检查等功能。
支持三种部署模式: standalone, sentinel, cluster。
"""

from __future__ import annotations

import logging
from typing import Optional

import orjson
import redis.asyncio as aioredis
from redis.asyncio.sentinel import Sentinel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key 模式常量
# ---------------------------------------------------------------------------

_KEY_PII_REQUEST = "aegis:pii:{session}:{request}"
_KEY_PII_SESSION = "aegis:pii:session:{session}"

# 默认 TTL (秒)
_DEFAULT_REQUEST_TTL = 1800   # 30 分钟
_DEFAULT_SESSION_TTL = 3600   # 1 小时


# ---------------------------------------------------------------------------
# Redis 客户端
# ---------------------------------------------------------------------------


class RedisClient:
    """异步 Redis 客户端封装。

    支持 standalone / sentinel / cluster 三种模式，提供 PII 映射的
    存储、读取、会话合并以及健康检查接口。

    Parameters
    ----------
    url : str | None
        Redis 连接 URL (standalone 模式)，默认 ``redis://localhost:6379/0``。
    mode : str
        部署模式: ``standalone`` | ``sentinel`` | ``cluster``。
    sentinel_master : str | None
        Sentinel 主节点名称 (sentinel 模式必填)。
    sentinel_nodes : list[tuple[str, int]] | None
        Sentinel 节点列表，如 ``[("host1", 26379), ("host2", 26379)]``。
    cluster_nodes : list[dict] | None
        Cluster 节点列表，如 ``[{"host": "h1", "port": 6379}]``。
    password : str | None
        Redis 密码。
    max_connections : int
        连接池最大连接数。
    socket_timeout : float
        Socket 超时 (秒)。
    retry_on_timeout : bool
        超时时是否自动重试。
    max_retries : int
        操作失败最大重试次数。
    """

    def __init__(
        self,
        url: str | None = None,
        mode: str = "standalone",
        sentinel_master: str | None = None,
        sentinel_nodes: list[tuple[str, int]] | None = None,
        cluster_nodes: list[dict] | None = None,
        password: str | None = None,
        max_connections: int = 100,
        socket_timeout: float = 0.5,
        retry_on_timeout: bool = True,
        max_retries: int = 3,
    ) -> None:
        self._mode = mode
        self._max_retries = max_retries
        self._client: Optional[aioredis.Redis] = None

        if mode == "sentinel":
            if not sentinel_master or not sentinel_nodes:
                raise ValueError("sentinel 模式需要提供 sentinel_master 和 sentinel_nodes")
            sentinel = Sentinel(
                sentinel_nodes,
                socket_timeout=socket_timeout,
                password=password,
            )
            self._client = sentinel.master_for(
                sentinel_master,
                redis_class=aioredis.Redis,
                socket_timeout=socket_timeout,
                retry_on_timeout=retry_on_timeout,
            )
        elif mode == "cluster":
            # redis-py 的 RedisCluster 异步版本
            from redis.asyncio.cluster import RedisCluster

            startup_nodes = cluster_nodes or [{"host": "localhost", "port": 6379}]
            self._client = RedisCluster(  # type: ignore[assignment]
                startup_nodes=[
                    aioredis.cluster.ClusterNode(**n) for n in startup_nodes  # type: ignore[attr-defined]
                ],
                password=password,
                socket_timeout=socket_timeout,
                retry_on_timeout=retry_on_timeout,
            )
        else:
            # standalone 模式
            redis_url = url or "redis://localhost:6379/0"
            pool = aioredis.ConnectionPool.from_url(
                redis_url,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                retry_on_timeout=retry_on_timeout,
                password=password,
                decode_responses=False,
            )
            self._client = aioredis.Redis(connection_pool=pool)

    # ------------------------------------------------------------------
    # PII 映射操作
    # ------------------------------------------------------------------

    async def store_mapping(
        self,
        session_id: str,
        request_id: str,
        mapping: dict,
        ttl: int = _DEFAULT_REQUEST_TTL,
    ) -> None:
        """存储单次请求的 PII 占位符映射。

        Parameters
        ----------
        session_id : str
            会话 ID。
        request_id : str
            请求 ID。
        mapping : dict
            占位符 → 真实值映射表，如 ``{"[PERSON_1]": "张三"}``。
        ttl : int
            过期时间 (秒)，默认 1800 (30 分钟)。
        """
        key = _KEY_PII_REQUEST.format(session=session_id, request=request_id)
        value = orjson.dumps(mapping)
        await self._execute_with_retry("set", key, value, ex=ttl)
        logger.debug("已存储请求映射: %s (TTL=%ds)", key, ttl)

    async def get_mapping(
        self,
        request_id: str,
        session_id: str | None = None,
    ) -> dict:
        """获取 PII 映射表。

        优先查找请求级映射，如果不存在且提供了 session_id 则回退到会话级映射。

        Parameters
        ----------
        request_id : str
            请求 ID。
        session_id : str | None
            会话 ID。如果提供，则在请求级映射不存在时回退到会话级。

        Returns
        -------
        dict
            映射表。如果不存在则返回空字典。
        """
        # 需要 session_id 来构建请求级 key
        if session_id:
            # 尝试请求级
            key = _KEY_PII_REQUEST.format(session=session_id, request=request_id)
            data = await self._execute_with_retry("get", key)
            if data:
                return orjson.loads(data)

            # 回退到会话级
            session_key = _KEY_PII_SESSION.format(session=session_id)
            data = await self._execute_with_retry("get", session_key)
            if data:
                return orjson.loads(data)

        return {}

    async def update_session_mapping(
        self,
        session_id: str,
        mapping: dict,
        ttl: int = _DEFAULT_SESSION_TTL,
    ) -> None:
        """合并新映射到会话级映射表。

        将新的占位符映射合并到已有的会话映射中（已有 key 不会被覆盖，
        新 key 会添加）。

        Parameters
        ----------
        session_id : str
            会话 ID。
        mapping : dict
            新增的占位符映射。
        ttl : int
            过期时间 (秒)，默认 3600 (1 小时)。
        """
        key = _KEY_PII_SESSION.format(session=session_id)

        # 读取已有映射
        existing_data = await self._execute_with_retry("get", key)
        if existing_data:
            existing = orjson.loads(existing_data)
        else:
            existing = {}

        # 合并: 新映射覆盖旧值（同一 placeholder 以最新为准）
        existing.update(mapping)

        value = orjson.dumps(existing)
        await self._execute_with_retry("set", key, value, ex=ttl)
        logger.debug("已更新会话映射: %s (TTL=%ds, 条目数=%d)", key, ttl, len(existing))

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        """检查 Redis 连接健康状态。

        Returns
        -------
        dict
            包含 ``status`` (``"healthy"`` 或 ``"unhealthy"``) 和
            可选的 ``error`` 信息。
        """
        try:
            assert self._client is not None
            result = await self._client.ping()
            if result:
                return {"status": "healthy"}
            return {"status": "unhealthy", "error": "ping returned False"}
        except Exception as e:
            logger.warning("Redis 健康检查失败: %s", e)
            return {"status": "unhealthy", "error": str(e)}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """关闭连接池，释放资源。"""
        if self._client is not None:
            await self._client.aclose()  # type: ignore[union-attr]
            self._client = None
            logger.info("Redis 连接已关闭")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _execute_with_retry(self, command: str, *args, **kwargs):
        """带重试的 Redis 命令执行。

        网络抖动时自动重试最多 max_retries 次，全部失败则抛出异常。
        """
        assert self._client is not None, "Redis 客户端未初始化"
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                method = getattr(self._client, command)
                return await method(*args, **kwargs)
            except (
                aioredis.ConnectionError,
                aioredis.TimeoutError,
                OSError,
            ) as e:
                last_error = e
                logger.warning(
                    "Redis 命令 %s 执行失败 (尝试 %d/%d): %s",
                    command,
                    attempt,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    continue

        # 所有重试用完
        raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 模块级工厂 (单例)
# ---------------------------------------------------------------------------

_instance: Optional[RedisClient] = None


def get_redis_client(
    url: str | None = None,
    mode: str = "standalone",
    **kwargs,
) -> RedisClient:
    """获取 Redis 客户端单例。

    首次调用时创建实例，后续调用返回同一实例。

    Parameters
    ----------
    url : str | None
        Redis 连接 URL。
    mode : str
        部署模式。
    **kwargs
        传递给 RedisClient 构造函数的其他参数。

    Returns
    -------
    RedisClient
        单例实例。
    """
    global _instance
    if _instance is None:
        _instance = RedisClient(url=url, mode=mode, **kwargs)
    return _instance


def reset_redis_client() -> None:
    """重置 Redis 客户端单例（主要用于测试）。"""
    global _instance
    _instance = None
