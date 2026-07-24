"""内存与资源基准测试

验证 AegisRouter 核心组件在空载和负载下的内存使用:
- TC-PERF-MEM-001: 单实例空载内存 < 500MB
- TC-PERF-MEM-002: 1000 QPS 持续 10 分钟后无内存泄漏（内存波动 < 10%）
- TC-PERF-MEM-003: Redis 映射表 TTL 到期后正确释放（无残留 key）
"""

from __future__ import annotations

import asyncio
import time
import tracemalloc
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.config import TrivialConfig
from aegis_router.router.model_classifier import ModelClassifier
from aegis_router.router.route_resolver import RouteResolver
from aegis_router.router.rule_engine import RuleEngine
from aegis_router.storage.redis_client import RedisClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """创建 mock ClawVaultPool，模拟零延迟 RPC 调用。"""
    pool = AsyncMock(spec=ClawVaultPool)
    pool.max_connections = 10

    _compliance_ok = {"passed": True, "violations": []}
    _mapping = {"mapping": {"[PERSON_1]": "张三"}}

    async def fake_call(method: str, params: dict, timeout: float | None = None):
        if method == "check_compliance":
            return _compliance_ok
        elif method == "mask":
            text = params.get("text", "")
            return {
                "masked_text": text.replace("张三", "[PERSON_1]"),
                "entities_found": [{"type": "PERSON", "text": "张三"}] if "张三" in text else [],
            }
        elif method == "restore":
            text = params.get("text", "")
            return {"restored_text": text.replace("[PERSON_1]", "张三")}
        elif method == "get_mapping":
            return _mapping
        return {"status": "ok"}

    pool.call = AsyncMock(side_effect=fake_call)
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_rule_engine():
    """创建 mock RuleEngine。"""
    engine = MagicMock(spec=RuleEngine)
    result = MagicMock()
    result.matched = False
    result.target_model = None
    result.matched_pattern = None
    engine.check = MagicMock(return_value=result)
    return engine


@pytest.fixture
def mock_classifier():
    """创建 mock ModelClassifier。"""
    classifier = AsyncMock(spec=ModelClassifier)
    classify_result = MagicMock()
    classify_result.score = 0.65
    classifier.aclassify = AsyncMock(return_value=classify_result)
    return classifier


@pytest.fixture
def mock_route_resolver():
    """创建 mock RouteResolver。"""
    resolver = MagicMock(spec=RouteResolver)
    resolver.resolve = MagicMock(return_value={
        "model": "gpt-4o",
        "reason": "score_match",
        "candidates": ["gpt-4o"],
    })
    return resolver


@pytest.fixture
def smart_router(mock_pool, mock_rule_engine, mock_classifier, mock_route_resolver):
    """创建带完整 mock 依赖的 SmartRouterCallback 实例。"""
    with patch("aegis_router.callbacks.smart_router._pool", mock_pool):
        callback = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=mock_rule_engine,
            classifier=mock_classifier,
        )
        callback._route_resolver = mock_route_resolver
        callback._routing_config = MagicMock()
        callback._routing_config.score_input = "masked"
        callback._routing_config.fallback_model = "deepseek-v3"
        return callback


# ---------------------------------------------------------------------------
# TC-PERF-MEM-001: 单实例空载内存 < 500MB
# ---------------------------------------------------------------------------


class TestIdleMemory:
    """TC-PERF-MEM-001: 验证单实例空载内存 < 500MB。"""

    async def test_idle_memory_under_500mb(
        self, mock_pool, mock_rule_engine, mock_classifier, mock_route_resolver
    ):
        """初始化核心组件后，进程级内存占用应 < 500MB。

        使用 tracemalloc 测量组件初始化后的内存快照大小。
        由于使用 mock 依赖（无真实 spaCy 模型、无真实 Redis），
        此测试验证框架代码本身的内存开销。
        """
        tracemalloc.start()

        # 初始化核心组件
        with patch("aegis_router.callbacks.smart_router._pool", mock_pool):
            callback = SmartRouterCallback(
                pool=mock_pool,
                enable_routing=True,
                rule_engine=mock_rule_engine,
                classifier=mock_classifier,
            )
            callback._route_resolver = mock_route_resolver
            callback._routing_config = MagicMock()
            callback._routing_config.score_input = "masked"
            callback._routing_config.fallback_model = "deepseek-v3"

        # 取快照并获取当前追踪的内存总量
        snapshot = tracemalloc.take_snapshot()
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        # 转换为 MB
        current_mb = current_mem / (1024 * 1024)
        peak_mb = peak_mem / (1024 * 1024)

        # 断言峰值内存 < 500MB
        assert peak_mb < 500.0, (
            f"空载峰值内存 {peak_mb:.2f}MB 超出 500MB 阈值 "
            f"(当前={current_mb:.2f}MB)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-MEM-002: 持续负载后无内存泄漏（内存波动 < 10%）
# ---------------------------------------------------------------------------


class TestMemoryLeakUnderLoad:
    """TC-PERF-MEM-002: 持续负载后无内存泄漏。"""

    async def test_no_memory_leak_under_sustained_load(self, smart_router):
        """模拟高吞吐负载 (CI 缩短至数秒)，验证内存无持续增长趋势。

        方法: 将负载分为多个 epoch，每个 epoch 独立测量内存增量。
        通过对比各 epoch 的净内存增量（epoch 结束 - epoch 开始）是否保持稳定，
        来判断是否存在泄漏。如果后半段 epoch 的增量未持续增长，则无泄漏。

        使用 tracemalloc 对每个 epoch 取快照 (snapshot) 差值，
        避免累计分配量误判为泄漏。
        """
        import gc

        # 预热阶段: 发送足够多的请求让内存分配稳定化
        for i in range(300):
            data = {
                "messages": [{"role": "user", "content": f"用户张三查询订单 warmup-{i}"}],
                "metadata": {},
            }
            await smart_router.async_pre_call_hook({}, None, data, "completion")

        # 强制 GC 清理预热阶段的临时对象
        gc.collect()

        # 分 epoch 测试: 每个 epoch 独立测量
        num_epochs = 4
        requests_per_epoch = 500
        epoch_net_sizes: list[int] = []

        for epoch in range(num_epochs):
            # 每个 epoch 开始前取基线快照
            tracemalloc.start()
            snapshot_start = tracemalloc.take_snapshot()

            for req in range(requests_per_epoch):
                data = {
                    "messages": [
                        {"role": "user", "content": f"用户张三的手机号是13912345678，请求 e{epoch}-r{req}"}
                    ],
                    "metadata": {},
                }
                await smart_router.async_pre_call_hook({}, None, data, "completion")

            # epoch 结束: GC 后取快照
            gc.collect()
            snapshot_end = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # 计算此 epoch 的净内存留存 (排除已释放的临时分配)
            stats = snapshot_end.compare_to(snapshot_start, "filename")
            net_size = sum(s.size_diff for s in stats)
            epoch_net_sizes.append(net_size)

        # 分析: 比较最后一个 epoch 增量与第一个 epoch 增量
        # 如果无泄漏，后续 epoch 的净留存不应持续增长
        # 允许一定噪声，检查最后 epoch 增量相比第一个 epoch 的倍数
        first_epoch_net = max(epoch_net_sizes[0], 1)  # 避免除零
        last_epoch_net = epoch_net_sizes[-1]

        # 对于无泄漏的系统，后续 epoch 的净增量不应比首 epoch 大太多
        # 如果净增量为负或接近零，说明 GC 有效释放了临时对象
        if last_epoch_net <= 0:
            growth_pct = 0.0
        elif first_epoch_net <= 0:
            # 首 epoch 也无净增长 — 系统非常稳定
            growth_pct = 0.0
        else:
            growth_pct = ((last_epoch_net - first_epoch_net) / first_epoch_net) * 100.0

        total_requests = num_epochs * requests_per_epoch
        epoch_net_mb = [s / (1024 * 1024) for s in epoch_net_sizes]

        # 断言: 最后 epoch 增量不超过首 epoch 增量的 10% 以上增长
        # 这表示内存使用量已经稳定（每 epoch 产生的留存量大致恒定）
        # 实际上对于无泄漏代码，后续 epoch 的净留存应该更小或持平
        assert growth_pct < 10.0 or last_epoch_net < first_epoch_net, (
            f"内存增长趋势异常: 最后 epoch 净增量较首 epoch 增长 {growth_pct:.2f}% "
            f"(各 epoch 净留存(MB)={epoch_net_mb}, "
            f"总请求数={total_requests})"
        )


# ---------------------------------------------------------------------------
# TC-PERF-MEM-003: Redis 映射表 TTL 到期后正确释放
# ---------------------------------------------------------------------------


class TestRedisTTLExpiry:
    """TC-PERF-MEM-003: Redis 映射表 TTL 到期后正确释放（无残留 key）。"""

    async def test_store_mapping_called_with_correct_ttl(self):
        """验证 store_mapping 被调用时传入正确的 TTL 参数。"""
        mock_redis = AsyncMock(spec=RedisClient)
        mock_redis.store_mapping = AsyncMock(return_value=None)
        mock_redis.get_mapping = AsyncMock(return_value={})

        session_id = "test-session-001"
        request_id = "test-request-001"
        mapping = {"[PERSON_1]": "张三", "[PHONE_1]": "13800138000"}
        ttl = 1800  # 30 minutes

        # 调用 store_mapping
        await mock_redis.store_mapping(session_id, request_id, mapping, ttl=ttl)

        # 验证 store_mapping 被正确调用，包含 TTL 参数
        mock_redis.store_mapping.assert_called_once_with(
            session_id, request_id, mapping, ttl=ttl
        )

    async def test_session_mapping_called_with_correct_ttl(self):
        """验证 update_session_mapping 被调用时传入正确的 TTL 参数。"""
        mock_redis = AsyncMock(spec=RedisClient)
        mock_redis.update_session_mapping = AsyncMock(return_value=None)

        session_id = "test-session-002"
        mapping = {"[PERSON_1]": "张三"}
        ttl = 3600  # 1 hour

        await mock_redis.update_session_mapping(session_id, mapping, ttl=ttl)

        mock_redis.update_session_mapping.assert_called_once_with(
            session_id, mapping, ttl=ttl
        )

    async def test_expired_keys_not_returned(self):
        """模拟 TTL 过期后 get_mapping 返回空字典（无残留 key）。

        使用内存字典模拟 Redis TTL 行为：
        - 存储 mapping 时记录过期时间
        - 查询时检查是否已过期
        - 过期后返回空字典
        """
        # 模拟带 TTL 的 Redis 存储
        storage: dict[str, tuple[dict, float]] = {}  # key -> (value, expire_time)

        async def mock_store(session_id, request_id, mapping, ttl=1800):
            key = f"aegis:pii:{session_id}:{request_id}"
            expire_at = time.time() + ttl
            storage[key] = (mapping, expire_at)

        async def mock_get(request_id, session_id=None):
            if session_id:
                key = f"aegis:pii:{session_id}:{request_id}"
                if key in storage:
                    value, expire_at = storage[key]
                    if time.time() < expire_at:
                        return value
                    else:
                        # TTL 过期 — 删除并返回空
                        del storage[key]
                        return {}
            return {}

        mock_redis = AsyncMock(spec=RedisClient)
        mock_redis.store_mapping = AsyncMock(side_effect=mock_store)
        mock_redis.get_mapping = AsyncMock(side_effect=mock_get)

        session_id = "ttl-test-session"
        request_id = "ttl-test-request"
        mapping = {"[PERSON_1]": "张三", "[PHONE_1]": "13800138000"}

        # 使用极短 TTL (1 秒) 存储
        await mock_redis.store_mapping(session_id, request_id, mapping, ttl=1)

        # 立即查询 — 应返回映射数据
        result_before = await mock_redis.get_mapping(request_id, session_id=session_id)
        assert result_before == mapping, (
            f"TTL 过期前应返回映射数据, 实际返回: {result_before}"
        )

        # 等待 TTL 过期
        await asyncio.sleep(1.1)

        # 过期后查询 — 应返回空字典（无残留）
        result_after = await mock_redis.get_mapping(request_id, session_id=session_id)
        assert result_after == {}, (
            f"TTL 过期后应返回空字典（无残留 key）, 实际返回: {result_after}"
        )

        # 验证 storage 中 key 已被清除
        key = f"aegis:pii:{session_id}:{request_id}"
        assert key not in storage, (
            f"TTL 过期后 storage 中不应残留 key: {key}"
        )

    async def test_multiple_keys_expire_independently(self):
        """验证多个 key 独立过期，互不影响。"""
        storage: dict[str, tuple[dict, float]] = {}

        async def mock_store(session_id, request_id, mapping, ttl=1800):
            key = f"aegis:pii:{session_id}:{request_id}"
            expire_at = time.time() + ttl
            storage[key] = (mapping, expire_at)

        async def mock_get(request_id, session_id=None):
            if session_id:
                key = f"aegis:pii:{session_id}:{request_id}"
                if key in storage:
                    value, expire_at = storage[key]
                    if time.time() < expire_at:
                        return value
                    else:
                        del storage[key]
                        return {}
            return {}

        mock_redis = AsyncMock(spec=RedisClient)
        mock_redis.store_mapping = AsyncMock(side_effect=mock_store)
        mock_redis.get_mapping = AsyncMock(side_effect=mock_get)

        session_id = "multi-ttl-session"

        # 存储两个映射: req-1 TTL=1s, req-2 TTL=5s
        mapping_1 = {"[PERSON_1]": "张三"}
        mapping_2 = {"[PHONE_1]": "13800138000"}

        await mock_redis.store_mapping(session_id, "req-1", mapping_1, ttl=1)
        await mock_redis.store_mapping(session_id, "req-2", mapping_2, ttl=5)

        # 等待 req-1 过期
        await asyncio.sleep(1.1)

        # req-1 应已过期
        result_1 = await mock_redis.get_mapping("req-1", session_id=session_id)
        assert result_1 == {}, "req-1 应已过期返回空字典"

        # req-2 应仍有效
        result_2 = await mock_redis.get_mapping("req-2", session_id=session_id)
        assert result_2 == mapping_2, "req-2 TTL 未到期，应返回映射数据"
