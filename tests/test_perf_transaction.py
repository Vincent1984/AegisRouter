"""事务路由性能基准测试

验证 Transaction Router 核心组件的性能指标:
- TC-PERF-TXN-001: 方案生成延迟 < 5ms（10 模板 × 5 Agent）
- TC-PERF-TXN-002: 请求分发延迟 < 0.1ms（HashMap lookup）
- TC-PERF-TXN-003: 方案内存占用 < 10KB
- TC-PERF-TXN-004: 1000 QPS 并发下分发无锁竞争、零错误
"""

from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from unittest.mock import patch

import pytest

from aegis_router.router.capability_profiles import (
    CapabilityProfileManager,
    DEFAULT_PROFILES,
)
from aegis_router.router.routing_plan_store import RoutingPlanStore
from aegis_router.router.template_models import AgentDef, TemplateDef
from aegis_router.router.template_plan_generator import TemplatePlanGenerator


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
    """计算第 p 百分位数 (0-100)。"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    weight = idx - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------

# 合成模型池 (8 个模型，覆盖不同能力和成本区间)
MODELS = [
    {"name": "local-7b", "params": {"benchmark_mmlu": 65.0, "benchmark_humaneval": 45.0, "benchmark_math": 40.0, "context_window": 32000, "cost_per_1m_input": 0.0}},
    {"name": "deepseek-v4-pro", "params": {"benchmark_mmlu": 90.2, "benchmark_humaneval": 88.5, "benchmark_math": 82.0, "context_window": 128000, "cost_per_1m_input": 0.27}},
    {"name": "gpt-5.2", "params": {"benchmark_mmlu": 88.0, "benchmark_humaneval": 82.0, "benchmark_math": 78.0, "context_window": 128000, "cost_per_1m_input": 2.5}},
    {"name": "gpt-5.5", "params": {"benchmark_mmlu": 92.5, "benchmark_humaneval": 93.0, "benchmark_math": 90.0, "context_window": 200000, "cost_per_1m_input": 5.0}},
    {"name": "gpt-5.6-sol", "params": {"benchmark_mmlu": 95.0, "benchmark_humaneval": 95.5, "benchmark_math": 94.0, "context_window": 200000, "cost_per_1m_input": 15.0}},
    {"name": "codex-mini", "params": {"benchmark_mmlu": 70.0, "benchmark_humaneval": 92.0, "benchmark_math": 55.0, "context_window": 64000, "cost_per_1m_input": 1.5}},
    {"name": "gemini-2.5-flash", "params": {"benchmark_mmlu": 85.0, "benchmark_humaneval": 78.0, "benchmark_math": 72.0, "context_window": 1000000, "cost_per_1m_input": 0.15}},
    {"name": "gemini-2.5-pro", "params": {"benchmark_mmlu": 91.0, "benchmark_humaneval": 88.0, "benchmark_math": 85.0, "context_window": 2000000, "cost_per_1m_input": 2.5}},
]

# 可用的 Profile 名称列表 (从 DEFAULT_PROFILES 获取)
PROFILE_NAMES = list(DEFAULT_PROFILES.keys())

# 10 个合成模板，每个包含 5 个 Agent
TEMPLATES: dict[str, TemplateDef] = {}
for i in range(10):
    tpl_name = f"template_{i:02d}"
    agents = [
        AgentDef(
            name=f"agent_{j}",
            capability_profile=PROFILE_NAMES[j % len(PROFILE_NAMES)],
        )
        for j in range(5)
    ]
    TEMPLATES[tpl_name] = TemplateDef(
        name=tpl_name,
        description=f"合成测试模板 {i}",
        agents=agents,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def profile_manager(tmp_path):
    """创建使用 DEFAULT_PROFILES 的 CapabilityProfileManager。

    传入不存在的 config_path 让管理器回退到内置默认 Profile。
    """
    non_existent_path = tmp_path / "non_existent_profiles.yaml"
    return CapabilityProfileManager(config_path=non_existent_path)


@pytest.fixture
def plan_generator(profile_manager):
    """创建 TemplatePlanGenerator 实例。"""
    return TemplatePlanGenerator(
        profile_manager=profile_manager,
        models=MODELS,
        fallback_model="local-7b",
        trigger_reason="perf_test",
    )


@pytest.fixture
def filled_store(plan_generator):
    """预填充的 RoutingPlanStore（10 模板 × 5 Agent = 50 条目）。"""
    return plan_generator.generate_all(TEMPLATES)


# ---------------------------------------------------------------------------
# TC-PERF-TXN-001: 方案生成延迟 < 5ms（10 模板 × 5 Agent）
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPlanGenerationLatency:
    """TC-PERF-TXN-001: 方案生成延迟基准测试。"""

    def test_generate_all_p95_under_5ms(self, profile_manager):
        """generate_all() 对 10 模板 × 5 Agent 的 P95 延迟应 < 5ms。"""
        latencies: list[float] = []
        iterations = 100

        # 预热: 让 Python 的内部缓存和 JIT 优化稳定
        generator = TemplatePlanGenerator(
            profile_manager=profile_manager,
            models=MODELS,
            fallback_model="local-7b",
            trigger_reason="warmup",
        )
        for _ in range(5):
            generator.generate_all(TEMPLATES)

        # 正式测量 (suppress AuditLogger IO to measure pure computation)
        for _ in range(iterations):
            gen = TemplatePlanGenerator(
                profile_manager=profile_manager,
                models=MODELS,
                fallback_model="local-7b",
                trigger_reason="perf_bench",
            )
            with patch.object(gen._audit, "log_plan_generation_event"):
                start = time.perf_counter()
                store = gen.generate_all(TEMPLATES)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            # 验证生成结果正确
            assert len(store) == 50, f"方案条目数应为 50，实际为 {len(store)}"

        p95 = percentile(latencies, 95)
        avg = statistics.mean(latencies)
        p50 = percentile(latencies, 50)

        print(
            f"\n[TC-PERF-TXN-001] 方案生成延迟:"
            f"\n  P50: {p50:.3f}ms"
            f"\n  P95: {p95:.3f}ms"
            f"\n  平均: {avg:.3f}ms"
            f"\n  最小: {min(latencies):.3f}ms"
            f"\n  最大: {max(latencies):.3f}ms"
            f"\n  迭代次数: {iterations}"
        )

        assert p95 < 5.0, (
            f"方案生成 P95 延迟 {p95:.3f}ms 超出 5ms 阈值 "
            f"(avg={avg:.3f}ms, min={min(latencies):.3f}ms, max={max(latencies):.3f}ms)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-TXN-002: 请求分发延迟 < 0.1ms（HashMap lookup）
# ---------------------------------------------------------------------------


class TestLookupLatency:
    """TC-PERF-TXN-002: 请求分发 (HashMap lookup) 延迟基准测试。"""

    def test_get_model_p99_under_100_microseconds(self, filled_store):
        """get_model() 查表 P99 延迟应 < 0.1ms (100 微秒)。"""
        latencies: list[float] = []
        iterations = 10000

        # 构建查询 key 列表（循环使用 50 条目）
        keys = [
            (f"template_{i:02d}", f"agent_{j}")
            for i in range(10)
            for j in range(5)
        ]

        # 预热
        for k in keys[:10]:
            filled_store.get_model(k[0], k[1])

        # 正式测量
        for i in range(iterations):
            template, agent = keys[i % len(keys)]
            start = time.perf_counter()
            result = filled_store.get_model(template, agent)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            # 验证查询结果非空
            assert result is not None, (
                f"get_model('{template}', '{agent}') 返回 None"
            )

        p99 = percentile(latencies, 99)
        p95 = percentile(latencies, 95)
        avg = statistics.mean(latencies)

        print(
            f"\n[TC-PERF-TXN-002] HashMap lookup 延迟:"
            f"\n  P95: {p95:.4f}ms"
            f"\n  P99: {p99:.4f}ms"
            f"\n  平均: {avg:.4f}ms"
            f"\n  最小: {min(latencies):.4f}ms"
            f"\n  最大: {max(latencies):.4f}ms"
            f"\n  迭代次数: {iterations}"
        )

        assert p99 < 0.1, (
            f"HashMap lookup P99 延迟 {p99:.4f}ms 超出 0.1ms 阈值 "
            f"(avg={avg:.4f}ms, max={max(latencies):.4f}ms)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-TXN-003: 方案内存占用 < 10KB
# ---------------------------------------------------------------------------


class TestPlanMemoryUsage:
    """TC-PERF-TXN-003: 方案内存占用基准测试。"""

    def test_store_memory_under_10kb(self, profile_manager):
        """RoutingPlanStore (10 模板 × 5 Agent) 内存占用应 < 10KB。"""
        # 使用 tracemalloc 精确测量 RoutingPlanStore 占用
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # 创建并填充 store
        generator = TemplatePlanGenerator(
            profile_manager=profile_manager,
            models=MODELS,
            fallback_model="local-7b",
            trigger_reason="mem_test",
        )
        store = generator.generate_all(TEMPLATES)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # 计算内存差异
        stats = snapshot_after.compare_to(snapshot_before, "filename")
        total_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
        total_kb = total_bytes / 1024.0

        # 验证 store 包含预期条目
        assert len(store) == 50, f"方案条目数应为 50，实际为 {len(store)}"

        print(
            f"\n[TC-PERF-TXN-003] 方案内存占用:"
            f"\n  总内存增量: {total_kb:.2f}KB ({total_bytes} bytes)"
            f"\n  条目数: {len(store)}"
            f"\n  每条目平均: {total_bytes / 50:.1f} bytes"
        )

        assert total_kb < 10.0, (
            f"方案内存占用 {total_kb:.2f}KB 超出 10KB 阈值 "
            f"(条目数={len(store)}, 总字节={total_bytes})"
        )


# ---------------------------------------------------------------------------
# TC-PERF-TXN-004: 1000 QPS 并发下分发无锁竞争、零错误
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConcurrentDispatch:
    """TC-PERF-TXN-004: 并发分发无锁竞争、零错误基准测试。"""

    async def test_concurrent_dispatch_zero_errors(self, filled_store):
        """1000 QPS 并发下分发应无锁竞争、零错误。

        使用 12 个 worker 持续并发查询 store，模拟 1000 QPS 负载，
        验证所有查询返回正确结果。
        """
        num_workers = 12
        duration_seconds = 3  # 持续 3 秒
        errors: list[str] = []
        success_count = 0
        total_lookups = 0

        # 预计算期望结果 (作为验证基准)
        expected: dict[tuple[str, str], str] = {}
        for i in range(10):
            for j in range(5):
                tpl = f"template_{i:02d}"
                agent = f"agent_{j}"
                model = filled_store.get_model(tpl, agent)
                expected[(tpl, agent)] = model

        keys = list(expected.keys())

        async def worker(worker_id: int) -> tuple[int, int, list[str]]:
            """单个 worker 持续查询直到超时。"""
            local_success = 0
            local_errors: list[str] = []
            end_time = time.perf_counter() + duration_seconds
            idx = worker_id  # 每个 worker 从不同位置开始

            while time.perf_counter() < end_time:
                key = keys[idx % len(keys)]
                template, agent = key

                result = filled_store.get_model(template, agent)

                if result is None:
                    local_errors.append(
                        f"worker-{worker_id}: get_model('{template}', '{agent}') 返回 None"
                    )
                elif result != expected[key]:
                    local_errors.append(
                        f"worker-{worker_id}: get_model('{template}', '{agent}') "
                        f"= '{result}'，预期 '{expected[key]}'"
                    )
                else:
                    local_success += 1

                idx += 1

                # 每 100 次 yield 控制权
                if idx % 100 == 0:
                    await asyncio.sleep(0)

            return local_success, len(local_errors), local_errors

        # 启动所有 worker
        tasks = [
            asyncio.create_task(worker(i))
            for i in range(num_workers)
        ]
        results = await asyncio.gather(*tasks)

        # 汇总结果
        for s, e, errs in results:
            success_count += s
            total_lookups += s + e
            errors.extend(errs)

        error_count = len(errors)
        qps = total_lookups / duration_seconds if duration_seconds > 0 else 0

        print(
            f"\n[TC-PERF-TXN-004] 并发分发测试:"
            f"\n  Workers: {num_workers}"
            f"\n  持续时间: {duration_seconds}s"
            f"\n  总查询数: {total_lookups}"
            f"\n  成功: {success_count}"
            f"\n  错误: {error_count}"
            f"\n  QPS: {qps:.0f}"
        )

        # 验证零错误
        assert error_count == 0, (
            f"并发分发出现 {error_count} 个错误 "
            f"(总查询={total_lookups}, QPS={qps:.0f})\n"
            f"前 5 个错误: {errors[:5]}"
        )

        # 验证实际 QPS >= 1000
        assert qps >= 1000, (
            f"实际 QPS {qps:.0f} 低于 1000 阈值 "
            f"(总查询={total_lookups}, 时长={duration_seconds}s)"
        )
