"""Unit tests for MetricsCollector — 分步骤耗时打点与指标收集。

覆盖:
- StepTimer 计时精度
- MetricsCollector 延迟记录与统计
- 百分位数计算 (P50, P95, P99)
- Token 消耗记录与聚合
- 组件健康状态跟踪
- 线程安全
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from aegis_router.observability.metrics import (
    ComponentStatus,
    LatencyStats,
    MetricsCollector,
    StepTimer,
    metrics_collector,
    percentile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def collector() -> MetricsCollector:
    """Provide a fresh MetricsCollector per test."""
    return MetricsCollector(max_samples=100)


# ---------------------------------------------------------------------------
# Tests: percentile function
# ---------------------------------------------------------------------------


class TestPercentile:
    """Verify percentile calculation."""

    def test_empty_list(self):
        """空列表返回 0.0。"""
        assert percentile([], 50) == 0.0

    def test_single_element(self):
        """单元素列表任何百分位返回该元素。"""
        assert percentile([5.0], 50) == 5.0
        assert percentile([5.0], 99) == 5.0

    def test_p50_even_count(self):
        """P50 on even-count sorted list uses interpolation."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = percentile(data, 50)
        assert result == pytest.approx(2.5)

    def test_p50_odd_count(self):
        """P50 on odd-count sorted list returns middle value."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = percentile(data, 50)
        assert result == pytest.approx(3.0)

    def test_p95(self):
        """P95 on a 100-element list returns the 95th percentile."""
        data = [float(i) for i in range(1, 101)]  # 1.0 ... 100.0
        result = percentile(data, 95)
        # rank = 0.95 * 99 = 94.05 → between index 94 (val 95) and 95 (val 96)
        assert result == pytest.approx(95.05, abs=0.01)

    def test_p99(self):
        """P99 on a 100-element list."""
        data = [float(i) for i in range(1, 101)]
        result = percentile(data, 99)
        # rank = 0.99 * 99 = 98.01 → between index 98 (val 99) and 99 (val 100)
        assert result == pytest.approx(99.01, abs=0.01)

    def test_p0_returns_minimum(self):
        """P0 returns the minimum value."""
        data = [10.0, 20.0, 30.0]
        assert percentile(data, 0) == 10.0

    def test_p100_returns_maximum(self):
        """P100 returns the maximum value."""
        data = [10.0, 20.0, 30.0]
        assert percentile(data, 100) == 30.0


# ---------------------------------------------------------------------------
# Tests: StepTimer
# ---------------------------------------------------------------------------


class TestStepTimer:
    """Verify StepTimer records correct elapsed time."""

    def test_sync_context_manager(self, collector: MetricsCollector):
        """Sync with-block records elapsed_ms."""
        with collector.step_timer("mask") as timer:
            time.sleep(0.01)  # ~10ms

        assert timer.elapsed_ms >= 9.0  # at least 9ms (allow clock jitter)
        assert timer.elapsed_ms < 100.0  # less than 100ms

    def test_sync_records_to_collector(self, collector: MetricsCollector):
        """StepTimer records the latency to the collector."""
        with collector.step_timer("route") as timer:
            time.sleep(0.005)

        stats = collector.get_latency_stats("route")
        assert stats.count == 1
        assert stats.p50_ms >= 4.0

    @pytest.mark.asyncio
    async def test_async_context_manager(self, collector: MetricsCollector):
        """Async with-block records elapsed_ms."""
        async with collector.step_timer("llm") as timer:
            await asyncio.sleep(0.01)

        assert timer.elapsed_ms >= 9.0
        assert timer.elapsed_ms < 100.0

    @pytest.mark.asyncio
    async def test_async_records_to_collector(self, collector: MetricsCollector):
        """Async StepTimer records the latency to the collector."""
        async with collector.step_timer("restore") as timer:
            await asyncio.sleep(0.005)

        stats = collector.get_latency_stats("restore")
        assert stats.count == 1
        assert stats.min_ms >= 4.0

    def test_step_name_preserved(self, collector: MetricsCollector):
        """Timer preserves the step name."""
        timer = collector.step_timer("total")
        assert timer.step == "total"


# ---------------------------------------------------------------------------
# Tests: MetricsCollector — Latency
# ---------------------------------------------------------------------------


class TestMetricsCollectorLatency:
    """Verify latency recording and statistics."""

    def test_record_and_retrieve(self, collector: MetricsCollector):
        """Record latency samples and retrieve stats."""
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            collector.record_latency("mask", v)

        stats = collector.get_latency_stats("mask")
        assert stats.count == 5
        assert stats.min_ms == 10.0
        assert stats.max_ms == 50.0
        assert stats.avg_ms == 30.0

    def test_p50_p95_p99(self, collector: MetricsCollector):
        """Percentiles calculated correctly from recorded samples."""
        # Record 100 values: 1.0 to 100.0
        for i in range(1, 101):
            collector.record_latency("llm", float(i))

        stats = collector.get_latency_stats("llm")
        assert stats.count == 100
        assert stats.p50_ms == pytest.approx(50.5, abs=0.1)
        assert stats.p95_ms == pytest.approx(95.05, abs=0.1)
        assert stats.p99_ms == pytest.approx(99.01, abs=0.1)

    def test_empty_step_returns_zero_stats(self, collector: MetricsCollector):
        """Querying a step with no data returns zero stats."""
        stats = collector.get_latency_stats("nonexistent")
        assert stats.count == 0
        assert stats.min_ms == 0.0
        assert stats.p50_ms == 0.0

    def test_rolling_window(self):
        """Samples exceeding max_samples are evicted (FIFO)."""
        collector = MetricsCollector(max_samples=5)
        for i in range(10):
            collector.record_latency("mask", float(i))

        stats = collector.get_latency_stats("mask")
        assert stats.count == 5
        # Should have values 5, 6, 7, 8, 9 (oldest evicted)
        assert stats.min_ms == 5.0
        assert stats.max_ms == 9.0

    def test_get_all_latency_stats(self, collector: MetricsCollector):
        """get_all_latency_stats returns stats for all recorded steps."""
        collector.record_latency("mask", 10.0)
        collector.record_latency("route", 5.0)
        collector.record_latency("llm", 100.0)

        all_stats = collector.get_all_latency_stats()
        assert "mask" in all_stats
        assert "route" in all_stats
        assert "llm" in all_stats
        assert all_stats["mask"].count == 1
        assert all_stats["route"].min_ms == 5.0


# ---------------------------------------------------------------------------
# Tests: MetricsCollector — Request Counters
# ---------------------------------------------------------------------------


class TestMetricsCollectorCounters:
    """Verify request counter functionality."""

    def test_increment_success(self, collector: MetricsCollector):
        """Success increments both total and success counters."""
        collector.increment_request(success=True)
        collector.increment_request(success=True)

        counts = collector.get_request_counts()
        assert counts["total"] == 2
        assert counts["success"] == 2
        assert counts["error"] == 0

    def test_increment_error(self, collector: MetricsCollector):
        """Error increments total and error counters."""
        collector.increment_request(success=False)

        counts = collector.get_request_counts()
        assert counts["total"] == 1
        assert counts["success"] == 0
        assert counts["error"] == 1

    def test_mixed_counts(self, collector: MetricsCollector):
        """Mixed success/error counts."""
        for _ in range(7):
            collector.increment_request(success=True)
        for _ in range(3):
            collector.increment_request(success=False)

        counts = collector.get_request_counts()
        assert counts["total"] == 10
        assert counts["success"] == 7
        assert counts["error"] == 3


# ---------------------------------------------------------------------------
# Tests: MetricsCollector — Token Usage
# ---------------------------------------------------------------------------


class TestMetricsCollectorTokenUsage:
    """Verify token usage recording and aggregation."""

    def test_record_and_aggregate_by_model(self, collector: MetricsCollector):
        """Token usage aggregated by model."""
        collector.record_token_usage("gpt-4o", "key-a", 100, 50)
        collector.record_token_usage("gpt-4o", "key-b", 200, 100)
        collector.record_token_usage("deepseek-v3", "key-a", 50, 25)

        by_model = collector.get_token_usage_by_model()
        assert by_model["gpt-4o"]["tokens_in"] == 300
        assert by_model["gpt-4o"]["tokens_out"] == 150
        assert by_model["gpt-4o"]["tokens_total"] == 450
        assert by_model["deepseek-v3"]["tokens_total"] == 75

    def test_record_and_aggregate_by_api_key(self, collector: MetricsCollector):
        """Token usage aggregated by API key."""
        collector.record_token_usage("gpt-4o", "key-a", 100, 50)
        collector.record_token_usage("deepseek-v3", "key-a", 200, 100)
        collector.record_token_usage("gpt-4o", "key-b", 50, 25)

        by_key = collector.get_token_usage_by_api_key()
        assert by_key["key-a"]["tokens_in"] == 300
        assert by_key["key-a"]["tokens_out"] == 150
        assert by_key["key-a"]["tokens_total"] == 450
        assert by_key["key-b"]["tokens_total"] == 75

    def test_detailed_usage(self, collector: MetricsCollector):
        """Detailed token usage returns per (model, api_key) breakdown."""
        collector.record_token_usage("gpt-4o", "key-a", 100, 50)
        collector.record_token_usage("gpt-4o", "key-a", 50, 25)

        detailed = collector.get_token_usage_detailed()
        assert len(detailed) == 1
        entry = detailed[0]
        assert entry["model"] == "gpt-4o"
        assert entry["api_key"] == "key-a"
        assert entry["tokens_in"] == 150
        assert entry["tokens_out"] == 75
        assert entry["tokens_total"] == 225

    def test_no_usage_returns_empty(self, collector: MetricsCollector):
        """No token usage recorded returns empty aggregations."""
        assert collector.get_token_usage_by_model() == {}
        assert collector.get_token_usage_by_api_key() == {}
        assert collector.get_token_usage_detailed() == []


# ---------------------------------------------------------------------------
# Tests: MetricsCollector — Health Status
# ---------------------------------------------------------------------------


class TestMetricsCollectorHealth:
    """Verify component health tracking."""

    def test_initial_status_is_unknown(self, collector: MetricsCollector):
        """All components start in UNKNOWN state."""
        health = collector.get_health_status()
        assert health["clawvault"] == "unknown"
        assert health["redis"] == "unknown"
        assert health["routellm"] == "unknown"

    def test_set_and_get_status(self, collector: MetricsCollector):
        """Setting component status updates correctly."""
        collector.set_component_status("clawvault", ComponentStatus.UP)
        collector.set_component_status("redis", ComponentStatus.UP)
        collector.set_component_status("routellm", ComponentStatus.DOWN)

        health = collector.get_health_status()
        assert health["clawvault"] == "up"
        assert health["redis"] == "up"
        assert health["routellm"] == "down"

    def test_is_healthy_all_up(self, collector: MetricsCollector):
        """is_healthy returns True when all components are UP."""
        collector.set_component_status("clawvault", ComponentStatus.UP)
        collector.set_component_status("redis", ComponentStatus.UP)
        collector.set_component_status("routellm", ComponentStatus.UP)
        assert collector.is_healthy() is True

    def test_is_healthy_one_down(self, collector: MetricsCollector):
        """is_healthy returns False when any component is not UP."""
        collector.set_component_status("clawvault", ComponentStatus.UP)
        collector.set_component_status("redis", ComponentStatus.DOWN)
        collector.set_component_status("routellm", ComponentStatus.UP)
        assert collector.is_healthy() is False

    def test_is_healthy_unknown(self, collector: MetricsCollector):
        """is_healthy returns False when any component is UNKNOWN."""
        assert collector.is_healthy() is False

    def test_custom_component(self, collector: MetricsCollector):
        """Can track additional custom components."""
        collector.set_component_status("custom-service", ComponentStatus.UP)
        health = collector.get_health_status()
        assert health["custom-service"] == "up"


# ---------------------------------------------------------------------------
# Tests: MetricsCollector — Thread Safety
# ---------------------------------------------------------------------------


class TestMetricsCollectorThreadSafety:
    """Verify thread-safe counter increments."""

    def test_concurrent_counter_increments(self):
        """Counter remains consistent under concurrent access."""
        collector = MetricsCollector()
        num_threads = 10
        increments_per_thread = 100

        def worker():
            for _ in range(increments_per_thread):
                collector.increment_request(success=True)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counts = collector.get_request_counts()
        assert counts["total"] == num_threads * increments_per_thread
        assert counts["success"] == num_threads * increments_per_thread

    def test_concurrent_latency_recording(self):
        """Latency buffer remains consistent under concurrent access."""
        collector = MetricsCollector(max_samples=1000)
        num_threads = 10
        records_per_thread = 50

        def worker(thread_id: int):
            for i in range(records_per_thread):
                collector.record_latency("mask", float(thread_id * 100 + i))

        threads = [
            threading.Thread(target=worker, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = collector.get_latency_stats("mask")
        assert stats.count == num_threads * records_per_thread

    def test_concurrent_token_usage(self):
        """Token usage aggregation is consistent under concurrent access."""
        collector = MetricsCollector()
        num_threads = 10
        records_per_thread = 50

        def worker():
            for _ in range(records_per_thread):
                collector.record_token_usage("gpt-4o", "key-a", 10, 5)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        by_model = collector.get_token_usage_by_model()
        expected_in = num_threads * records_per_thread * 10
        expected_out = num_threads * records_per_thread * 5
        assert by_model["gpt-4o"]["tokens_in"] == expected_in
        assert by_model["gpt-4o"]["tokens_out"] == expected_out


# ---------------------------------------------------------------------------
# Tests: MetricsCollector — Reset
# ---------------------------------------------------------------------------


class TestMetricsCollectorReset:
    """Verify reset clears all state."""

    def test_reset_clears_everything(self, collector: MetricsCollector):
        """reset() clears latencies, counters, tokens, and resets health."""
        collector.record_latency("mask", 10.0)
        collector.increment_request(success=True)
        collector.record_token_usage("gpt-4o", "key-a", 100, 50)
        collector.set_component_status("redis", ComponentStatus.UP)

        collector.reset()

        assert collector.get_latency_stats("mask").count == 0
        assert collector.get_request_counts()["total"] == 0
        assert collector.get_token_usage_by_model() == {}
        assert collector.get_health_status()["redis"] == "unknown"


# ---------------------------------------------------------------------------
# Tests: Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    """Verify the module-level metrics_collector instance."""

    def test_singleton_is_metrics_collector(self):
        """metrics_collector is a MetricsCollector instance."""
        assert isinstance(metrics_collector, MetricsCollector)

    def test_singleton_is_functional(self):
        """Module-level collector can record and retrieve data."""
        metrics_collector.reset()
        metrics_collector.record_latency("test_step", 42.0)
        stats = metrics_collector.get_latency_stats("test_step")
        assert stats.count == 1
        assert stats.p50_ms == 42.0
        metrics_collector.reset()
