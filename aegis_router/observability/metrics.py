"""指标收集模块

提供轻量级内存指标收集，覆盖:
- 分步骤耗时打点（脱敏、路由决策、LLM 响应、还原）
- Token 消耗统计（按 API Key / 模型维度聚合）
- 组件健康状态跟踪
- 百分位数计算（P50, P95, P99）

线程安全设计，适用于 LiteLLM 多异步任务场景。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default max samples per latency buffer (rolling window)
DEFAULT_MAX_SAMPLES = 1000

# Known latency step names
STEP_MASK = "mask"
STEP_ROUTE = "route"
STEP_LLM = "llm"
STEP_RESTORE = "restore"
STEP_TOTAL = "total"

ALL_STEPS = (STEP_MASK, STEP_ROUTE, STEP_LLM, STEP_RESTORE, STEP_TOTAL)


# ---------------------------------------------------------------------------
# Component Health
# ---------------------------------------------------------------------------


class ComponentStatus(str, Enum):
    """组件健康状态枚举。"""

    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    """某一步骤的延迟统计摘要。"""

    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


@dataclass
class TokenUsageRecord:
    """单次 Token 消耗记录。"""

    model: str
    api_key: str
    tokens_in: int
    tokens_out: int

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


# ---------------------------------------------------------------------------
# Percentile Calculation
# ---------------------------------------------------------------------------


def percentile(sorted_data: list[float], p: float) -> float:
    """计算百分位数（线性插值法）。

    Args:
        sorted_data: 已排序的浮点数列表。
        p: 百分位数，取值范围 [0, 100]。

    Returns:
        百分位数值。若列表为空则返回 0.0。
    """
    if not sorted_data:
        return 0.0

    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    # 使用排位法：rank = p/100 * (n - 1)
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    fraction = rank - lower

    if upper >= n:
        return sorted_data[-1]

    return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])


# ---------------------------------------------------------------------------
# StepTimer — 分步骤计时上下文管理器
# ---------------------------------------------------------------------------


class StepTimer:
    """分步骤计时器，支持同步和异步上下文管理器。

    Usage (async):
        async with metrics.step_timer("mask") as timer:
            result = await do_masking()
        print(timer.elapsed_ms)

    Usage (sync):
        with metrics.step_timer("mask") as timer:
            result = do_masking()
        print(timer.elapsed_ms)
    """

    def __init__(self, step: str, collector: MetricsCollector) -> None:
        self.step = step
        self._collector = collector
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> StepTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._collector.record_latency(self.step, self.elapsed_ms)

    async def __aenter__(self) -> StepTimer:
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self._collector.record_latency(self.step, self.elapsed_ms)


# ---------------------------------------------------------------------------
# MetricsCollector — 核心指标收集器
# ---------------------------------------------------------------------------


class MetricsCollector:
    """内存指标收集器（线程安全）。

    跟踪:
    - 各步骤延迟（固定大小滚动窗口）
    - 请求计数器（total / success / error）
    - Token 消耗（按 model 和 api_key 聚合）
    - 组件健康状态
    """

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self._max_samples = max_samples
        self._lock = threading.Lock()

        # Latency buffers: step_name -> list[float] (ms values)
        self._latencies: dict[str, list[float]] = defaultdict(list)

        # Request counters
        self._request_total: int = 0
        self._request_success: int = 0
        self._request_error: int = 0

        # Token usage: (model, api_key) -> {"tokens_in": int, "tokens_out": int}
        self._token_usage: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"tokens_in": 0, "tokens_out": 0}
        )

        # Component health status
        self._health: dict[str, ComponentStatus] = {
            "clawvault": ComponentStatus.UNKNOWN,
            "redis": ComponentStatus.UNKNOWN,
            "routellm": ComponentStatus.UNKNOWN,
        }

    # ------------------------------------------------------------------
    # Latency Recording
    # ------------------------------------------------------------------

    def record_latency(self, step: str, elapsed_ms: float) -> None:
        """记录某步骤的一次延迟样本。

        当样本数超过 max_samples 时，移除最早的样本（FIFO）。

        Args:
            step: 步骤名称（mask, route, llm, restore, total）。
            elapsed_ms: 延迟毫秒数。
        """
        with self._lock:
            buf = self._latencies[step]
            buf.append(elapsed_ms)
            if len(buf) > self._max_samples:
                # 移除最早的样本
                del buf[: len(buf) - self._max_samples]

    def get_latency_stats(self, step: str) -> LatencyStats:
        """获取某步骤的延迟统计摘要。

        Args:
            step: 步骤名称。

        Returns:
            LatencyStats 数据对象，包含 count, min, max, avg, p50, p95, p99。
        """
        with self._lock:
            buf = list(self._latencies.get(step, []))

        if not buf:
            return LatencyStats()

        sorted_buf = sorted(buf)
        count = len(sorted_buf)

        return LatencyStats(
            count=count,
            min_ms=round(sorted_buf[0], 2),
            max_ms=round(sorted_buf[-1], 2),
            avg_ms=round(sum(sorted_buf) / count, 2),
            p50_ms=round(percentile(sorted_buf, 50), 2),
            p95_ms=round(percentile(sorted_buf, 95), 2),
            p99_ms=round(percentile(sorted_buf, 99), 2),
        )

    def get_all_latency_stats(self) -> dict[str, LatencyStats]:
        """获取所有步骤的延迟统计。

        Returns:
            字典: step_name -> LatencyStats。
        """
        with self._lock:
            steps = list(self._latencies.keys())

        return {step: self.get_latency_stats(step) for step in steps}

    # ------------------------------------------------------------------
    # Request Counters
    # ------------------------------------------------------------------

    def increment_request(self, success: bool = True) -> None:
        """增加请求计数。

        Args:
            success: True 表示成功请求，False 表示失败请求。
        """
        with self._lock:
            self._request_total += 1
            if success:
                self._request_success += 1
            else:
                self._request_error += 1

    def get_request_counts(self) -> dict[str, int]:
        """获取请求计数。

        Returns:
            字典包含 total, success, error 计数。
        """
        with self._lock:
            return {
                "total": self._request_total,
                "success": self._request_success,
                "error": self._request_error,
            }

    # ------------------------------------------------------------------
    # Token Usage
    # ------------------------------------------------------------------

    def record_token_usage(
        self,
        model: str,
        api_key: str,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """记录一次 Token 消耗。

        Args:
            model: 模型名称。
            api_key: API Key（建议传 hash 而非明文）。
            tokens_in: 输入 Token 数。
            tokens_out: 输出 Token 数。
        """
        with self._lock:
            key = (model, api_key)
            self._token_usage[key]["tokens_in"] += tokens_in
            self._token_usage[key]["tokens_out"] += tokens_out

    def get_token_usage_by_model(self) -> dict[str, dict[str, int]]:
        """按模型聚合 Token 消耗。

        Returns:
            字典: model -> {"tokens_in": int, "tokens_out": int, "tokens_total": int}。
        """
        with self._lock:
            aggregated: dict[str, dict[str, int]] = defaultdict(
                lambda: {"tokens_in": 0, "tokens_out": 0, "tokens_total": 0}
            )
            for (model, _api_key), usage in self._token_usage.items():
                aggregated[model]["tokens_in"] += usage["tokens_in"]
                aggregated[model]["tokens_out"] += usage["tokens_out"]
                aggregated[model]["tokens_total"] += (
                    usage["tokens_in"] + usage["tokens_out"]
                )
            return dict(aggregated)

    def get_token_usage_by_api_key(self) -> dict[str, dict[str, int]]:
        """按 API Key 聚合 Token 消耗。

        Returns:
            字典: api_key -> {"tokens_in": int, "tokens_out": int, "tokens_total": int}。
        """
        with self._lock:
            aggregated: dict[str, dict[str, int]] = defaultdict(
                lambda: {"tokens_in": 0, "tokens_out": 0, "tokens_total": 0}
            )
            for (_model, api_key), usage in self._token_usage.items():
                aggregated[api_key]["tokens_in"] += usage["tokens_in"]
                aggregated[api_key]["tokens_out"] += usage["tokens_out"]
                aggregated[api_key]["tokens_total"] += (
                    usage["tokens_in"] + usage["tokens_out"]
                )
            return dict(aggregated)

    def get_token_usage_detailed(self) -> list[dict[str, any]]:
        """获取按 (model, api_key) 维度的详细 Token 消耗。

        Returns:
            列表: [{"model": str, "api_key": str, "tokens_in": int, "tokens_out": int, "tokens_total": int}]。
        """
        with self._lock:
            result = []
            for (model, api_key), usage in self._token_usage.items():
                result.append({
                    "model": model,
                    "api_key": api_key,
                    "tokens_in": usage["tokens_in"],
                    "tokens_out": usage["tokens_out"],
                    "tokens_total": usage["tokens_in"] + usage["tokens_out"],
                })
            return result

    # ------------------------------------------------------------------
    # Component Health
    # ------------------------------------------------------------------

    def set_component_status(self, component: str, status: ComponentStatus) -> None:
        """设置组件健康状态。

        Args:
            component: 组件名称（clawvault, redis, routellm）。
            status: 健康状态枚举值。
        """
        with self._lock:
            self._health[component] = status

    def get_health_status(self) -> dict[str, str]:
        """获取所有组件的健康状态。

        Returns:
            字典: component -> status_string (up/down/unknown)。
        """
        with self._lock:
            return {k: v.value for k, v in self._health.items()}

    def is_healthy(self) -> bool:
        """判断系统整体是否健康（所有组件均 up）。

        Returns:
            True 当所有组件状态为 UP 时。
        """
        with self._lock:
            return all(v == ComponentStatus.UP for v in self._health.values())

    # ------------------------------------------------------------------
    # Timer Factory
    # ------------------------------------------------------------------

    def step_timer(self, step: str) -> StepTimer:
        """创建分步骤计时器。

        Args:
            step: 步骤名称（mask, route, llm, restore, total）。

        Returns:
            StepTimer 上下文管理器实例。
        """
        return StepTimer(step=step, collector=self)

    # ------------------------------------------------------------------
    # Reset (for testing)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置所有指标（用于测试或重新初始化）。"""
        with self._lock:
            self._latencies.clear()
            self._request_total = 0
            self._request_success = 0
            self._request_error = 0
            self._token_usage.clear()
            self._health = {
                "clawvault": ComponentStatus.UNKNOWN,
                "redis": ComponentStatus.UNKNOWN,
                "routellm": ComponentStatus.UNKNOWN,
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: 全局共享的 MetricsCollector 实例
metrics_collector = MetricsCollector()
