"""降级策略管理器

提供统一的组件健康状态追踪和降级决策 API：
- ClawVault 挂掉 → bypass 脱敏，直通转发，记录 CRITICAL 告警
- Redis 不可用 + PII 检出 → 拒绝请求，返回 HTTP 503
- Redis 不可用 + 无 PII → 请求放行
- RouteLLM 推理超时 → 默认路由到 fallback_model

设计原则：
- 轻量级，无后台线程
- 懒健康检查 (调用时检查，不轮询)
- 自动恢复 (组件恢复后自动重新启用功能)
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ComponentState(str, Enum):
    """组件健康状态枚举。"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DegradationError(Exception):
    """降级策略触发的异常 — 用于拒绝请求 (HTTP 503)。"""

    def __init__(self, message: str, component: str, details: Optional[str] = None):
        self.message = message
        self.component = component
        self.details = details
        super().__init__(message)


class DegradationManager:
    """统一降级策略管理器。

    追踪 ClawVault、Redis、RouteLLM 分类器三个核心组件的健康状态，
    提供一致的降级决策接口。

    状态管理：
    - 通过 `report_*` 方法记录组件状态变化
    - 通过 `check_*` 方法查询当前降级策略
    - 自动恢复: 当组件重新可用时，状态回到 HEALTHY

    Parameters
    ----------
    redis_client : object | None
        具有 ``health_check()`` 异步方法的 Redis 客户端实例。
    fallback_model : str
        RouteLLM 超时时的默认路由模型。
    """

    def __init__(
        self,
        redis_client=None,
        fallback_model: str = "deepseek-v3",
    ) -> None:
        self._redis_client = redis_client
        self._fallback_model = fallback_model

        # Component states
        self._clawvault_state = ComponentState.UNKNOWN
        self._redis_state = ComponentState.UNKNOWN
        self._classifier_state = ComponentState.UNKNOWN

        # Timestamps for state transitions
        self._clawvault_last_change: Optional[float] = None
        self._redis_last_change: Optional[float] = None
        self._classifier_last_change: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def clawvault_state(self) -> ComponentState:
        """当前 ClawVault 健康状态。"""
        return self._clawvault_state

    @property
    def redis_state(self) -> ComponentState:
        """当前 Redis 健康状态。"""
        return self._redis_state

    @property
    def classifier_state(self) -> ComponentState:
        """当前 RouteLLM 分类器健康状态。"""
        return self._classifier_state

    @property
    def fallback_model(self) -> str:
        """RouteLLM 超时时的兜底模型。"""
        return self._fallback_model

    # ------------------------------------------------------------------
    # State Reporting — ClawVault
    # ------------------------------------------------------------------

    def report_clawvault_healthy(self) -> None:
        """报告 ClawVault 恢复正常。"""
        if self._clawvault_state != ComponentState.HEALTHY:
            previous = self._clawvault_state
            self._clawvault_state = ComponentState.HEALTHY
            self._clawvault_last_change = time.time()
            logger.critical(
                "ClawVault 恢复正常 (previous_state=%s) — 脱敏功能已重新启用",
                previous.value,
            )

    def report_clawvault_unhealthy(self) -> None:
        """报告 ClawVault 不可用。"""
        if self._clawvault_state != ComponentState.UNHEALTHY:
            self._clawvault_state = ComponentState.UNHEALTHY
            self._clawvault_last_change = time.time()
            logger.critical(
                "ClawVault 不可用 — 进入降级模式: bypass 脱敏，直通转发"
            )

    # ------------------------------------------------------------------
    # State Reporting — Redis
    # ------------------------------------------------------------------

    def report_redis_healthy(self) -> None:
        """报告 Redis 恢复正常。"""
        if self._redis_state != ComponentState.HEALTHY:
            previous = self._redis_state
            self._redis_state = ComponentState.HEALTHY
            self._redis_last_change = time.time()
            logger.critical(
                "Redis 恢复正常 (previous_state=%s) — PII 映射存储已恢复",
                previous.value,
            )

    def report_redis_unhealthy(self) -> None:
        """报告 Redis 不可用。"""
        if self._redis_state != ComponentState.UNHEALTHY:
            self._redis_state = ComponentState.UNHEALTHY
            self._redis_last_change = time.time()
            logger.critical(
                "Redis 不可用 — 需脱敏的请求将被拒绝 (HTTP 503)"
            )

    # ------------------------------------------------------------------
    # State Reporting — Classifier
    # ------------------------------------------------------------------

    def report_classifier_healthy(self) -> None:
        """报告 RouteLLM 分类器恢复正常。"""
        if self._classifier_state != ComponentState.HEALTHY:
            previous = self._classifier_state
            self._classifier_state = ComponentState.HEALTHY
            self._classifier_last_change = time.time()
            logger.critical(
                "RouteLLM 分类器恢复正常 (previous_state=%s)",
                previous.value,
            )

    def report_classifier_unhealthy(self) -> None:
        """报告 RouteLLM 分类器不可用/超时。"""
        if self._classifier_state != ComponentState.UNHEALTHY:
            self._classifier_state = ComponentState.UNHEALTHY
            self._classifier_last_change = time.time()
            logger.critical(
                "RouteLLM 分类器不可用 — 默认路由到 %s",
                self._fallback_model,
            )

    # ------------------------------------------------------------------
    # Degradation Decisions
    # ------------------------------------------------------------------

    async def check_redis_health(self) -> ComponentState:
        """懒检查 Redis 健康状态。

        调用 RedisClient.health_check() 并更新内部状态。

        Returns
        -------
        ComponentState
            当前 Redis 健康状态。
        """
        if self._redis_client is None:
            # 无 Redis 客户端配置时，视为健康 (不做映射存储)
            return ComponentState.HEALTHY

        try:
            result = await self._redis_client.health_check()
            if result.get("status") == "healthy":
                self.report_redis_healthy()
                return ComponentState.HEALTHY
            else:
                self.report_redis_unhealthy()
                return ComponentState.UNHEALTHY
        except Exception as e:
            logger.warning("Redis 健康检查异常: %s", e)
            self.report_redis_unhealthy()
            return ComponentState.UNHEALTHY

    def should_reject_for_redis(self, pii_detected: bool) -> bool:
        """判断是否因 Redis 不可用而拒绝请求。

        策略:
        - Redis 不可用 AND PII 已检测到 → 拒绝 (True)
        - Redis 不可用 AND 无 PII → 放行 (False)
        - Redis 正常 → 放行 (False)

        Parameters
        ----------
        pii_detected : bool
            当前请求是否检测到 PII 实体。

        Returns
        -------
        bool
            True 表示应拒绝请求。
        """
        if self._redis_state == ComponentState.UNHEALTHY and pii_detected:
            return True
        return False

    def enforce_redis_policy(self, pii_detected: bool, request_id: str = "") -> None:
        """执行 Redis 降级策略。

        如果 should_reject_for_redis 返回 True，抛出 DegradationError。

        Parameters
        ----------
        pii_detected : bool
            当前请求是否检测到 PII。
        request_id : str
            请求 ID（用于日志）。

        Raises
        ------
        DegradationError
            当 Redis 不可用且检测到 PII 时抛出。
        """
        if self.should_reject_for_redis(pii_detected):
            logger.critical(
                "拒绝请求 (Redis 不可用 + PII 检出): request_id=%s",
                request_id,
            )
            raise DegradationError(
                message="Service temporarily unavailable: PII storage backend is down",
                component="redis",
                details=f"Redis unhealthy, PII detected in request {request_id}",
            )

    # ------------------------------------------------------------------
    # Status Summary
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """获取所有组件的健康状态摘要。

        Returns
        -------
        dict
            包含各组件状态和最后变更时间的字典。
        """
        return {
            "clawvault": {
                "state": self._clawvault_state.value,
                "last_change": self._clawvault_last_change,
            },
            "redis": {
                "state": self._redis_state.value,
                "last_change": self._redis_last_change,
            },
            "classifier": {
                "state": self._classifier_state.value,
                "last_change": self._classifier_last_change,
            },
        }
