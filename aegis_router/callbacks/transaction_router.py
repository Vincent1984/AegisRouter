"""事务级路由回调 (TransactionRouterCallback)

继承 BaseRouterCallback，实现事务级路由策略：
读取 metadata.transaction → 查 RoutingPlanStore → 设 data["model"]。

公共管道（合规检测、PII 脱敏、响应还原）由基类处理。

Failover 集成:
  - LLM 调用失败时，使用 failover 链中下一个模型重试
  - 仅影响当次请求，不修改全局方案表
  - 复用 route_config.yaml 中的 failover.chains 配置

设计参考: design.md TransactionRouterCallback 节
需求参考: FR-5.1 ~ FR-5.8, FR-6.1 ~ FR-6.4, FR-8.2
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aegis_router.callbacks.base_router import BaseRouterCallback
from aegis_router.callbacks.degradation import DegradationManager
from aegis_router.callbacks.exceptions import TemplateNotFoundError
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.observability.audit_logger import AuditLogger
from aegis_router.router.routing_plan_store import RoutingPlanStore

logger = logging.getLogger(__name__)


class TransactionRouterCallback(BaseRouterCallback):
    """事务级路由插件 — 查表分发，极简逻辑。

    启动时由 TemplatePlanGenerator 预计算方案表，
    分发时纯内存查表 (template, agent) → model，延迟 < 0.1ms。

    Failover 集成 (FR-6):
      - async_log_failure_event 中检查 failover 链
      - 失败时选择链中下一个模型注入 metadata
      - 仅影响当次请求，不修改全局 RoutingPlanStore

    路由策略:
      1. 读 metadata.transaction
      2. 无 transaction → fallback
      3. 有 transaction → 查表:
         - 模板不存在 → HTTP 400
         - Agent 不在模板中 → fallback + UNKNOWN_AGENT 警告
         - 命中 → 使用预计算模型
    """

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Override to ensure LiteLLM detects this method on the concrete class."""
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)

    def __init__(
        self,
        plan_store: RoutingPlanStore | None = None,
        fallback_model: str = "deepseek-v3",
        failover_chains: dict[str, list[str]] | None = None,
        failover_enabled: bool = True,
        pool: ClawVaultPool | None = None,
        degradation_manager: DegradationManager | None = None,
        config_dir: str | None = None,
    ) -> None:
        """初始化事务级路由插件。

        Args:
            plan_store: 预计算的路由方案表。为 None 时创建空表。
            fallback_model: 无方案匹配时的降级模型名称。
            failover_chains: Failover 链配置 {model → [fallback1, fallback2, ...]}。
                             复用 route_config.yaml 中 failover.chains 段。
            failover_enabled: 是否启用 failover 功能。
            pool: ClawVault 连接池（传递给基类）。
            degradation_manager: 降级管理器（传递给基类）。
            config_dir: 配置目录路径（兼容 plugin_loader 调用）。
        """
        super().__init__(
            pool=pool if pool is not None else ClawVaultPool(),
            degradation_manager=degradation_manager,
        )
        self._plan_store = plan_store if plan_store is not None else RoutingPlanStore()
        self._fallback_model = fallback_model
        self._failover_chains: dict[str, list[str]] = failover_chains or {}
        self._failover_enabled = failover_enabled
        self._config_watcher: Optional[Any] = None
        self._audit = AuditLogger()

        logger.info(
            "TransactionRouterCallback initialized "
            "(fallback_model=%s, plan_entries=%d, failover_enabled=%s, "
            "failover_chains=%d)",
            self._fallback_model,
            len(self._plan_store),
            self._failover_enabled,
            len(self._failover_chains),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def plan_store(self) -> RoutingPlanStore:
        """当前路由方案表（支持外部原子替换）。"""
        return self._plan_store

    @plan_store.setter
    def plan_store(self, new_store: RoutingPlanStore) -> None:
        """原子替换方案表（配置热更新时使用）。"""
        self._plan_store = new_store

    @property
    def fallback_model(self) -> str:
        """降级模型名称。"""
        return self._fallback_model

    @property
    def failover_chains(self) -> dict[str, list[str]]:
        """Failover 链配置（只读）。"""
        return self._failover_chains

    @property
    def failover_enabled(self) -> bool:
        """是否启用 failover。"""
        return self._failover_enabled

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """停止内部 ConfigWatcher（如果存在）。"""
        if self._config_watcher is not None:
            self._config_watcher.stop()
            self._config_watcher = None
            logger.info("TransactionRouterCallback: ConfigWatcher stopped")

    # ------------------------------------------------------------------
    # Abstract Method Implementation
    # ------------------------------------------------------------------

    async def _execute_routing(
        self,
        data: dict,
        masked_text: str,
        original_text: str,
        prompt_hash: str,
    ) -> None:
        """执行事务级路由：读 metadata → 查表 → 设 model。

        同时将 failover 链信息存入请求 metadata，供失败时使用。

        Args:
            data: LiteLLM 请求数据字典（会被修改以设置目标模型）。
            masked_text: PII 脱敏后的提示文本。
            original_text: 原始提示文本。
            prompt_hash: 原始提示的 SHA-256 哈希。
        """
        metadata = data.get("metadata") or {}
        txn = metadata.get("transaction")

        # --- 无 transaction metadata → fallback ---
        if txn is None:
            data["model"] = self._fallback_model
            metadata["target_model"] = self._fallback_model
            metadata["route_reason"] = "fallback"
            metadata["routing_plugin"] = "transaction"
            metadata["_routing_warnings"] = []
            # Audit: dispatch with fallback reason
            request_id = metadata.get("request_id", data.get("request_id", "unknown"))
            self._audit.log_dispatch_event(
                request_id=request_id,
                template="",
                agent="",
                assigned_model=self._fallback_model,
                reason="fallback",
                warnings=[],
            )
            logger.info(
                "事务路由: 无 transaction metadata, "
                "使用 fallback 模型 '%s' (prompt_hash=%s)",
                self._fallback_model,
                prompt_hash[:16],
            )
            return

        template = txn.get("template", "")
        agent = txn.get("agent", "")

        # Store routing context in metadata for observability (FR-8.2)
        metadata["transaction_template"] = template
        metadata["transaction_agent"] = agent
        metadata["routing_plugin"] = "transaction"

        # --- 检查模板是否存在 ---
        template_plan = self._plan_store.get_template_plan(template)
        if not template_plan:
            # 模板不存在 → HTTP 400
            raise TemplateNotFoundError(template)

        # --- 查表分发 ---
        model = self._plan_store.get_model(template, agent)

        if model is None:
            # Agent 不在模板中 → fallback + UNKNOWN_AGENT 警告
            logger.warning(
                "UNKNOWN_AGENT: template=%s, agent=%s — "
                "使用 fallback 模型 '%s'",
                template,
                agent,
                self._fallback_model,
            )
            data["model"] = self._fallback_model
            metadata["target_model"] = self._fallback_model
            metadata["route_reason"] = "unknown_agent"
            metadata["_routing_warnings"] = ["UNKNOWN_AGENT"]
            # Audit: dispatch with unknown reason + UNKNOWN_AGENT warning
            request_id = metadata.get("request_id", data.get("request_id", "unknown"))
            self._audit.log_dispatch_event(
                request_id=request_id,
                template=template,
                agent=agent,
                assigned_model=self._fallback_model,
                reason="unknown",
                warnings=["UNKNOWN_AGENT"],
            )
            return

        # --- 正常命中 ---
        data["model"] = model
        metadata["target_model"] = model
        metadata["route_reason"] = "plan"
        metadata["_routing_warnings"] = []

        # Audit: dispatch with plan reason
        request_id = metadata.get("request_id", data.get("request_id", "unknown"))
        self._audit.log_dispatch_event(
            request_id=request_id,
            template=template,
            agent=agent,
            assigned_model=model,
            reason="plan",
            warnings=[],
        )

        # --- 存储 failover 链信息到 metadata (FR-6.1, FR-6.2) ---
        if self._failover_enabled and model in self._failover_chains:
            chain = self._failover_chains[model]
            metadata["_failover_chain"] = list(chain)  # 副本，不修改原始配置
            metadata["_failover_index"] = 0
            metadata["_original_model"] = model

        logger.info(
            "事务路由: template=%s, agent=%s → model=%s (prompt_hash=%s)",
            template,
            agent,
            model,
            prompt_hash[:16],
        )

    # ------------------------------------------------------------------
    # Failover: async_log_failure_event (FR-6.1 ~ FR-6.4)
    # ------------------------------------------------------------------

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """异步失败回调 — 实现 failover 链重试。

        当 LLM 调用失败时:
        1. 检查是否有可用的 failover 链
        2. 有则选择下一个模型，注入到请求 metadata
        3. 记录 AGENT_FAILOVER 警告日志
        4. 不修改全局 RoutingPlanStore (FR-6.3)

        Args:
            kwargs: LiteLLM 调用参数（含 metadata）。
            response_obj: 失败响应对象。
            start_time: 调用开始时间。
            end_time: 调用结束时间。
        """
        failed_model = kwargs.get("model", "unknown")
        metadata = kwargs.get("metadata")
        if metadata is None:
            metadata = kwargs.get("litellm_params", {}).get("metadata")
        if metadata is None:
            metadata = {}

        if not self._failover_enabled:
            logger.warning(
                "async_failure_event: model=%s (failover disabled)",
                failed_model,
            )
            return

        # 获取 failover 链状态
        failover_chain = metadata.get("_failover_chain")
        failover_index = metadata.get("_failover_index", 0)
        original_model = metadata.get("_original_model", failed_model)

        # 如果没有 failover 链信息，尝试从配置中查找
        if failover_chain is None:
            if failed_model in self._failover_chains:
                failover_chain = list(self._failover_chains[failed_model])
                failover_index = 0
                original_model = failed_model
            else:
                logger.warning(
                    "async_failure_event: model=%s — 无 failover 链可用",
                    failed_model,
                )
                return

        # 检查链是否已耗尽
        if failover_index >= len(failover_chain):
            logger.error(
                "AGENT_FAILOVER_EXHAUSTED: original_model=%s — "
                "failover 链已耗尽，所有备选模型均失败 (chain=%s)",
                original_model,
                failover_chain,
            )
            return

        # 选择下一个模型
        next_model = failover_chain[failover_index]

        # 更新 metadata 中的 failover 状态（仅影响当次请求 FR-6.3）
        metadata["_failover_index"] = failover_index + 1
        metadata["_failover_model"] = next_model
        metadata["_failover_from"] = failed_model

        # 记录 AGENT_FAILOVER 警告 (FR-8.4)
        logger.warning(
            "AGENT_FAILOVER: original_model=%s, failed_model=%s → "
            "next_model=%s (chain_index=%d/%d)",
            original_model,
            failed_model,
            next_model,
            failover_index + 1,
            len(failover_chain),
        )

    # ------------------------------------------------------------------
    # Helper: Get next failover model (for external callers)
    # ------------------------------------------------------------------

    def get_next_failover_model(
        self,
        failed_model: str,
        metadata: dict | None = None,
    ) -> Optional[str]:
        """获取 failover 链中的下一个模型。

        此方法不修改全局方案表 (FR-6.3)，仅返回下一个备选模型名称。

        Args:
            failed_model: 失败的模型名称。
            metadata: 请求 metadata，包含 failover 状态。

        Returns:
            下一个备选模型名称。链耗尽时返回 None。
        """
        if not self._failover_enabled:
            return None

        if metadata:
            chain = metadata.get("_failover_chain")
            index = metadata.get("_failover_index", 0)
        else:
            chain = self._failover_chains.get(failed_model)
            index = 0

        if not chain or index >= len(chain):
            return None

        return chain[index]
