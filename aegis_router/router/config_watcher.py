"""配置热更新监听模块

使用 watchdog 监听以下配置文件的变更，自动重载配置并重建路由表:
- config/models.yaml
- config/route_config.yaml
- config/route_overrides.yaml
- config/capability_profiles.yaml
- config/transaction_templates.yaml
- config/agent_workbuddy.yaml

设计参考: design.md 2.3.6 节
需求参考: FR-2.5, FR-3.6, FR-4.2, NFR-2.2, FR-8.3
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from aegis_router.config import AegisConfig, load_config, reload_config
from aegis_router.observability.audit_logger import AuditLogger
from aegis_router.router.model_scorer import ModelScorer, build_routing_table

logger = logging.getLogger(__name__)

# 需要监听的配置文件名
WATCHED_FILES = {
    "models.yaml",
    "route_config.yaml",
    "route_overrides.yaml",
    "capability_profiles.yaml",
    "transaction_templates.yaml",
    "agent_workbuddy.yaml",
}

# 触发事务方案重算的文件集合
TRANSACTION_PLAN_TRIGGER_FILES = {
    "models.yaml",
    "capability_profiles.yaml",
    "transaction_templates.yaml",
}

# 触发 Agent-WorkBuddy 方案重算的文件集合
AGENT_WORKBUDDY_PLAN_TRIGGER_FILES = {
    "models.yaml",
    "capability_profiles.yaml",
    "agent_workbuddy.yaml",
}

# 默认防抖时间（秒）
DEFAULT_DEBOUNCE_SECONDS = 2.0


class _ConfigFileHandler(FileSystemEventHandler):
    """watchdog 文件事件处理器 — 仅对指定文件的修改事件进行处理。"""

    def __init__(self, watcher: "ConfigWatcher") -> None:
        super().__init__()
        self._watcher = watcher

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return

        # 获取文件名并检查是否为监听目标
        filename = Path(event.src_path).name
        if filename not in WATCHED_FILES:
            return

        logger.debug("Detected modification: %s", event.src_path)
        self._watcher._schedule_reload(filename)


class ConfigWatcher:
    """配置热更新监听器 — 后台线程监控配置文件变更。

    当监听的 YAML 配置文件发生修改时，自动重载配置并重建路由表，
    通过回调通知上层模块（如 smart_router.py）。

    当 capability_profiles.yaml、transaction_templates.yaml 或 models.yaml
    发生变更时，自动重算事务级路由方案表并原子替换。

    当 capability_profiles.yaml、agent_workbuddy.yaml 或 models.yaml
    发生变更时，自动重算 Agent-WorkBuddy 路由方案表并原子替换。

    注意：Docker overlay fs 环境下 inotify 不触发，热更新功能仅在宿主机环境生效。

    Attributes:
        config_dir: 配置文件目录
        debounce_seconds: 防抖时间窗口（秒）
    """

    def __init__(
        self,
        config_dir: str | Path,
        on_routing_table_updated: Optional[Callable[[list[dict[str, Any]]], None]] = None,
        on_transaction_plan_updated: Optional[Callable[["RoutingPlanStore"], None]] = None,
        on_agent_workbuddy_plan_updated: Optional[Callable[["AgentPlanStore"], None]] = None,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        """初始化配置监听器。

        Args:
            config_dir: 配置文件目录路径
            on_routing_table_updated: 路由表更新后的回调函数，接收新路由表作为参数
            on_transaction_plan_updated: 事务方案更新后的回调，接收新 RoutingPlanStore
            on_agent_workbuddy_plan_updated: Agent-WorkBuddy 方案更新后的回调，接收新 AgentPlanStore
            debounce_seconds: 防抖时间窗口（秒），默认 2.0s
        """
        self._config_dir = Path(config_dir)
        self._callback = on_routing_table_updated
        self._transaction_plan_callback = on_transaction_plan_updated
        self._agent_workbuddy_plan_callback = on_agent_workbuddy_plan_updated
        self._debounce_seconds = debounce_seconds

        # Audit logger
        self._audit = AuditLogger()

        # 线程安全锁
        self._lock = threading.RLock()

        # 当前状态
        self._config: Optional[AegisConfig] = None
        self._routing_table: list[dict[str, Any]] = []

        # 事务方案相关状态
        self._transaction_plan_store: Optional["RoutingPlanStore"] = None

        # Agent-WorkBuddy 方案相关状态
        self._agent_workbuddy_plan_store: Optional["AgentPlanStore"] = None

        # watchdog observer
        self._observer: Optional[Observer] = None
        self._running = False

        # 防抖定时器
        self._debounce_timer: Optional[threading.Timer] = None
        self._pending_changed_files: set[str] = set()
        self._last_trigger_time: float = 0.0

    def start(self) -> None:
        """启动配置文件监听（后台线程）。

        首次启动时会加载配置并构建初始路由表。
        """
        if self._running:
            logger.warning("ConfigWatcher already running, ignoring start()")
            return

        # 初始加载
        self._do_reload()

        # 设置 watchdog observer
        handler = _ConfigFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._config_dir), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        self._running = True

        logger.info(
            "ConfigWatcher started, watching: %s",
            [str(self._config_dir / f) for f in sorted(WATCHED_FILES)],
        )

    def stop(self) -> None:
        """停止配置文件监听并清理资源。"""
        if not self._running:
            return

        # 取消待执行的防抖定时器
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
            self._debounce_timer = None

        # 停止 observer
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        self._running = False
        logger.info("ConfigWatcher stopped")

    def get_current_routing_table(self) -> list[dict[str, Any]]:
        """线程安全地获取当前路由表。

        Returns:
            当前路由表的副本
        """
        with self._lock:
            return list(self._routing_table)

    def get_current_config(self) -> Optional[AegisConfig]:
        """线程安全地获取当前配置对象。

        Returns:
            当前 AegisConfig 实例，未加载时返回 None
        """
        with self._lock:
            return self._config

    @property
    def is_running(self) -> bool:
        """返回监听器是否正在运行。"""
        return self._running

    def _schedule_reload(self, filename: str) -> None:
        """安排一次防抖重载。

        在防抖时间窗口内的多次触发只会执行一次重载。

        Args:
            filename: 触发变更的文件名
        """
        now = time.time()
        logger.debug("Scheduling reload due to change in: %s", filename)

        # 记录变更的文件（在防抖窗口内累积）
        self._pending_changed_files.add(filename)

        # 取消之前的定时器
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()

        # 设置新的防抖定时器
        self._debounce_timer = threading.Timer(
            self._debounce_seconds, self._do_reload
        )
        self._debounce_timer.daemon = True
        self._debounce_timer.start()
        self._last_trigger_time = now

    def _do_reload(self) -> None:
        """执行配置重载和路由表重建。

        此方法是线程安全的，使用锁保护状态更新。
        如果配置解析失败，保留之前的配置不变。
        """
        # 获取并清空本次触发的变更文件集合
        changed_files = self._pending_changed_files.copy()
        self._pending_changed_files.clear()

        try:
            # 重载配置
            new_config = load_config(self._config_dir)

            # 构建评分器
            scoring_cfg = new_config.routing.scoring
            weights = scoring_cfg.weights.model_dump()
            normalization = scoring_cfg.normalization.model_dump()
            tolerance = scoring_cfg.range_tolerance

            scorer = ModelScorer(
                weights=weights,
                normalization=normalization,
                tolerance=tolerance,
            )

            # 准备模型数据
            models_data = [
                {
                    "name": m.name,
                    "litellm_model": m.litellm_model,
                    "params": m.params.model_dump(),
                }
                for m in new_config.models.models
            ]

            # 准备覆盖数据
            overrides_data = {
                name: override.model_dump()
                for name, override in new_config.overrides.overrides.items()
            }

            # 构建路由表
            new_routing_table = build_routing_table(models_data, scorer, overrides_data)

            # 原子更新（持有锁）
            with self._lock:
                self._config = new_config
                self._routing_table = new_routing_table

            # 更新全局配置单例
            reload_config(self._config_dir)

            logger.info(
                "Config reloaded successfully from %s, routing table has %d entries",
                self._config_dir,
                len(new_routing_table),
            )

            # 触发回调
            if self._callback is not None:
                try:
                    self._callback(new_routing_table)
                except Exception as cb_err:
                    logger.error("Callback error after config reload: %s", cb_err)

        except Exception as e:
            logger.error(
                "Failed to reload config from %s: %s. Keeping previous config.",
                self._config_dir,
                e,
            )

        # 判断是否需要重算事务方案
        if changed_files & TRANSACTION_PLAN_TRIGGER_FILES:
            self._do_transaction_plan_reload(changed_files)

        # 判断是否需要重算 Agent-WorkBuddy 方案
        if changed_files & AGENT_WORKBUDDY_PLAN_TRIGGER_FILES:
            self._do_agent_workbuddy_plan_reload(changed_files)

    def _do_transaction_plan_reload(self, changed_files: set[str]) -> None:
        """重算事务级路由方案表。

        当 capability_profiles.yaml、transaction_templates.yaml 或 models.yaml
        变更时调用，重新生成方案并原子替换。

        Args:
            changed_files: 本次触发变更的文件名集合
        """
        from aegis_router.router.capability_profiles import CapabilityProfileManager
        from aegis_router.router.routing_plan_store import RoutingPlanStore
        from aegis_router.router.template_models import load_templates
        from aegis_router.router.template_plan_generator import TemplatePlanGenerator

        trigger_reason = ", ".join(sorted(changed_files & TRANSACTION_PLAN_TRIGGER_FILES))
        logger.info(
            "检测到配置变更 [%s]，开始重算事务路由方案...",
            trigger_reason,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # 1. 重新加载 AegisConfig（已在 _do_reload 中完成）
            config = self.get_current_config()
            if config is None:
                config = load_config(self._config_dir)

            fallback_model = config.routing.fallback_model

            # 2. 重新加载 CapabilityProfileManager
            profiles_path = self._config_dir / "capability_profiles.yaml"
            profile_manager = CapabilityProfileManager(config_path=profiles_path)

            # 3. 重新加载模板
            templates_path = self._config_dir / "transaction_templates.yaml"
            templates = load_templates(config_path=templates_path)

            # 4. 准备模型数据
            models_data: list[dict[str, Any]] = []
            for entry in config.models.models:
                models_data.append({
                    "name": entry.name,
                    "litellm_model": entry.litellm_model,
                    "params": entry.params.model_dump(),
                })

            # 5. 生成新方案
            if templates and models_data:
                generator = TemplatePlanGenerator(
                    profile_manager=profile_manager,
                    models=models_data,
                    fallback_model=fallback_model,
                )
                new_store = generator.generate_all(templates)
            else:
                new_store = RoutingPlanStore()
                logger.info(
                    "事务方案重算: 模板或模型为空，方案表清空 "
                    "(templates=%d, models=%d)",
                    len(templates),
                    len(models_data),
                )

            # 6. 获取旧方案（用于对比日志）
            with self._lock:
                old_store = self._transaction_plan_store

            old_plans = old_store.get_all_plans() if old_store else {}
            new_plans = new_store.get_all_plans()

            # 7. 日志输出新旧方案差异
            self._log_plan_diff(old_plans, new_plans, trigger_reason, timestamp)

            # 8. 原子替换
            with self._lock:
                self._transaction_plan_store = new_store

            # 9. 触发事务方案回调（外部消费者使用此回调更新引用）
            if self._transaction_plan_callback is not None:
                try:
                    self._transaction_plan_callback(new_store)
                except Exception as cb_err:
                    logger.error(
                        "Transaction plan callback error: %s", cb_err
                    )

            logger.info(
                "事务方案重算完成 (timestamp=%s, entries=%d)",
                timestamp,
                len(new_store),
            )

        except Exception as e:
            logger.error(
                "事务方案重算失败: %s. 保持上一版方案不变。",
                e,
            )

    def _log_plan_diff(
        self,
        old_plans: dict[str, dict[str, str]],
        new_plans: dict[str, dict[str, str]],
        trigger_reason: str,
        timestamp: str,
    ) -> None:
        """输出新旧方案对比日志。

        FR-8.3: 配置版本追踪 — 记录配置变更导致方案重算的时间戳和变更摘要。

        Args:
            old_plans: 旧方案 {template → {agent → model}}
            new_plans: 新方案 {template → {agent → model}}
            trigger_reason: 触发原因（变更文件名）
            timestamp: 变更时间戳
        """
        lines: list[str] = []
        lines.append(f"事务方案变更对比 (触发: {trigger_reason}, 时间: {timestamp})")
        lines.append("-" * 60)

        all_templates = sorted(set(list(old_plans.keys()) + list(new_plans.keys())))

        has_changes = False

        # Structured diff for audit event
        added_templates: list[str] = []
        removed_templates: list[str] = []
        changed_assignments: list[dict[str, str]] = []

        for tpl_name in all_templates:
            old_agents = old_plans.get(tpl_name, {})
            new_agents = new_plans.get(tpl_name, {})

            # 检查模板是否新增
            if not old_agents and new_agents:
                has_changes = True
                added_templates.append(tpl_name)
                lines.append(f"  模板 '{tpl_name}': [新增]")
                for agent, model in sorted(new_agents.items()):
                    lines.append(f"    {agent} → {model}")
                continue

            # 检查模板是否删除
            if old_agents and not new_agents:
                has_changes = True
                removed_templates.append(tpl_name)
                lines.append(f"  模板 '{tpl_name}': [删除]")
                continue

            # 检查 Agent 级变更
            all_agents = sorted(set(list(old_agents.keys()) + list(new_agents.keys())))
            template_changes: list[str] = []

            for agent in all_agents:
                old_model = old_agents.get(agent)
                new_model = new_agents.get(agent)

                if old_model is None and new_model is not None:
                    template_changes.append(
                        f"    {agent}: [新增] → {new_model}"
                    )
                    changed_assignments.append({
                        "template": tpl_name,
                        "agent": agent,
                        "old_model": "",
                        "new_model": new_model,
                    })
                elif old_model is not None and new_model is None:
                    template_changes.append(
                        f"    {agent}: {old_model} → [删除]"
                    )
                    changed_assignments.append({
                        "template": tpl_name,
                        "agent": agent,
                        "old_model": old_model,
                        "new_model": "",
                    })
                elif old_model != new_model:
                    template_changes.append(
                        f"    {agent}: {old_model} → {new_model}"
                    )
                    changed_assignments.append({
                        "template": tpl_name,
                        "agent": agent,
                        "old_model": old_model or "",
                        "new_model": new_model or "",
                    })

            if template_changes:
                has_changes = True
                lines.append(f"  模板 '{tpl_name}' 变化:")
                lines.extend(template_changes)
            else:
                lines.append(f"  模板 '{tpl_name}' 无变化")

        if not has_changes:
            lines.append("  所有模板方案无变化")

        logger.info("\n".join(lines))

        # Emit structured config change audit event (FR-8.3)
        changed_files_list = [f.strip() for f in trigger_reason.split(",") if f.strip()]
        total_changes = (
            len(added_templates) + len(removed_templates) + len(changed_assignments)
        )
        self._audit.log_config_change_event(
            changed_files=changed_files_list,
            trigger_reason=trigger_reason,
            plan_diff_summary={
                "added_templates": added_templates,
                "removed_templates": removed_templates,
                "changed_assignments": changed_assignments,
            },
            total_changes=total_changes,
        )

    def get_transaction_plan_store(self) -> Optional["RoutingPlanStore"]:
        """线程安全地获取当前事务方案表。

        Returns:
            当前 RoutingPlanStore 实例，未初始化时返回 None
        """
        with self._lock:
            return self._transaction_plan_store

    def set_transaction_plan_store(self, store: "RoutingPlanStore") -> None:
        """线程安全地设置事务方案表（初始化时使用）。

        Args:
            store: 初始 RoutingPlanStore 实例
        """
        with self._lock:
            self._transaction_plan_store = store

    def get_agent_workbuddy_plan_store(self) -> Optional["AgentPlanStore"]:
        """线程安全地获取当前 Agent-WorkBuddy 方案表。

        Returns:
            当前 AgentPlanStore 实例，未初始化时返回 None
        """
        with self._lock:
            return self._agent_workbuddy_plan_store

    def set_agent_workbuddy_plan_store(self, store: "AgentPlanStore") -> None:
        """线程安全地设置 Agent-WorkBuddy 方案表（初始化时使用）。

        Args:
            store: 初始 AgentPlanStore 实例
        """
        with self._lock:
            self._agent_workbuddy_plan_store = store

    def _do_agent_workbuddy_plan_reload(self, changed_files: set[str]) -> None:
        """重算 Agent-WorkBuddy 路由方案表。

        当 capability_profiles.yaml、agent_workbuddy.yaml 或 models.yaml
        变更时调用，重新生成方案并原子替换。

        注意：Docker overlay fs 环境下 inotify 不触发，此功能仅在宿主机环境生效。

        Args:
            changed_files: 本次触发变更的文件名集合
        """
        from aegis_router.router.agent_plan_generator import (
            AgentPlanGenerator,
            load_agent_workbuddy_config,
        )
        from aegis_router.router.agent_plan_store import AgentPlanStore
        from aegis_router.router.capability_profiles import CapabilityProfileManager

        trigger_reason = ", ".join(sorted(changed_files & AGENT_WORKBUDDY_PLAN_TRIGGER_FILES))
        logger.info(
            "检测到配置变更 [%s]，开始重算 Agent-WorkBuddy 路由方案...",
            trigger_reason,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # 1. 获取当前配置（已在 _do_reload 中完成）
            config = self.get_current_config()
            if config is None:
                config = load_config(self._config_dir)

            fallback_model = config.routing.fallback_model

            # 2. 重新加载 CapabilityProfileManager
            profiles_path = self._config_dir / "capability_profiles.yaml"
            profile_manager = CapabilityProfileManager(config_path=profiles_path)

            # 3. 重新加载 agent_workbuddy.yaml
            agent_workbuddy_path = self._config_dir / "agent_workbuddy.yaml"
            agents = load_agent_workbuddy_config(config_path=agent_workbuddy_path)

            # 4. 准备模型数据
            models_data: list[dict[str, Any]] = []
            for entry in config.models.models:
                models_data.append({
                    "name": entry.name,
                    "litellm_model": entry.litellm_model,
                    "params": entry.params.model_dump(),
                })

            # 5. 生成新方案
            if agents and models_data:
                generator = AgentPlanGenerator(
                    profile_manager=profile_manager,
                    models=models_data,
                    fallback_model=fallback_model,
                    trigger_reason=trigger_reason,
                )
                new_store = generator.generate_all(agents)
            else:
                new_store = AgentPlanStore()
                logger.info(
                    "Agent-WorkBuddy 方案重算: Agent 或模型为空，方案表清空 "
                    "(agents=%d, models=%d)",
                    len(agents),
                    len(models_data),
                )

            # 6. 获取旧方案（用于对比日志）
            with self._lock:
                old_store = self._agent_workbuddy_plan_store

            old_plans = old_store.get_all_plans() if old_store else {}
            new_plans = new_store.get_all_plans()

            # 7. 日志输出新旧方案差异
            self._log_agent_workbuddy_plan_diff(
                old_plans, new_plans, trigger_reason, timestamp
            )

            # 8. 原子替换
            with self._lock:
                self._agent_workbuddy_plan_store = new_store

            # 9. 触发 Agent-WorkBuddy 方案回调
            if self._agent_workbuddy_plan_callback is not None:
                try:
                    self._agent_workbuddy_plan_callback(new_store)
                except Exception as cb_err:
                    logger.error(
                        "Agent-WorkBuddy plan callback error: %s", cb_err
                    )

            logger.info(
                "Agent-WorkBuddy 方案重算完成 (timestamp=%s, entries=%d)",
                timestamp,
                len(new_store),
            )

        except Exception as e:
            logger.error(
                "Agent-WorkBuddy 方案重算失败: %s. 保持上一版方案不变。",
                e,
            )

    def _log_agent_workbuddy_plan_diff(
        self,
        old_plans: dict[str, str],
        new_plans: dict[str, str],
        trigger_reason: str,
        timestamp: str,
    ) -> None:
        """输出 Agent-WorkBuddy 新旧方案对比日志。

        单维度对比：仅比较 agent → model 的变化。

        Args:
            old_plans: 旧方案 {agent → model}
            new_plans: 新方案 {agent → model}
            trigger_reason: 触发原因（变更文件名）
            timestamp: 变更时间戳
        """
        lines: list[str] = []
        lines.append(
            f"Agent-WorkBuddy 方案变更对比 (触发: {trigger_reason}, 时间: {timestamp})"
        )
        lines.append("-" * 60)

        all_agents = sorted(set(list(old_plans.keys()) + list(new_plans.keys())))

        has_changes = False
        added_agents: list[str] = []
        removed_agents: list[str] = []
        changed_assignments: list[dict[str, str]] = []

        for agent in all_agents:
            old_model = old_plans.get(agent)
            new_model = new_plans.get(agent)

            if old_model is None and new_model is not None:
                has_changes = True
                added_agents.append(agent)
                lines.append(f"  {agent}: [新增] → {new_model}")
            elif old_model is not None and new_model is None:
                has_changes = True
                removed_agents.append(agent)
                lines.append(f"  {agent}: {old_model} → [删除]")
            elif old_model != new_model:
                has_changes = True
                lines.append(f"  {agent}: {old_model} → {new_model}")
                changed_assignments.append({
                    "agent": agent,
                    "old_model": old_model or "",
                    "new_model": new_model or "",
                })

        if not has_changes:
            lines.append("  所有 Agent 方案无变化")

        logger.info("\n".join(lines))

        # Emit structured config change audit event
        changed_files_list = [f.strip() for f in trigger_reason.split(",") if f.strip()]
        total_changes = (
            len(added_agents) + len(removed_agents) + len(changed_assignments)
        )
        self._audit.log_config_change_event(
            changed_files=changed_files_list,
            trigger_reason=trigger_reason,
            plan_diff_summary={
                "added_agents": added_agents,
                "removed_agents": removed_agents,
                "changed_assignments": changed_assignments,
            },
            total_changes=total_changes,
        )
