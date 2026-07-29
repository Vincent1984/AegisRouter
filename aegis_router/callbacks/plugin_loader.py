"""路由插件加载器 (Plugin Loader)

根据 config.yaml 中的 `routing_plugin` 字段加载对应的路由策略插件。

支持的插件:
  - conversation: 对话级路由 (SmartRouterCallback) — 默认值
  - transaction:  事务级路由 (TransactionRouterCallback)

用法:
    from aegis_router.callbacks.plugin_loader import load_routing_plugin

    callback = load_routing_plugin(config_dir="./config")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from aegis_router.callbacks.base_router import BaseRouterCallback

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Active plugin registry (module-level singleton)
# ---------------------------------------------------------------------------

# 当前活跃的路由插件实例和类型，供 /health/routing 等外部模块查询
_active_plugin_instance: "BaseRouterCallback | None" = None
_active_plugin_type: str = "unknown"


def get_active_plugin_instance() -> "BaseRouterCallback | None":
    """获取当前活跃的路由插件实例。"""
    return _active_plugin_instance


def get_active_plugin_type() -> str:
    """获取当前活跃的路由插件类型名称（如 'conversation' 或 'transaction'）。"""
    return _active_plugin_type


# ---------------------------------------------------------------------------
# Supported plugins registry
# ---------------------------------------------------------------------------

# 可选值 → (模块路径, 类名) 的映射
SUPPORTED_PLUGINS: dict[str, tuple[str, str]] = {
    "conversation": (
        "aegis_router.callbacks.smart_router",
        "SmartRouterCallback",
    ),
    "transaction": (
        "aegis_router.callbacks.transaction_router",
        "TransactionRouterCallback",
    ),
}


# ---------------------------------------------------------------------------
# Config reading helper
# ---------------------------------------------------------------------------


def _read_routing_plugin_field(config_dir: str | Path) -> str:
    """从 config.yaml 中读取 routing_plugin 字段。

    Returns:
        插件名称字符串。未配置时返回默认值 'conversation'。
    """
    config_path = Path(config_dir) / "config.yaml"

    if not config_path.exists():
        logger.warning(
            "config.yaml not found at %s, using default routing_plugin='conversation'",
            config_path,
        )
        return "conversation"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.error(
            "Failed to parse config.yaml: %s. Using default routing_plugin='conversation'",
            e,
        )
        return "conversation"

    if not isinstance(data, dict):
        return "conversation"

    return data.get("routing_plugin", "conversation")


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------


def _initialize_transaction_plugin(
    config_dir: Path,
    **kwargs,
) -> BaseRouterCallback:
    """初始化事务级路由插件，包含方案预计算。

    加载流程:
    1. 使用 load_config() 加载 AegisConfig（含 models + route_config）
    2. 创建 CapabilityProfileManager
    3. 加载 transaction_templates.yaml
    4. 转换模型数据为 dict 格式
    5. 创建 TemplatePlanGenerator 并调用 generate_all()
    6. 构造 TransactionRouterCallback 实例
    7. 输出完整方案表到启动日志

    当配置文件不存在时，优雅降级为空方案（所有请求走 fallback）。

    Parameters
    ----------
    config_dir : Path
        配置目录路径。
    **kwargs
        额外参数传递给 TransactionRouterCallback 构造函数。

    Returns
    -------
    BaseRouterCallback
        初始化完成的 TransactionRouterCallback 实例。
    """
    from aegis_router.callbacks.transaction_router import TransactionRouterCallback
    from aegis_router.config import load_config
    from aegis_router.router.capability_profiles import CapabilityProfileManager
    from aegis_router.router.routing_plan_store import RoutingPlanStore
    from aegis_router.router.template_models import load_templates
    from aegis_router.router.template_plan_generator import TemplatePlanGenerator

    # Step 1: 加载聚合配置
    aegis_config = load_config(config_dir)
    fallback_model = aegis_config.routing.fallback_model

    # Step 1.5: 提取 failover 链配置 (FR-6.2)
    failover_chains = aegis_config.failover.chains
    failover_enabled = aegis_config.failover.enabled

    # Step 2: 创建 CapabilityProfileManager
    profiles_path = config_dir / "capability_profiles.yaml"
    profile_manager = CapabilityProfileManager(config_path=profiles_path)

    # Step 3: 加载事务模板
    templates_path = config_dir / "transaction_templates.yaml"
    templates = load_templates(config_path=templates_path)

    # Step 4: 转换模型条目为 dict 格式（TemplatePlanGenerator 需要）
    models_data: list[dict[str, Any]] = []
    for entry in aegis_config.models.models:
        models_data.append({
            "name": entry.name,
            "litellm_model": entry.litellm_model,
            "params": entry.params.model_dump(),
        })

    # Step 5 & 6: 生成方案表
    if templates and models_data:
        generator = TemplatePlanGenerator(
            profile_manager=profile_manager,
            models=models_data,
            fallback_model=fallback_model,
        )
        plan_store = generator.generate_all(templates)
    else:
        plan_store = RoutingPlanStore()
        if not templates:
            logger.info(
                "Transaction Router: 无模板定义，方案表为空，所有请求将使用 fallback 模型 '%s'",
                fallback_model,
            )
        if not models_data:
            logger.info(
                "Transaction Router: 无模型定义，方案表为空，所有请求将使用 fallback 模型 '%s'",
                fallback_model,
            )

    # Step 7: 输出方案表到启动日志
    _log_plan_table(plan_store, fallback_model)

    # 构造插件实例
    instance = TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model=fallback_model,
        failover_chains=failover_chains,
        failover_enabled=failover_enabled,
        config_dir=str(config_dir),
        **kwargs,
    )

    # Step 8: 启动 ConfigWatcher 热更新 (FR-2.5, FR-3.6, FR-4.2)
    try:
        from aegis_router.router.config_watcher import ConfigWatcher

        def _on_plan_updated(new_store: "RoutingPlanStore") -> None:
            """热更新回调：原子替换 plan_store 引用。"""
            instance.plan_store = new_store
            logger.info(
                "Transaction Router: 方案热更新完成, new_entries=%d",
                len(new_store),
            )

        config_watcher = ConfigWatcher(
            config_dir=config_dir,
            on_transaction_plan_updated=_on_plan_updated,
        )
        config_watcher.set_transaction_plan_store(plan_store)
        config_watcher.start()

        # 将 watcher 引用存储在实例上，便于后续停止
        instance._config_watcher = config_watcher

        logger.info(
            "Transaction Router: ConfigWatcher started, "
            "monitoring capability_profiles.yaml, transaction_templates.yaml, models.yaml"
        )
    except Exception as e:
        logger.error(
            "Transaction Router: Failed to start ConfigWatcher: %s. "
            "Hot-reload disabled.",
            e,
        )

    return instance


def _log_plan_table(plan_store: "RoutingPlanStore", fallback_model: str) -> None:
    """输出完整方案表到启动日志。

    格式:
    ┌─────────────────────────────────────────────────────────┐
    │ Transaction Router - Routing Plan Table                  │
    ├──────────────┬──────────────────┬───────────────────────┤
    │ Template     │ Agent            │ Model                 │
    ├──────────────┼──────────────────┼───────────────────────┤
    │ ...          │ ...              │ ...                   │
    └──────────────┴──────────────────┴───────────────────────┘
    """
    from aegis_router.router.routing_plan_store import RoutingPlanStore

    all_plans = plan_store.get_all_plans()

    if not all_plans:
        logger.info(
            "Transaction Router: 方案表为空 (fallback_model=%s)",
            fallback_model,
        )
        return

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Transaction Router - Routing Plan Table")
    lines.append(f"  Fallback Model: {fallback_model}")
    lines.append(f"  Total Entries: {len(plan_store)}")
    lines.append("-" * 70)
    lines.append(f"  {'Template':<20} {'Agent':<20} {'Model':<25}")
    lines.append("-" * 70)

    for template_name in sorted(all_plans.keys()):
        agent_map = all_plans[template_name]
        for agent_name in sorted(agent_map.keys()):
            model = agent_map[agent_name]
            lines.append(f"  {template_name:<20} {agent_name:<20} {model:<25}")

    lines.append("=" * 70)

    logger.info("\n".join(lines))


def load_routing_plugin(
    config_dir: str | Path | None = None,
    **kwargs,
) -> BaseRouterCallback:
    """根据配置加载路由策略插件实例。

    Parameters
    ----------
    config_dir : str | Path | None
        配置目录路径。为 None 时使用环境变量 AEGIS_CONFIG_DIR 或默认 './config'。
    **kwargs
        额外参数传递给插件构造函数。

    Returns
    -------
    BaseRouterCallback
        路由插件实例。

    Raises
    ------
    ValueError
        当 routing_plugin 字段的值不在支持列表中时抛出。
    """
    if config_dir is None:
        config_dir = os.environ.get("AEGIS_CONFIG_DIR", "./config")

    config_dir = Path(config_dir)

    # 读取插件名
    plugin_name = _read_routing_plugin_field(config_dir)

    logger.info("Loading routing plugin: '%s'", plugin_name)

    # 验证插件名
    if plugin_name not in SUPPORTED_PLUGINS:
        available = ", ".join(sorted(SUPPORTED_PLUGINS.keys()))
        raise ValueError(
            f"Unknown routing_plugin '{plugin_name}'. "
            f"Supported values: [{available}]"
        )

    global _active_plugin_instance, _active_plugin_type

    # --- 事务级路由: 专用初始化路径 ---
    if plugin_name == "transaction":
        instance = _initialize_transaction_plugin(config_dir, **kwargs)
        logger.info(
            "Routing plugin '%s' loaded successfully: %s",
            plugin_name,
            type(instance).__name__,
        )
        _active_plugin_instance = instance
        _active_plugin_type = plugin_name
        return instance

    # --- 其他插件: 通用初始化路径 ---
    module_path, class_name = SUPPORTED_PLUGINS[plugin_name]

    try:
        import importlib

        module = importlib.import_module(module_path)
        plugin_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Failed to import routing plugin '{plugin_name}' "
            f"(module={module_path}, class={class_name}): {e}"
        ) from e

    # 实例化插件
    try:
        instance = plugin_class(config_dir=str(config_dir), **kwargs)
    except TypeError:
        # 某些插件可能不接受 config_dir 参数，尝试无参数实例化
        instance = plugin_class(**kwargs)

    logger.info(
        "Routing plugin '%s' loaded successfully: %s",
        plugin_name,
        type(instance).__name__,
    )

    _active_plugin_instance = instance
    _active_plugin_type = plugin_name
    return instance


def get_available_plugins() -> list[str]:
    """返回所有支持的插件名称列表。"""
    return sorted(SUPPORTED_PLUGINS.keys())
