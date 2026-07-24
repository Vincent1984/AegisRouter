"""配置热更新监听模块

使用 watchdog 监听以下配置文件的变更，自动重载配置并重建路由表:
- config/models.yaml
- config/route_config.yaml
- config/route_overrides.yaml

设计参考: design.md 2.3.6 节
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from aegis_router.config import AegisConfig, load_config, reload_config
from aegis_router.router.model_scorer import ModelScorer, build_routing_table

logger = logging.getLogger(__name__)

# 需要监听的配置文件名
WATCHED_FILES = {"models.yaml", "route_config.yaml", "route_overrides.yaml"}

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

    Attributes:
        config_dir: 配置文件目录
        debounce_seconds: 防抖时间窗口（秒）
    """

    def __init__(
        self,
        config_dir: str | Path,
        on_routing_table_updated: Optional[Callable[[list[dict[str, Any]]], None]] = None,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        """初始化配置监听器。

        Args:
            config_dir: 配置文件目录路径
            on_routing_table_updated: 路由表更新后的回调函数，接收新路由表作为参数
            debounce_seconds: 防抖时间窗口（秒），默认 2.0s
        """
        self._config_dir = Path(config_dir)
        self._callback = on_routing_table_updated
        self._debounce_seconds = debounce_seconds

        # 线程安全锁
        self._lock = threading.RLock()

        # 当前状态
        self._config: Optional[AegisConfig] = None
        self._routing_table: list[dict[str, Any]] = []

        # watchdog observer
        self._observer: Optional[Observer] = None
        self._running = False

        # 防抖定时器
        self._debounce_timer: Optional[threading.Timer] = None
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
