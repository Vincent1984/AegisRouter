"""Unit tests for plugin_loader agent_workbuddy support.

验证检查点:
- TC-PLUGIN-001: SUPPORTED_PLUGINS 包含 "agent_workbuddy"
- TC-PLUGIN-002: `_initialize_agent_workbuddy_plugin()` 创建有效实例
- TC-PLUGIN-003: `routing_plugin: agent_workbuddy` 正确加载
- TC-PLUGIN-004: agent_workbuddy.yaml 缺失时方案表为空
- TC-PLUGIN-005: 插件互斥验证
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from aegis_router.callbacks.plugin_loader import (
    SUPPORTED_PLUGINS,
    _initialize_agent_workbuddy_plugin,
    get_active_plugin_instance,
    get_active_plugin_type,
    load_routing_plugin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_full_agent_workbuddy_config_dir() -> str:
    """Create a temp config directory with all required YAML files for agent_workbuddy.

    Includes: config.yaml, models.yaml, route_config.yaml, agent_workbuddy.yaml.
    Returns the temp directory path.
    """
    tmpdir = tempfile.mkdtemp()
    config_dir = Path(tmpdir)

    # config.yaml
    config_data = {
        "routing_plugin": "agent_workbuddy",
        "model_list": [],
    }
    (config_dir / "config.yaml").write_text(
        yaml.dump(config_data), encoding="utf-8"
    )

    # models.yaml
    models_data = {
        "models": [
            {
                "name": "test-model",
                "litellm_model": "openai/test",
                "params": {
                    "context_window": 128000,
                    "benchmark_mmlu": 80.0,
                    "benchmark_humaneval": 75.0,
                    "benchmark_math": 70.0,
                    "cost_per_1m_input": 1.0,
                    "cost_per_1m_output": 3.0,
                },
            },
        ]
    }
    (config_dir / "models.yaml").write_text(
        yaml.dump(models_data), encoding="utf-8"
    )

    # route_config.yaml
    route_config_data = {
        "routing": {
            "fallback_model": "test-model",
        }
    }
    (config_dir / "route_config.yaml").write_text(
        yaml.dump(route_config_data), encoding="utf-8"
    )

    # agent_workbuddy.yaml
    agent_workbuddy_data = {
        "agents": [
            {
                "name": "test_agent",
                "capability_profile": "medium",
            },
        ]
    }
    (config_dir / "agent_workbuddy.yaml").write_text(
        yaml.dump(agent_workbuddy_data), encoding="utf-8"
    )

    return tmpdir


# ---------------------------------------------------------------------------
# TC-PLUGIN-001: SUPPORTED_PLUGINS 包含 "agent_workbuddy"
# ---------------------------------------------------------------------------


class TestSupportedPluginsRegistry:
    """TC-PLUGIN-001: SUPPORTED_PLUGINS 包含 agent_workbuddy 条目。"""

    def test_agent_workbuddy_key_exists(self):
        """'agent_workbuddy' 应为 SUPPORTED_PLUGINS 的合法 key。"""
        assert "agent_workbuddy" in SUPPORTED_PLUGINS

    def test_agent_workbuddy_module_path(self):
        """注册条目的模块路径应指向 agent_workbuddy_router 模块。"""
        module_path, _ = SUPPORTED_PLUGINS["agent_workbuddy"]
        assert module_path == "aegis_router.callbacks.agent_workbuddy_router"

    def test_agent_workbuddy_class_name(self):
        """注册条目的类名应为 AgentWorkbuddyCallback。"""
        _, class_name = SUPPORTED_PLUGINS["agent_workbuddy"]
        assert class_name == "AgentWorkbuddyCallback"


# ---------------------------------------------------------------------------
# TC-PLUGIN-002: _initialize_agent_workbuddy_plugin() 创建有效实例
# ---------------------------------------------------------------------------


class TestInitializeAgentWorkbuddyPlugin:
    """TC-PLUGIN-002: _initialize_agent_workbuddy_plugin() 创建有效实例。"""

    def test_returns_agent_workbuddy_callback_instance(self):
        """调用 _initialize_agent_workbuddy_plugin 应返回 AgentWorkbuddyCallback 实例。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback

        instance = _initialize_agent_workbuddy_plugin(Path(tmpdir))

        assert isinstance(instance, AgentWorkbuddyCallback)

    def test_instance_has_plan_store(self):
        """返回的实例应包含 plan_store 属性。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        instance = _initialize_agent_workbuddy_plugin(Path(tmpdir))

        assert hasattr(instance, "plan_store")
        assert instance.plan_store is not None

    def test_instance_has_fallback_model(self):
        """返回的实例应包含正确的 fallback_model。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        instance = _initialize_agent_workbuddy_plugin(Path(tmpdir))

        assert instance.fallback_model == "test-model"

    def test_instance_is_base_router_subclass(self):
        """返回的实例应是 BaseRouterCallback 的子类。"""
        from aegis_router.callbacks.base_router import BaseRouterCallback

        tmpdir = _create_full_agent_workbuddy_config_dir()

        instance = _initialize_agent_workbuddy_plugin(Path(tmpdir))

        assert isinstance(instance, BaseRouterCallback)


# ---------------------------------------------------------------------------
# TC-PLUGIN-003: routing_plugin: agent_workbuddy 正确加载
# ---------------------------------------------------------------------------


class TestLoadRoutingPluginAgentWorkbuddy:
    """TC-PLUGIN-003: load_routing_plugin 正确加载 agent_workbuddy。"""

    def test_returns_agent_workbuddy_callback(self):
        """load_routing_plugin 应返回 AgentWorkbuddyCallback 实例。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback

        plugin = load_routing_plugin(config_dir=tmpdir)

        assert isinstance(plugin, AgentWorkbuddyCallback)

    def test_active_plugin_type_is_agent_workbuddy(self):
        """加载后 get_active_plugin_type() 应返回 'agent_workbuddy'。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        load_routing_plugin(config_dir=tmpdir)

        assert get_active_plugin_type() == "agent_workbuddy"

    def test_active_plugin_instance_matches(self):
        """get_active_plugin_instance() 应返回 load_routing_plugin 返回的同一实例。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        plugin = load_routing_plugin(config_dir=tmpdir)

        assert get_active_plugin_instance() is plugin


# ---------------------------------------------------------------------------
# TC-PLUGIN-004: agent_workbuddy.yaml 缺失时方案表为空
# ---------------------------------------------------------------------------


class TestAgentWorkbuddyMissingConfig:
    """TC-PLUGIN-004: agent_workbuddy.yaml 缺失时方案表为空。"""

    def test_no_crash_without_agent_workbuddy_yaml(self):
        """缺少 agent_workbuddy.yaml 时插件仍正常加载。"""
        tmpdir = tempfile.mkdtemp()
        config_dir = Path(tmpdir)

        # 仅创建 config.yaml（指定 agent_workbuddy）
        config_data = {"routing_plugin": "agent_workbuddy"}
        (config_dir / "config.yaml").write_text(
            yaml.dump(config_data), encoding="utf-8"
        )

        from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback

        plugin = load_routing_plugin(config_dir=tmpdir)

        assert isinstance(plugin, AgentWorkbuddyCallback)

    def test_empty_plan_store_without_agent_workbuddy_yaml(self):
        """缺少 agent_workbuddy.yaml 时 plan_store 应为空。"""
        tmpdir = tempfile.mkdtemp()
        config_dir = Path(tmpdir)

        config_data = {"routing_plugin": "agent_workbuddy"}
        (config_dir / "config.yaml").write_text(
            yaml.dump(config_data), encoding="utf-8"
        )

        plugin = load_routing_plugin(config_dir=tmpdir)

        assert len(plugin.plan_store) == 0


# ---------------------------------------------------------------------------
# TC-PLUGIN-005: 插件互斥验证
# ---------------------------------------------------------------------------


class TestPluginMutualExclusion:
    """TC-PLUGIN-005: 插件互斥 — agent_workbuddy 加载后排斥其他插件。"""

    def test_active_type_not_conversation(self):
        """加载 agent_workbuddy 后，active_plugin_type 不应为 'conversation'。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        load_routing_plugin(config_dir=tmpdir)

        assert get_active_plugin_type() != "conversation"

    def test_active_type_not_transaction(self):
        """加载 agent_workbuddy 后，active_plugin_type 不应为 'transaction'。"""
        tmpdir = _create_full_agent_workbuddy_config_dir()

        load_routing_plugin(config_dir=tmpdir)

        assert get_active_plugin_type() != "transaction"

    def test_instance_is_not_smart_router(self):
        """加载 agent_workbuddy 后，实例不应为 SmartRouterCallback。"""
        from aegis_router.callbacks.smart_router import SmartRouterCallback

        tmpdir = _create_full_agent_workbuddy_config_dir()

        plugin = load_routing_plugin(config_dir=tmpdir)

        assert not isinstance(plugin, SmartRouterCallback)

    def test_instance_is_not_transaction_router(self):
        """加载 agent_workbuddy 后，实例不应为 TransactionRouterCallback。"""
        from aegis_router.callbacks.transaction_router import TransactionRouterCallback

        tmpdir = _create_full_agent_workbuddy_config_dir()

        plugin = load_routing_plugin(config_dir=tmpdir)

        assert not isinstance(plugin, TransactionRouterCallback)

    def test_replaces_previously_loaded_plugin(self):
        """加载 agent_workbuddy 应替换之前已加载的 conversation 插件。"""
        # 先加载 conversation 插件
        conv_tmpdir = tempfile.mkdtemp()
        conv_config_dir = Path(conv_tmpdir)
        (conv_config_dir / "config.yaml").write_text(
            yaml.dump({"routing_plugin": "conversation"}), encoding="utf-8"
        )
        load_routing_plugin(config_dir=conv_tmpdir)
        assert get_active_plugin_type() == "conversation"

        # 再加载 agent_workbuddy 插件
        wb_tmpdir = _create_full_agent_workbuddy_config_dir()
        load_routing_plugin(config_dir=wb_tmpdir)

        # 验证已替换
        assert get_active_plugin_type() == "agent_workbuddy"

        from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback

        assert isinstance(get_active_plugin_instance(), AgentWorkbuddyCallback)
