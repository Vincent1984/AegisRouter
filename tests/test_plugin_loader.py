"""Tests for Plugin Loader (Phase 1 验证检查点 V1-1 ~ V1-4)

验证:
- V1-2: routing_plugin: conversation → 加载 SmartRouterCallback
- V1-3: routing_plugin: unknown_plugin → ValueError，含可选值列表
- V1-4: 无 routing_plugin 字段 → 默认加载 conversation（向后兼容）
- 额外: routing_plugin: transaction → 能正确 import（类存在时）
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aegis_router.callbacks.plugin_loader import (
    SUPPORTED_PLUGINS,
    _read_routing_plugin_field,
    get_available_plugins,
    load_routing_plugin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_temp_config(config_data: dict) -> str:
    """Create a temp directory with a config.yaml containing given data.

    Returns the temp directory path.
    """
    tmpdir = tempfile.mkdtemp()
    config_path = Path(tmpdir) / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return tmpdir


def _create_temp_config_raw(content: str) -> str:
    """Create a temp directory with raw config.yaml content.

    Returns the temp directory path.
    """
    tmpdir = tempfile.mkdtemp()
    config_path = Path(tmpdir) / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return tmpdir


# ---------------------------------------------------------------------------
# Test: _read_routing_plugin_field
# ---------------------------------------------------------------------------


class TestReadRoutingPluginField:
    """Test the config field reader."""

    def test_conversation_explicit(self):
        """routing_plugin: conversation → 返回 'conversation'"""
        tmpdir = _create_temp_config({"routing_plugin": "conversation"})
        assert _read_routing_plugin_field(tmpdir) == "conversation"

    def test_transaction_explicit(self):
        """routing_plugin: transaction → 返回 'transaction'"""
        tmpdir = _create_temp_config({"routing_plugin": "transaction"})
        assert _read_routing_plugin_field(tmpdir) == "transaction"

    def test_unknown_plugin(self):
        """routing_plugin: unknown → 返回 'unknown'（验证由调用方处理）"""
        tmpdir = _create_temp_config({"routing_plugin": "unknown_plugin"})
        assert _read_routing_plugin_field(tmpdir) == "unknown_plugin"

    def test_field_missing_defaults_to_conversation(self):
        """V1-4: 无 routing_plugin 字段 → 默认 'conversation'"""
        tmpdir = _create_temp_config({"model_list": []})
        assert _read_routing_plugin_field(tmpdir) == "conversation"

    def test_empty_config_defaults_to_conversation(self):
        """空 config.yaml → 默认 'conversation'"""
        tmpdir = _create_temp_config_raw("")
        assert _read_routing_plugin_field(tmpdir) == "conversation"

    def test_config_file_not_exist_defaults_to_conversation(self):
        """config.yaml 不存在 → 默认 'conversation'"""
        tmpdir = tempfile.mkdtemp()  # No config.yaml created
        assert _read_routing_plugin_field(tmpdir) == "conversation"

    def test_invalid_yaml_defaults_to_conversation(self):
        """config.yaml 语法错误 → 默认 'conversation'"""
        tmpdir = _create_temp_config_raw("{{invalid yaml::")
        assert _read_routing_plugin_field(tmpdir) == "conversation"


# ---------------------------------------------------------------------------
# Test: load_routing_plugin
# ---------------------------------------------------------------------------


class TestLoadRoutingPlugin:
    """Test the plugin loading function."""

    def test_load_conversation_plugin(self):
        """V1-2: routing_plugin: conversation → 加载 SmartRouterCallback"""
        tmpdir = _create_temp_config({"routing_plugin": "conversation"})

        plugin = load_routing_plugin(config_dir=tmpdir)

        from aegis_router.callbacks.smart_router import SmartRouterCallback

        assert isinstance(plugin, SmartRouterCallback)

    def test_load_conversation_plugin_default(self):
        """V1-4: 无 routing_plugin 字段 → 默认加载 SmartRouterCallback"""
        tmpdir = _create_temp_config({"model_list": []})

        plugin = load_routing_plugin(config_dir=tmpdir)

        from aegis_router.callbacks.smart_router import SmartRouterCallback

        assert isinstance(plugin, SmartRouterCallback)

    def test_unknown_plugin_raises_value_error(self):
        """V1-3: routing_plugin: unknown_plugin → ValueError，含可选值"""
        tmpdir = _create_temp_config({"routing_plugin": "unknown_plugin"})

        with pytest.raises(ValueError) as exc_info:
            load_routing_plugin(config_dir=tmpdir)

        error_msg = str(exc_info.value)
        assert "unknown_plugin" in error_msg
        assert "conversation" in error_msg
        assert "transaction" in error_msg

    def test_unknown_plugin_error_lists_all_supported(self):
        """错误信息包含所有支持的插件名"""
        tmpdir = _create_temp_config({"routing_plugin": "bad_name"})

        with pytest.raises(ValueError) as exc_info:
            load_routing_plugin(config_dir=tmpdir)

        error_msg = str(exc_info.value)
        for plugin_name in SUPPORTED_PLUGINS:
            assert plugin_name in error_msg

    def test_load_transaction_plugin_success(self):
        """transaction 插件加载成功，返回 TransactionRouterCallback 实例"""
        tmpdir = _create_temp_config({"routing_plugin": "transaction"})

        plugin = load_routing_plugin(config_dir=tmpdir)

        from aegis_router.callbacks.transaction_router import TransactionRouterCallback

        assert isinstance(plugin, TransactionRouterCallback)

    def test_env_var_fallback_for_config_dir(self):
        """config_dir=None 时使用 AEGIS_CONFIG_DIR 环境变量"""
        tmpdir = _create_temp_config({"routing_plugin": "conversation"})

        with patch.dict(os.environ, {"AEGIS_CONFIG_DIR": tmpdir}):
            plugin = load_routing_plugin(config_dir=None)

        from aegis_router.callbacks.smart_router import SmartRouterCallback

        assert isinstance(plugin, SmartRouterCallback)


# ---------------------------------------------------------------------------
# Test: get_available_plugins
# ---------------------------------------------------------------------------


class TestGetAvailablePlugins:
    """Test the plugin listing function."""

    def test_returns_sorted_list(self):
        """返回排序后的插件名列表"""
        plugins = get_available_plugins()
        assert plugins == sorted(plugins)

    def test_contains_conversation_and_transaction(self):
        """包含 conversation 和 transaction"""
        plugins = get_available_plugins()
        assert "conversation" in plugins
        assert "transaction" in plugins

    def test_returns_list_type(self):
        """返回 list 类型"""
        plugins = get_available_plugins()
        assert isinstance(plugins, list)


# ---------------------------------------------------------------------------
# Test: SmartRouterCallback 仍然是 BaseRouterCallback 子类 (V1-1 补充)
# ---------------------------------------------------------------------------


class TestSmartRouterInheritance:
    """Verify SmartRouterCallback correctly inherits from BaseRouterCallback."""

    def test_is_subclass_of_base_router(self):
        """SmartRouterCallback 是 BaseRouterCallback 的子类"""
        from aegis_router.callbacks.base_router import BaseRouterCallback
        from aegis_router.callbacks.smart_router import SmartRouterCallback

        assert issubclass(SmartRouterCallback, BaseRouterCallback)

    def test_has_execute_routing_method(self):
        """SmartRouterCallback 实现了 _execute_routing 方法"""
        from aegis_router.callbacks.smart_router import SmartRouterCallback

        assert hasattr(SmartRouterCallback, "_execute_routing")
        # Should not be abstract
        import inspect

        assert not getattr(
            SmartRouterCallback._execute_routing, "__isabstractmethod__", False
        )

    def test_instance_is_base_router(self):
        """SmartRouterCallback 实例 isinstance 检查通过"""
        from aegis_router.callbacks.base_router import BaseRouterCallback
        from aegis_router.callbacks.smart_router import SmartRouterCallback

        instance = SmartRouterCallback(enable_routing=False)
        assert isinstance(instance, BaseRouterCallback)
