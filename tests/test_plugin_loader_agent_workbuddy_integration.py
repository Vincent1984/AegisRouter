"""Integration tests for agent_workbuddy plugin startup plan generation.

验证检查点 V3-2:
- 当 routing_plugin=agent_workbuddy 且所有配置文件齐全时，方案表正确生成
- 启动日志包含 Agent-WorkBuddy 方案表
- 当 agent_workbuddy.yaml 不存在时，插件仍能正常加载（优雅降级，空方案表）
- override_model 被正确尊重
- fallback_model 来自 route_config.yaml
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import yaml

from aegis_router.callbacks.plugin_loader import load_routing_plugin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_full_config_dir() -> str:
    """Create a temp config directory with all required YAML files for agent_workbuddy.

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
                "name": "cheap-model",
                "litellm_model": "ollama/cheap",
                "params": {
                    "context_window": 32000,
                    "benchmark_mmlu": 60.0,
                    "benchmark_humaneval": 40.0,
                    "benchmark_math": 35.0,
                    "cost_per_1m_input": 0.0,
                    "cost_per_1m_output": 0.0,
                },
            },
            {
                "name": "mid-model",
                "litellm_model": "openai/mid",
                "params": {
                    "context_window": 128000,
                    "benchmark_mmlu": 85.0,
                    "benchmark_humaneval": 80.0,
                    "benchmark_math": 75.0,
                    "cost_per_1m_input": 1.0,
                    "cost_per_1m_output": 4.0,
                },
            },
            {
                "name": "strong-model",
                "litellm_model": "openai/strong",
                "params": {
                    "context_window": 200000,
                    "benchmark_mmlu": 92.0,
                    "benchmark_humaneval": 93.0,
                    "benchmark_math": 90.0,
                    "cost_per_1m_input": 5.0,
                    "cost_per_1m_output": 20.0,
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
            "fallback_model": "mid-model",
        }
    }
    (config_dir / "route_config.yaml").write_text(
        yaml.dump(route_config_data), encoding="utf-8"
    )

    # agent_workbuddy.yaml — flat agents list (not nested templates)
    agent_workbuddy_data = {
        "agents": [
            {
                "name": "classifier",
                "capability_profile": "lightweight",
            },
            {
                "name": "processor",
                "capability_profile": "medium",
            },
            {
                "name": "fixed_agent",
                "capability_profile": "heavy",
                "override_model": "strong-model",
            },
        ],
    }
    (config_dir / "agent_workbuddy.yaml").write_text(
        yaml.dump(agent_workbuddy_data), encoding="utf-8"
    )

    return tmpdir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentWorkbuddyPluginIntegration:
    """Integration tests for plan generation during agent_workbuddy plugin startup."""

    def test_full_config_loads_populated_plan_store(self):
        """当所有配置文件齐全时，插件加载带有方案的 plan_store。"""
        tmpdir = _create_full_config_dir()

        plugin = load_routing_plugin(config_dir=tmpdir)

        from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback

        assert isinstance(plugin, AgentWorkbuddyCallback)

        # plan_store 应包含方案条目
        assert len(plugin.plan_store) > 0

        # 验证所有 agent 的方案已生成
        model_classifier = plugin.plan_store.get_model("classifier")
        model_processor = plugin.plan_store.get_model("processor")
        model_fixed = plugin.plan_store.get_model("fixed_agent")

        assert model_classifier is not None
        assert model_processor is not None
        assert model_fixed is not None

    def test_override_model_respected(self):
        """override_model 应直接使用指定模型，不经过 Profile 评分。"""
        tmpdir = _create_full_config_dir()

        plugin = load_routing_plugin(config_dir=tmpdir)

        # fixed_agent 声明 override_model: strong-model
        model = plugin.plan_store.get_model("fixed_agent")
        assert model == "strong-model"

    def test_fallback_model_from_route_config(self):
        """fallback_model 应来自 route_config.yaml。"""
        tmpdir = _create_full_config_dir()

        plugin = load_routing_plugin(config_dir=tmpdir)

        assert plugin.fallback_model == "mid-model"

    def test_no_agent_workbuddy_file_graceful_degradation(self):
        """当 agent_workbuddy.yaml 不存在时，插件仍正常加载，方案表为空。"""
        tmpdir = tempfile.mkdtemp()
        config_dir = Path(tmpdir)

        # 仅创建 config.yaml 指定 agent_workbuddy 插件
        config_data = {"routing_plugin": "agent_workbuddy"}
        (config_dir / "config.yaml").write_text(
            yaml.dump(config_data), encoding="utf-8"
        )

        plugin = load_routing_plugin(config_dir=tmpdir)

        from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback

        assert isinstance(plugin, AgentWorkbuddyCallback)
        assert len(plugin.plan_store) == 0

    def test_startup_log_contains_plan_table(self, caplog):
        """启动日志包含 Agent-WorkBuddy 方案表。"""
        tmpdir = _create_full_config_dir()

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            plugin = load_routing_plugin(config_dir=tmpdir)

        # 验证日志中包含关键信息
        log_text = caplog.text
        assert "Agent-WorkBuddy" in log_text
        assert "classifier" in log_text
        assert "processor" in log_text
        assert "fixed_agent" in log_text

    def test_startup_log_empty_plan(self, caplog):
        """当方案表为空时，日志应明确说明。"""
        tmpdir = tempfile.mkdtemp()
        config_dir = Path(tmpdir)

        config_data = {"routing_plugin": "agent_workbuddy"}
        (config_dir / "config.yaml").write_text(
            yaml.dump(config_data), encoding="utf-8"
        )

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            plugin = load_routing_plugin(config_dir=tmpdir)

        log_text = caplog.text
        assert "方案表为空" in log_text
