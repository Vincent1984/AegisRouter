"""V4-2 验证: YAML 语法错误时启动失败，输出明确错误

需求: FR-4.3 — 语法错误保护: YAML 语法错误时拒绝加载，进程启动失败并输出明确错误信息

验证:
- 当 agent_workbuddy.yaml 包含 YAML 语法错误时，load_routing_plugin() 抛出 ValueError
- 错误信息包含 "语法错误" 字样
- 错误信息包含 YAML 解析器的详细错误描述
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from aegis_router.callbacks.plugin_loader import load_routing_plugin


# ---------------------------------------------------------------------------
# Helper: 创建包含 YAML 语法错误的配置目录
# ---------------------------------------------------------------------------


def _create_config_dir_with_invalid_yaml(
    yaml_content: str,
) -> str:
    """创建一个临时配置目录，其中 agent_workbuddy.yaml 包含无效 YAML 内容。

    同时创建必须的 config.yaml、models.yaml、route_config.yaml。
    """
    tmpdir = tempfile.mkdtemp()
    config_dir = Path(tmpdir)

    # config.yaml — 指定 agent_workbuddy 插件
    config_data = {
        "routing_plugin": "agent_workbuddy",
        "model_list": [],
    }
    (config_dir / "config.yaml").write_text(
        yaml.dump(config_data), encoding="utf-8"
    )

    # models.yaml — 最小可用模型定义
    models_data = {
        "models": [
            {
                "name": "test-model",
                "litellm_model": "openai/test",
                "params": {
                    "context_window": 128000,
                    "benchmark_mmlu": 80.0,
                    "benchmark_humaneval": 70.0,
                    "benchmark_math": 65.0,
                    "cost_per_1m_input": 1.0,
                    "cost_per_1m_output": 4.0,
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

    # agent_workbuddy.yaml — 写入无效 YAML 内容
    (config_dir / "agent_workbuddy.yaml").write_text(
        yaml_content, encoding="utf-8"
    )

    return tmpdir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestYamlSyntaxErrorStartupFailure:
    """V4-2: YAML 语法错误时启动失败，输出明确错误"""

    def test_unclosed_bracket_raises_valueerror(self):
        """未闭合的方括号导致启动失败，抛出 ValueError"""
        invalid_yaml = (
            "agents:\n"
            "  - name: test_agent\n"
            "    capability_profile: [unclosed bracket\n"
        )
        tmpdir = _create_config_dir_with_invalid_yaml(invalid_yaml)

        with pytest.raises(ValueError):
            load_routing_plugin(config_dir=tmpdir)

    def test_error_message_contains_syntax_error_chinese(self):
        """错误信息包含 '语法错误' 字样"""
        invalid_yaml = (
            "agents:\n"
            "  - name: test\n"
            "    invalid: {unclosed: brace\n"
        )
        tmpdir = _create_config_dir_with_invalid_yaml(invalid_yaml)

        with pytest.raises(ValueError, match="语法错误"):
            load_routing_plugin(config_dir=tmpdir)

    def test_error_message_contains_yaml_parse_details(self):
        """错误信息包含 YAML 解析器的详细错误描述"""
        invalid_yaml = "{{invalid yaml content that causes parse error"
        tmpdir = _create_config_dir_with_invalid_yaml(invalid_yaml)

        with pytest.raises(ValueError) as exc_info:
            load_routing_plugin(config_dir=tmpdir)

        error_msg = str(exc_info.value)
        # 应包含中文 "语法错误" 标识
        assert "语法错误" in error_msg
        # 应包含插件标识
        assert "Agent-WorkBuddy" in error_msg

    def test_tab_indentation_error(self):
        """Tab 缩进导致 YAML 解析错误时启动失败"""
        invalid_yaml = "agents:\n\t- name: bad_indent\n"
        tmpdir = _create_config_dir_with_invalid_yaml(invalid_yaml)

        with pytest.raises(ValueError, match="语法错误"):
            load_routing_plugin(config_dir=tmpdir)

    def test_duplicate_key_colon_error(self):
        """多余冒号导致 YAML 解析错误"""
        invalid_yaml = (
            "agents:\n"
            "  - name: agent1\n"
            "    capability_profile: : invalid\n"
        )
        tmpdir = _create_config_dir_with_invalid_yaml(invalid_yaml)

        with pytest.raises(ValueError, match="语法错误"):
            load_routing_plugin(config_dir=tmpdir)
