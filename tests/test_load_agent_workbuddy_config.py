"""load_agent_workbuddy_config YAML 加载函数测试

覆盖:
- 加载有效 YAML 文件
- 处理缺失文件（返回空列表 + 日志警告）
- 处理空文件
- 处理 YAML 语法错误（抛出 ValueError）
- 处理缺少 agents 字段
- 处理缺少 name 字段的条目
- 处理缺少 capability_profile 的条目（默认 medium）
- override_model / description 可选字段解析
"""

import logging
from pathlib import Path

import pytest

from aegis_router.router.agent_plan_generator import (
    AgentWorkbuddyDef,
    load_agent_workbuddy_config,
)


# ============================================================
# 有效 YAML 加载测试
# ============================================================


class TestLoadValidConfig:
    """正常 YAML 加载测试"""

    def test_loads_basic_agents(self, tmp_path):
        """加载包含基本 Agent 定义的 YAML 文件"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - name: intent_classifier\n"
            "    capability_profile: lightweight\n"
            "  - name: reasoning_engine\n"
            "    capability_profile: strong_reasoning\n",
            encoding="utf-8",
        )

        result = load_agent_workbuddy_config(config)

        assert len(result) == 2
        assert result[0].name == "intent_classifier"
        assert result[0].capability_profile == "lightweight"
        assert result[1].name == "reasoning_engine"
        assert result[1].capability_profile == "strong_reasoning"

    def test_loads_override_model(self, tmp_path):
        """正确解析 override_model 字段"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - name: heavy_analyst\n"
            "    capability_profile: heavy\n"
            "    override_model: gpt-5.6-sol\n",
            encoding="utf-8",
        )

        result = load_agent_workbuddy_config(config)

        assert len(result) == 1
        assert result[0].override_model == "gpt-5.6-sol"

    def test_loads_description(self, tmp_path):
        """正确解析 description 字段"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            '  - name: intent_classifier\n'
            '    capability_profile: lightweight\n'
            '    description: "意图分类 Agent"\n',
            encoding="utf-8",
        )

        result = load_agent_workbuddy_config(config)

        assert len(result) == 1
        assert result[0].description == "意图分类 Agent"

    def test_returns_agentworkbuddydef_instances(self, tmp_path):
        """返回的每个元素都是 AgentWorkbuddyDef 实例"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - name: test_agent\n"
            "    capability_profile: medium\n",
            encoding="utf-8",
        )

        result = load_agent_workbuddy_config(config)

        assert len(result) == 1
        assert isinstance(result[0], AgentWorkbuddyDef)

    def test_default_override_model_is_none(self, tmp_path):
        """未配置 override_model 时默认为 None"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - name: basic_agent\n"
            "    capability_profile: lightweight\n",
            encoding="utf-8",
        )

        result = load_agent_workbuddy_config(config)

        assert result[0].override_model is None

    def test_default_description_is_none(self, tmp_path):
        """未配置 description 时默认为 None"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - name: basic_agent\n"
            "    capability_profile: lightweight\n",
            encoding="utf-8",
        )

        result = load_agent_workbuddy_config(config)

        assert result[0].description is None


# ============================================================
# 文件不存在测试
# ============================================================


class TestLoadMissingFile:
    """文件不存在时返回空列表 + 日志警告"""

    def test_missing_file_returns_empty_list(self, tmp_path):
        """文件不存在时返回空列表"""
        non_existent = tmp_path / "does_not_exist.yaml"

        result = load_agent_workbuddy_config(non_existent)

        assert result == []

    def test_missing_file_logs_warning(self, tmp_path, caplog):
        """文件不存在时记录警告日志"""
        non_existent = tmp_path / "does_not_exist.yaml"

        with caplog.at_level(logging.WARNING):
            load_agent_workbuddy_config(non_existent)

        assert "不存在" in caplog.text


# ============================================================
# YAML 语法错误测试
# ============================================================


class TestLoadYamlSyntaxError:
    """YAML 语法错误时抛出 ValueError"""

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        """YAML 语法错误时抛出 ValueError"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - name: test\n"
            "    capability_profile: [invalid: yaml: syntax\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="YAML 语法错误"):
            load_agent_workbuddy_config(config)

    def test_invalid_yaml_error_includes_path(self, tmp_path):
        """错误信息包含文件路径"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text("{{invalid yaml content", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            load_agent_workbuddy_config(config)

        assert "agent_workbuddy.yaml" in str(exc_info.value)


# ============================================================
# 空文件 / 缺少字段测试
# ============================================================


class TestLoadEdgeCases:
    """边界场景测试"""

    def test_empty_file_returns_empty_list(self, tmp_path):
        """空文件返回空列表"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text("", encoding="utf-8")

        result = load_agent_workbuddy_config(config)

        assert result == []

    def test_no_agents_key_returns_empty_list(self, tmp_path):
        """无 agents 字段返回空列表"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text("something_else: true\n", encoding="utf-8")

        result = load_agent_workbuddy_config(config)

        assert result == []

    def test_agents_empty_list_returns_empty_list(self, tmp_path):
        """agents 为空列表时返回空列表"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text("agents: []\n", encoding="utf-8")

        result = load_agent_workbuddy_config(config)

        assert result == []

    def test_missing_name_skips_entry(self, tmp_path, caplog):
        """缺少 name 字段的条目被跳过并记录警告"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - capability_profile: lightweight\n"
            "  - name: valid_agent\n"
            "    capability_profile: medium\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_agent_workbuddy_config(config)

        assert len(result) == 1
        assert result[0].name == "valid_agent"
        assert "name" in caplog.text

    def test_missing_capability_profile_defaults_to_medium(self, tmp_path):
        """缺少 capability_profile 时默认为 medium"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - name: agent_no_profile\n",
            encoding="utf-8",
        )

        result = load_agent_workbuddy_config(config)

        assert len(result) == 1
        assert result[0].capability_profile == "medium"

    def test_non_dict_entry_skipped(self, tmp_path, caplog):
        """非字典类型的条目被跳过"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text(
            "agents:\n"
            "  - just_a_string\n"
            "  - name: valid\n"
            "    capability_profile: lightweight\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_agent_workbuddy_config(config)

        assert len(result) == 1
        assert result[0].name == "valid"

    def test_non_dict_top_level_returns_empty(self, tmp_path):
        """顶层不是字典时返回空列表"""
        config = tmp_path / "agent_workbuddy.yaml"
        config.write_text("- item1\n- item2\n", encoding="utf-8")

        result = load_agent_workbuddy_config(config)

        assert result == []
