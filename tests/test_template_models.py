"""TemplateDef / AgentDef 数据模型与 YAML 加载器测试

覆盖:
- 加载有效 YAML 文件
- 处理缺失文件
- 处理空文件
- 处理格式错误的 YAML
- 字段验证（name、capability_profile 不能为空）
- override_model 解析
- Agent 缺少必填字段时跳过并记录警告
"""

import logging

import pytest

from aegis_router.router.template_models import (
    AgentDef,
    TemplateDef,
    load_templates,
)


# ============================================================
# AgentDef 数据类测试
# ============================================================


class TestAgentDef:
    """AgentDef dataclass 基础测试"""

    def test_basic_creation(self):
        """AgentDef 可正确创建"""
        agent = AgentDef(
            name="intent_classifier",
            capability_profile="lightweight",
        )
        assert agent.name == "intent_classifier"
        assert agent.capability_profile == "lightweight"
        assert agent.override_model is None

    def test_with_override_model(self):
        """AgentDef 支持 override_model"""
        agent = AgentDef(
            name="generator",
            capability_profile="heavy",
            override_model="gpt-5.6-sol",
        )
        assert agent.name == "generator"
        assert agent.capability_profile == "heavy"
        assert agent.override_model == "gpt-5.6-sol"

    def test_override_model_default_none(self):
        """override_model 默认值为 None"""
        agent = AgentDef(name="test", capability_profile="medium")
        assert agent.override_model is None


# ============================================================
# TemplateDef 数据类测试
# ============================================================


class TestTemplateDef:
    """TemplateDef dataclass 基础测试"""

    def test_basic_creation(self):
        """TemplateDef 可正确创建"""
        agents = [
            AgentDef(name="a1", capability_profile="lightweight"),
            AgentDef(name="a2", capability_profile="medium"),
        ]
        template = TemplateDef(
            name="test_template",
            description="测试模板",
            agents=agents,
        )
        assert template.name == "test_template"
        assert template.description == "测试模板"
        assert len(template.agents) == 2
        assert template.agents[0].name == "a1"
        assert template.agents[1].name == "a2"

    def test_empty_agents_list(self):
        """TemplateDef 允许空 Agent 列表"""
        template = TemplateDef(
            name="empty",
            description="无 Agent",
            agents=[],
        )
        assert template.agents == []


# ============================================================
# load_templates: 正常加载测试
# ============================================================


class TestLoadTemplatesValid:
    """正常 YAML 加载测试"""

    def test_load_single_template(self, tmp_path):
        """加载包含单个模板的 YAML"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  resume_screening:
    description: "简历筛选流程"
    agents:
      - name: intent_classifier
        capability_profile: lightweight
      - name: resume_parser
        capability_profile: long_context
""",
            encoding="utf-8",
        )

        result = load_templates(str(config))

        assert len(result) == 1
        assert "resume_screening" in result

        template = result["resume_screening"]
        assert template.name == "resume_screening"
        assert template.description == "简历筛选流程"
        assert len(template.agents) == 2
        assert template.agents[0].name == "intent_classifier"
        assert template.agents[0].capability_profile == "lightweight"
        assert template.agents[1].name == "resume_parser"
        assert template.agents[1].capability_profile == "long_context"

    def test_load_multiple_templates(self, tmp_path):
        """加载包含多个模板的 YAML"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  template_a:
    description: "模板A"
    agents:
      - name: agent1
        capability_profile: lightweight
  template_b:
    description: "模板B"
    agents:
      - name: agent2
        capability_profile: medium
      - name: agent3
        capability_profile: heavy
""",
            encoding="utf-8",
        )

        result = load_templates(str(config))

        assert len(result) == 2
        assert "template_a" in result
        assert "template_b" in result
        assert len(result["template_a"].agents) == 1
        assert len(result["template_b"].agents) == 2

    def test_load_with_override_model(self, tmp_path):
        """加载包含 override_model 的模板"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  custom_pipeline:
    description: "自定义流程"
    agents:
      - name: analyzer
        capability_profile: medium
      - name: generator
        capability_profile: heavy
        override_model: gpt-5.6-sol
""",
            encoding="utf-8",
        )

        result = load_templates(str(config))

        template = result["custom_pipeline"]
        assert template.agents[0].override_model is None
        assert template.agents[1].override_model == "gpt-5.6-sol"

    def test_load_full_example(self, tmp_path):
        """加载完整示例（4 个模板）"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  resume_screening:
    description: "简历筛选流程"
    agents:
      - name: intent_classifier
        capability_profile: lightweight
      - name: resume_parser
        capability_profile: long_context
      - name: skill_matcher
        capability_profile: strong_reasoning
      - name: compliance_checker
        capability_profile: medium

  code_review:
    description: "代码审查流程"
    agents:
      - name: code_analyzer
        capability_profile: code_specialist
      - name: issue_detector
        capability_profile: strong_reasoning
      - name: fix_suggester
        capability_profile: code_specialist

  supplier_evaluation:
    description: "供应商评估流程"
    agents:
      - name: data_collector
        capability_profile: lightweight
      - name: performance_scorer
        capability_profile: medium
      - name: compliance_checker
        capability_profile: medium
      - name: tier_determiner
        capability_profile: strong_reasoning

  custom_pipeline:
    description: "自定义流程"
    agents:
      - name: analyzer
        capability_profile: medium
      - name: generator
        capability_profile: heavy
        override_model: gpt-5.6-sol
""",
            encoding="utf-8",
        )

        result = load_templates(str(config))

        assert len(result) == 4
        assert len(result["resume_screening"].agents) == 4
        assert len(result["code_review"].agents) == 3
        assert len(result["supplier_evaluation"].agents) == 4
        assert len(result["custom_pipeline"].agents) == 2

        # 验证 override_model
        custom = result["custom_pipeline"]
        assert custom.agents[1].override_model == "gpt-5.6-sol"


# ============================================================
# load_templates: 文件缺失/空/异常处理
# ============================================================


class TestLoadTemplatesEdgeCases:
    """文件异常情况处理测试"""

    def test_missing_file_returns_empty(self):
        """文件不存在时返回空 dict"""
        result = load_templates("nonexistent/path/templates.yaml")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        """空文件返回空 dict"""
        config = tmp_path / "templates.yaml"
        config.write_text("", encoding="utf-8")

        result = load_templates(str(config))
        assert result == {}

    def test_malformed_yaml_returns_empty(self, tmp_path, caplog):
        """格式错误的 YAML 返回空 dict 并记录错误日志"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            "{{{{ invalid yaml: [[[",
            encoding="utf-8",
        )

        with caplog.at_level(logging.ERROR):
            result = load_templates(str(config))

        assert result == {}
        assert "解析失败" in caplog.text

    def test_yaml_without_templates_key_returns_empty(self, tmp_path):
        """YAML 中无 'templates' 键时返回空 dict"""
        config = tmp_path / "templates.yaml"
        config.write_text("other_key: value\n", encoding="utf-8")

        result = load_templates(str(config))
        assert result == {}

    def test_templates_key_not_dict_returns_empty(self, tmp_path):
        """'templates' 值不是字典时返回空 dict"""
        config = tmp_path / "templates.yaml"
        config.write_text("templates: not_a_dict\n", encoding="utf-8")

        result = load_templates(str(config))
        assert result == {}

    def test_null_yaml_content_returns_empty(self, tmp_path):
        """YAML 内容为 null 时返回空 dict"""
        config = tmp_path / "templates.yaml"
        config.write_text("null\n", encoding="utf-8")

        result = load_templates(str(config))
        assert result == {}


# ============================================================
# load_templates: 字段验证
# ============================================================


class TestLoadTemplatesValidation:
    """字段验证测试"""

    def test_agent_missing_name_skipped(self, tmp_path, caplog):
        """Agent 缺少 name 字段时跳过并记录警告"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  test:
    description: "测试"
    agents:
      - capability_profile: lightweight
      - name: valid_agent
        capability_profile: medium
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_templates(str(config))

        template = result["test"]
        assert len(template.agents) == 1
        assert template.agents[0].name == "valid_agent"
        assert "缺少 'name'" in caplog.text

    def test_agent_missing_capability_profile_skipped(self, tmp_path, caplog):
        """Agent 缺少 capability_profile 字段时跳过并记录警告"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  test:
    description: "测试"
    agents:
      - name: no_profile_agent
      - name: valid_agent
        capability_profile: medium
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_templates(str(config))

        template = result["test"]
        assert len(template.agents) == 1
        assert template.agents[0].name == "valid_agent"
        assert "缺少 'capability_profile'" in caplog.text

    def test_agent_empty_name_skipped(self, tmp_path, caplog):
        """Agent name 为空字符串时跳过"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  test:
    description: "测试"
    agents:
      - name: ""
        capability_profile: lightweight
      - name: valid
        capability_profile: medium
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_templates(str(config))

        template = result["test"]
        assert len(template.agents) == 1
        assert template.agents[0].name == "valid"

    def test_agent_empty_capability_profile_skipped(self, tmp_path, caplog):
        """Agent capability_profile 为空字符串时跳过"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  test:
    description: "测试"
    agents:
      - name: agent1
        capability_profile: ""
      - name: agent2
        capability_profile: medium
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_templates(str(config))

        template = result["test"]
        assert len(template.agents) == 1
        assert template.agents[0].name == "agent2"

    def test_template_with_no_valid_agents(self, tmp_path):
        """模板中所有 Agent 都无效时，模板仍被创建（agents 列表为空）"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  empty_agents:
    description: "所有 Agent 都无效"
    agents:
      - name: ""
        capability_profile: lightweight
      - capability_profile: medium
""",
            encoding="utf-8",
        )

        result = load_templates(str(config))

        assert "empty_agents" in result
        assert result["empty_agents"].agents == []

    def test_template_missing_description_uses_empty_string(self, tmp_path):
        """模板缺少 description 时使用空字符串"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  no_desc:
    agents:
      - name: agent1
        capability_profile: medium
""",
            encoding="utf-8",
        )

        result = load_templates(str(config))

        assert result["no_desc"].description == ""
        assert len(result["no_desc"].agents) == 1

    def test_template_missing_agents_key_uses_empty_list(self, tmp_path):
        """模板缺少 agents 键时使用空列表"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  no_agents:
    description: "没有 agents 字段"
""",
            encoding="utf-8",
        )

        result = load_templates(str(config))

        assert "no_agents" in result
        assert result["no_agents"].agents == []

    def test_invalid_agent_entry_non_dict_skipped(self, tmp_path, caplog):
        """Agent 条目不是字典时跳过"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  test:
    description: "测试"
    agents:
      - "just a string"
      - name: valid
        capability_profile: medium
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_templates(str(config))

        template = result["test"]
        assert len(template.agents) == 1
        assert template.agents[0].name == "valid"

    def test_agents_not_list_logged_warning(self, tmp_path, caplog):
        """agents 字段不是列表时跳过该模板的 agents"""
        config = tmp_path / "templates.yaml"
        config.write_text(
            """
templates:
  bad_agents:
    description: "agents 不是列表"
    agents: "not a list"
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            result = load_templates(str(config))

        # 模板不被加载（agents 字段无效时跳过）
        assert "bad_agents" not in result
