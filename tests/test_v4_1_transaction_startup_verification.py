"""V4-1 验证检查点: routing_plugin: transaction 启动成功，日志输出方案表

使用真实 config/ 目录验证:
1. 当 routing_plugin 设为 transaction 时，插件启动成功无报错
2. 启动日志输出完整方案表（包含所有 template → agent → model 分配）
3. plan_store 包含 4 个模板的全部 13 个 Agent 条目
   - resume_screening: 4 agents (intent_classifier, resume_parser, skill_matcher, compliance_checker)
   - code_review: 3 agents (code_analyzer, issue_detector, fix_suggester)
   - supplier_evaluation: 4 agents (data_collector, performance_scorer, compliance_checker, tier_determiner)
   - custom_pipeline: 2 agents (analyzer, generator)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from aegis_router.callbacks.plugin_loader import load_routing_plugin


# ---------------------------------------------------------------------------
# Helper: create a temp config dir mirroring real config/ but with
# routing_plugin set to 'transaction'
# ---------------------------------------------------------------------------


def _create_transaction_config_dir() -> str:
    """Copy real config/ to a temp dir with routing_plugin set to 'transaction'.

    This avoids modifying the actual config/config.yaml.
    """
    real_config_dir = Path(__file__).parent.parent / "config"
    tmpdir = tempfile.mkdtemp(prefix="aegis_v4_1_")
    config_dir = Path(tmpdir)

    # Copy all config files
    for f in real_config_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, config_dir / f.name)
        elif f.is_dir():
            shutil.copytree(f, config_dir / f.name)

    # Override routing_plugin to 'transaction'
    config_path = config_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config_data = yaml.safe_load(fh)

    config_data["routing_plugin"] = "transaction"

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.dump(config_data, fh, allow_unicode=True)

    return tmpdir


# ---------------------------------------------------------------------------
# V4-1 Verification Tests
# ---------------------------------------------------------------------------


class TestV4_1_TransactionStartupVerification:
    """V4-1: routing_plugin: transaction 启动成功，日志输出方案表。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create temp config dir with routing_plugin=transaction."""
        self.config_dir = _create_transaction_config_dir()
        yield
        # Cleanup handled by OS temp dir cleanup

    def test_plugin_starts_successfully(self):
        """插件以 routing_plugin=transaction 启动，无异常。"""
        from aegis_router.callbacks.transaction_router import TransactionRouterCallback

        plugin = load_routing_plugin(config_dir=self.config_dir)

        assert plugin is not None
        assert isinstance(plugin, TransactionRouterCallback)

    def test_plan_store_has_all_13_entries(self):
        """方案表包含 4 个模板共 13 个 Agent 条目。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        # Total: resume_screening(4) + code_review(3) + supplier_evaluation(4) + custom_pipeline(2) = 13
        assert len(plugin.plan_store) == 13

    def test_all_templates_present_in_plan_store(self):
        """方案表包含 transaction_templates.yaml 中定义的全部 4 个模板。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        all_plans = plugin.plan_store.get_all_plans()

        expected_templates = {
            "resume_screening",
            "code_review",
            "supplier_evaluation",
            "custom_pipeline",
        }
        assert set(all_plans.keys()) == expected_templates

    def test_resume_screening_agents(self):
        """resume_screening 模板包含 4 个正确的 Agent。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        plan = plugin.plan_store.get_template_plan("resume_screening")

        expected_agents = {
            "intent_classifier",
            "resume_parser",
            "skill_matcher",
            "compliance_checker",
        }
        assert set(plan.keys()) == expected_agents

    def test_code_review_agents(self):
        """code_review 模板包含 3 个正确的 Agent。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        plan = plugin.plan_store.get_template_plan("code_review")

        expected_agents = {
            "code_analyzer",
            "issue_detector",
            "fix_suggester",
        }
        assert set(plan.keys()) == expected_agents

    def test_supplier_evaluation_agents(self):
        """supplier_evaluation 模板包含 4 个正确的 Agent。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        plan = plugin.plan_store.get_template_plan("supplier_evaluation")

        expected_agents = {
            "data_collector",
            "performance_scorer",
            "compliance_checker",
            "tier_determiner",
        }
        assert set(plan.keys()) == expected_agents

    def test_custom_pipeline_agents(self):
        """custom_pipeline 模板包含 2 个正确的 Agent。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        plan = plugin.plan_store.get_template_plan("custom_pipeline")

        expected_agents = {"analyzer", "generator"}
        assert set(plan.keys()) == expected_agents

    def test_custom_pipeline_override_model(self):
        """custom_pipeline.generator 使用 override_model: gpt-5.6-sol。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        model = plugin.plan_store.get_model("custom_pipeline", "generator")
        assert model == "gpt-5.6-sol"

    def test_all_agents_have_valid_model_assignments(self):
        """所有 Agent 都被分配了 models.yaml 中已定义的模型。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        # Valid model names from models.yaml
        valid_models = {
            "local-7b",
            "deepseek-v4-pro",
            "claude-sonnet",
            "gpt-5.2",
            "gpt-5.4-mini",
            "gpt-5.5",
            "gpt-5.6-sol",
            "codex-mini",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3.1-pro",
        }

        all_plans = plugin.plan_store.get_all_plans()
        for template_name, agent_map in all_plans.items():
            for agent_name, model in agent_map.items():
                assert model in valid_models, (
                    f"Template '{template_name}', Agent '{agent_name}' "
                    f"assigned to unknown model '{model}'"
                )

    def test_log_contains_loading_message(self, caplog):
        """启动日志包含 'Loading routing plugin: transaction' 信息。"""
        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            load_routing_plugin(config_dir=self.config_dir)

        log_text = caplog.text
        assert "Loading routing plugin: 'transaction'" in log_text

    def test_log_contains_plan_table_header(self, caplog):
        """启动日志包含方案表标题 'Routing Plan Table'。"""
        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            load_routing_plugin(config_dir=self.config_dir)

        log_text = caplog.text
        assert "Routing Plan Table" in log_text

    def test_log_contains_all_template_names(self, caplog):
        """启动日志方案表包含全部 4 个模板名称。"""
        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            load_routing_plugin(config_dir=self.config_dir)

        log_text = caplog.text
        assert "resume_screening" in log_text
        assert "code_review" in log_text
        assert "supplier_evaluation" in log_text
        assert "custom_pipeline" in log_text

    def test_log_contains_agent_names(self, caplog):
        """启动日志方案表包含关键 Agent 名称。"""
        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            load_routing_plugin(config_dir=self.config_dir)

        log_text = caplog.text
        # Check representative agents from each template
        assert "intent_classifier" in log_text
        assert "resume_parser" in log_text
        assert "code_analyzer" in log_text
        assert "data_collector" in log_text
        assert "generator" in log_text

    def test_log_contains_model_assignments(self, caplog):
        """启动日志方案表包含模型分配信息。"""
        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            load_routing_plugin(config_dir=self.config_dir)

        log_text = caplog.text
        # The override model should definitely appear
        assert "gpt-5.6-sol" in log_text
        # Total entries count should appear
        assert "Total Entries: 13" in log_text

    def test_fallback_model_from_route_config(self):
        """fallback_model 正确从 route_config.yaml 加载 (deepseek-v4-pro)。"""
        plugin = load_routing_plugin(config_dir=self.config_dir)

        assert plugin.fallback_model == "deepseek-v4-pro"
