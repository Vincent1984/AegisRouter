"""Verification test V5-1: 启动日志包含完整方案表（各模板各 Agent 对应模型）

验证:
- 当 load_routing_plugin(config_dir) 启动 transaction 插件时，日志包含 "Routing Plan Table" 标题
- 日志中包含每个模板名
- 日志中包含每个 Agent 名
- 日志中包含每个 Agent 对应的模型名
- 日志中包含 fallback 模型
- 方案表格式包含表头 (Template / Agent / Model)
- 审计日志发出 plan_generation 事件（每个模板一次）

需求参考: FR-8.1, FR-9.1
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest
import yaml

from aegis_router.callbacks.plugin_loader import _log_plan_table, load_routing_plugin
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_multi_template_config_dir() -> str:
    """Create a temp config directory with multiple templates for thorough testing.

    Templates:
      - resume_screening: 3 agents (lightweight, medium, strong_reasoning)
      - code_review: 2 agents (code_specialist, strong_reasoning)
      - custom_pipeline: 1 agent with override_model

    Returns the temp directory path.
    """
    tmpdir = tempfile.mkdtemp()
    config_dir = Path(tmpdir)

    # config.yaml
    config_data = {
        "routing_plugin": "transaction",
        "model_list": [],
    }
    (config_dir / "config.yaml").write_text(
        yaml.dump(config_data), encoding="utf-8"
    )

    # models.yaml
    models_data = {
        "models": [
            {
                "name": "local-7b",
                "litellm_model": "ollama/local-7b",
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

    # transaction_templates.yaml — multiple templates
    templates_data = {
        "templates": {
            "resume_screening": {
                "description": "简历筛选流程",
                "agents": [
                    {
                        "name": "intent_classifier",
                        "capability_profile": "lightweight",
                    },
                    {
                        "name": "resume_parser",
                        "capability_profile": "medium",
                    },
                    {
                        "name": "skill_matcher",
                        "capability_profile": "strong_reasoning",
                    },
                ],
            },
            "code_review": {
                "description": "代码审查流程",
                "agents": [
                    {
                        "name": "code_analyzer",
                        "capability_profile": "strong_reasoning",
                    },
                    {
                        "name": "fix_suggester",
                        "capability_profile": "medium",
                    },
                ],
            },
            "custom_pipeline": {
                "description": "自定义流程",
                "agents": [
                    {
                        "name": "generator",
                        "capability_profile": "medium",
                        "override_model": "strong-model",
                    },
                ],
            },
        }
    }
    (config_dir / "transaction_templates.yaml").write_text(
        yaml.dump(templates_data), encoding="utf-8"
    )

    # capability_profiles.yaml — not created, will use built-in defaults

    return tmpdir


# ---------------------------------------------------------------------------
# Tests: _log_plan_table 直接调用
# ---------------------------------------------------------------------------


class TestLogPlanTableDirect:
    """Test _log_plan_table() function directly with a pre-built RoutingPlanStore."""

    def test_log_contains_plan_table_header(self, caplog):
        """日志包含 'Transaction Router - Routing Plan Table' 标题。"""
        store = RoutingPlanStore()
        store.set_model("template_a", "agent_1", "model_x")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fallback-model")

        assert "Transaction Router - Routing Plan Table" in caplog.text

    def test_log_contains_fallback_model(self, caplog):
        """日志包含 fallback 模型名称。"""
        store = RoutingPlanStore()
        store.set_model("tpl", "ag", "m")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "my-fallback-model")

        assert "my-fallback-model" in caplog.text

    def test_log_contains_total_entries(self, caplog):
        """日志包含方案总条目数。"""
        store = RoutingPlanStore()
        store.set_model("tpl_a", "agent_1", "model_1")
        store.set_model("tpl_a", "agent_2", "model_2")
        store.set_model("tpl_b", "agent_3", "model_3")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fallback")

        assert "Total Entries: 3" in caplog.text

    def test_log_contains_table_column_headers(self, caplog):
        """日志包含表头列名 (Template, Agent, Model)。"""
        store = RoutingPlanStore()
        store.set_model("tpl", "ag", "m")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fb")

        log_text = caplog.text
        assert "Template" in log_text
        assert "Agent" in log_text
        assert "Model" in log_text

    def test_log_contains_all_template_names(self, caplog):
        """日志包含所有模板名称。"""
        store = RoutingPlanStore()
        store.set_model("resume_screening", "agent_a", "model_1")
        store.set_model("code_review", "agent_b", "model_2")
        store.set_model("supplier_evaluation", "agent_c", "model_3")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fallback")

        log_text = caplog.text
        assert "resume_screening" in log_text
        assert "code_review" in log_text
        assert "supplier_evaluation" in log_text

    def test_log_contains_all_agent_names(self, caplog):
        """日志包含所有 Agent 名称。"""
        store = RoutingPlanStore()
        store.set_model("tpl", "intent_classifier", "m1")
        store.set_model("tpl", "resume_parser", "m2")
        store.set_model("tpl", "skill_matcher", "m3")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fallback")

        log_text = caplog.text
        assert "intent_classifier" in log_text
        assert "resume_parser" in log_text
        assert "skill_matcher" in log_text

    def test_log_contains_all_model_names(self, caplog):
        """日志包含所有已分配的模型名称。"""
        store = RoutingPlanStore()
        store.set_model("tpl", "ag1", "local-7b")
        store.set_model("tpl", "ag2", "gemini-2.5-pro")
        store.set_model("tpl", "ag3", "gpt-5.5")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fallback")

        log_text = caplog.text
        assert "local-7b" in log_text
        assert "gemini-2.5-pro" in log_text
        assert "gpt-5.5" in log_text

    def test_log_each_template_each_agent_has_model(self, caplog):
        """验证: 对于每个模板定义的每个 Agent，日志中能找到对应的模型分配行。"""
        store = RoutingPlanStore()
        expected_assignments = {
            ("resume_screening", "intent_classifier", "local-7b"),
            ("resume_screening", "resume_parser", "mid-model"),
            ("code_review", "code_analyzer", "strong-model"),
            ("code_review", "fix_suggester", "mid-model"),
        }
        for tpl, agent, model in expected_assignments:
            store.set_model(tpl, agent, model)

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fallback")

        log_text = caplog.text
        for tpl, agent, model in expected_assignments:
            assert tpl in log_text, f"Template '{tpl}' not found in log"
            assert agent in log_text, f"Agent '{agent}' not found in log"
            assert model in log_text, f"Model '{model}' not found in log"

    def test_empty_plan_store_logs_empty_message(self, caplog):
        """方案表为空时，日志显示 '方案表为空'。"""
        store = RoutingPlanStore()

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "default-fallback")

        log_text = caplog.text
        assert "方案表为空" in log_text
        assert "default-fallback" in log_text

    def test_log_format_has_separator_lines(self, caplog):
        """日志格式包含分隔线（'=' 和 '-' 行）。"""
        store = RoutingPlanStore()
        store.set_model("tpl", "ag", "m")

        with caplog.at_level(logging.INFO, logger="aegis_router.callbacks.plugin_loader"):
            _log_plan_table(store, "fb")

        log_text = caplog.text
        assert "=" * 70 in log_text
        assert "-" * 70 in log_text


# ---------------------------------------------------------------------------
# Tests: 通过 load_routing_plugin 完整启动流程验证
# ---------------------------------------------------------------------------


class TestStartupPlanLoggingIntegration:
    """Test startup plan logging through the full load_routing_plugin() flow."""

    def test_startup_log_contains_plan_table_header(self, caplog):
        """完整启动流程: 日志包含方案表标题。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO):
            load_routing_plugin(config_dir=tmpdir)

        assert "Transaction Router - Routing Plan Table" in caplog.text

    def test_startup_log_contains_all_templates(self, caplog):
        """完整启动流程: 日志包含所有模板名称。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO):
            load_routing_plugin(config_dir=tmpdir)

        log_text = caplog.text
        assert "resume_screening" in log_text
        assert "code_review" in log_text
        assert "custom_pipeline" in log_text

    def test_startup_log_contains_all_agents(self, caplog):
        """完整启动流程: 日志包含所有 Agent 名称。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO):
            load_routing_plugin(config_dir=tmpdir)

        log_text = caplog.text
        # resume_screening agents
        assert "intent_classifier" in log_text
        assert "resume_parser" in log_text
        assert "skill_matcher" in log_text
        # code_review agents
        assert "code_analyzer" in log_text
        assert "fix_suggester" in log_text
        # custom_pipeline agents
        assert "generator" in log_text

    def test_startup_log_contains_assigned_models(self, caplog):
        """完整启动流程: 日志包含各 Agent 分配的模型名称。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO):
            plugin = load_routing_plugin(config_dir=tmpdir)

        log_text = caplog.text

        # 验证 plan_store 中的每个分配都出现在日志中
        all_plans = plugin.plan_store.get_all_plans()
        for template_name, agent_map in all_plans.items():
            for agent_name, model in agent_map.items():
                assert template_name in log_text, (
                    f"Template '{template_name}' not in startup log"
                )
                assert agent_name in log_text, (
                    f"Agent '{agent_name}' not in startup log"
                )
                assert model in log_text, (
                    f"Model '{model}' (assigned to {template_name}/{agent_name}) "
                    f"not in startup log"
                )

    def test_startup_log_contains_fallback_model(self, caplog):
        """完整启动流程: 日志包含 fallback 模型名称。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO):
            load_routing_plugin(config_dir=tmpdir)

        # route_config.yaml 中配置 fallback_model: mid-model
        assert "mid-model" in caplog.text

    def test_startup_log_override_model_visible(self, caplog):
        """完整启动流程: override_model 分配的模型出现在日志中。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO):
            plugin = load_routing_plugin(config_dir=tmpdir)

        # custom_pipeline/generator 有 override_model: strong-model
        model = plugin.plan_store.get_model("custom_pipeline", "generator")
        assert model == "strong-model"
        assert "strong-model" in caplog.text

    def test_audit_logger_emits_plan_generation_events(self, caplog):
        """完整启动流程: 审计日志为每个模板发出 plan_generation 事件。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            load_routing_plugin(config_dir=tmpdir)

        # 收集所有审计 plan_generation 事件
        audit_records = [
            r for r in caplog.records
            if r.name == "aegis_router.audit"
        ]
        plan_events = []
        for record in audit_records:
            try:
                parsed = json.loads(record.getMessage())
                if parsed.get("event") == "plan_generation":
                    plan_events.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

        # 应为 3 个模板各产生一个 plan_generation 事件
        template_names_logged = {e["template_name"] for e in plan_events}
        assert "resume_screening" in template_names_logged
        assert "code_review" in template_names_logged
        assert "custom_pipeline" in template_names_logged

    def test_audit_plan_generation_contains_assignments(self, caplog):
        """审计 plan_generation 事件包含各 Agent 的模型分配。"""
        tmpdir = _create_multi_template_config_dir()

        with caplog.at_level(logging.INFO, logger="aegis_router.audit"):
            plugin = load_routing_plugin(config_dir=tmpdir)

        # 获取审计事件
        audit_records = [
            r for r in caplog.records
            if r.name == "aegis_router.audit"
        ]
        plan_events = {}
        for record in audit_records:
            try:
                parsed = json.loads(record.getMessage())
                if parsed.get("event") == "plan_generation":
                    plan_events[parsed["template_name"]] = parsed
            except (json.JSONDecodeError, TypeError):
                continue

        # 验证 resume_screening 模板的审计事件包含所有 agent 分配
        rs_event = plan_events.get("resume_screening")
        assert rs_event is not None, "No plan_generation event for resume_screening"
        assert "intent_classifier" in rs_event["assignments"]
        assert "resume_parser" in rs_event["assignments"]
        assert "skill_matcher" in rs_event["assignments"]

        # 验证 custom_pipeline 中 override_model 也被记录
        cp_event = plan_events.get("custom_pipeline")
        assert cp_event is not None, "No plan_generation event for custom_pipeline"
        assert cp_event["assignments"]["generator"] == "strong-model"
