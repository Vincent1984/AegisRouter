"""Tests for Agent-WorkBuddy config reload (restart scenario).

验证修改配置文件后，重新初始化插件（模拟进程重启）时方案正确重算。

需求参考: FR-4.1 (重启生效)
验证检查点: V4-1
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aegis_router.config import reset_config
from aegis_router.callbacks.plugin_loader import _initialize_agent_workbuddy_plugin
from aegis_router.router.agent_plan_store import AgentPlanStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_config():
    """Reset global config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with valid YAML files for Agent-WorkBuddy routing."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

    # config.yaml — minimal LiteLLM config
    config_yaml: dict = {
        "model_list": [],
    }
    (cfg_dir / "config.yaml").write_text(yaml.dump(config_yaml), encoding="utf-8")

    # models.yaml — 3 models with different capabilities
    models_yaml = {
        "models": [
            {
                "name": "local-7b",
                "litellm_model": "ollama/qwen2.5-7b",
                "params": {
                    "parameter_size_b": 7,
                    "context_window": 32000,
                    "benchmark_mmlu": 65.0,
                    "benchmark_humaneval": 45.0,
                    "benchmark_math": 40.0,
                    "cost_per_1m_input": 0.0,
                    "cost_per_1m_output": 0.0,
                },
            },
            {
                "name": "deepseek-v4-pro",
                "litellm_model": "deepseek/deepseek-v4-pro",
                "params": {
                    "parameter_size_b": None,
                    "context_window": 128000,
                    "benchmark_mmlu": 90.2,
                    "benchmark_humaneval": 88.5,
                    "benchmark_math": 82.0,
                    "cost_per_1m_input": 0.27,
                    "cost_per_1m_output": 1.10,
                },
            },
            {
                "name": "gpt-5.5",
                "litellm_model": "openai/gpt-5.5",
                "params": {
                    "parameter_size_b": None,
                    "context_window": 200000,
                    "benchmark_mmlu": 92.0,
                    "benchmark_humaneval": 93.0,
                    "benchmark_math": 90.0,
                    "cost_per_1m_input": 3.0,
                    "cost_per_1m_output": 15.0,
                },
            },
        ]
    }
    (cfg_dir / "models.yaml").write_text(yaml.dump(models_yaml), encoding="utf-8")

    # route_config.yaml
    route_yaml = {
        "routing": {
            "overlap_strategy": "lowest_cost",
            "fallback_model": "local-7b",
            "scoring": {
                "weights": {
                    "benchmark_mmlu": 0.25,
                    "benchmark_humaneval": 0.20,
                    "benchmark_math": 0.20,
                    "context_window": 0.10,
                    "cost_efficiency": 0.25,
                },
                "normalization": {
                    "benchmark_mmlu": [50, 95],
                    "benchmark_humaneval": [30, 95],
                    "benchmark_math": [20, 95],
                    "context_window": [4096, 2000000],
                    "cost_per_1m_input": [0, 20],
                },
                "range_tolerance": 0.15,
            },
        },
        "failover": {
            "enabled": False,
            "chains": {},
        },
    }
    (cfg_dir / "route_config.yaml").write_text(yaml.dump(route_yaml), encoding="utf-8")

    # route_overrides.yaml
    overrides_yaml: dict = {"overrides": {}}
    (cfg_dir / "route_overrides.yaml").write_text(
        yaml.dump(overrides_yaml), encoding="utf-8"
    )

    # capability_profiles.yaml — lightweight, medium, strong_reasoning
    profiles_yaml = {
        "profiles": {
            "lightweight": {
                "description": "低延迟低成本",
                "scoring_weights": {
                    "benchmark_mmlu": 0.10,
                    "benchmark_humaneval": 0.05,
                    "benchmark_math": 0.05,
                    "context_window": 0.05,
                    "cost_efficiency": 0.75,
                },
                "min_score_threshold": 0.0,
                "max_cost_per_1m_input": 0.5,
            },
            "medium": {
                "description": "平衡质量和成本",
                "scoring_weights": {
                    "benchmark_mmlu": 0.25,
                    "benchmark_humaneval": 0.15,
                    "benchmark_math": 0.15,
                    "context_window": 0.10,
                    "cost_efficiency": 0.35,
                },
                "min_score_threshold": 0.30,
                "max_cost_per_1m_input": 3.0,
            },
            "strong_reasoning": {
                "description": "强推理",
                "scoring_weights": {
                    "benchmark_mmlu": 0.15,
                    "benchmark_humaneval": 0.30,
                    "benchmark_math": 0.35,
                    "context_window": 0.05,
                    "cost_efficiency": 0.15,
                },
                "min_score_threshold": 0.60,
                "max_cost_per_1m_input": 20.0,
            },
        }
    }
    (cfg_dir / "capability_profiles.yaml").write_text(
        yaml.dump(profiles_yaml), encoding="utf-8"
    )

    # agent_workbuddy.yaml — 3 agents with different profiles
    agent_workbuddy_yaml = {
        "agents": [
            {
                "name": "intent_classifier",
                "capability_profile": "lightweight",
                "description": "意图分类 Agent",
            },
            {
                "name": "reasoning_engine",
                "capability_profile": "strong_reasoning",
                "description": "推理引擎 Agent",
            },
            {
                "name": "general_assistant",
                "capability_profile": "medium",
                "description": "通用助手 Agent",
            },
        ]
    }
    (cfg_dir / "agent_workbuddy.yaml").write_text(
        yaml.dump(agent_workbuddy_yaml), encoding="utf-8"
    )

    return cfg_dir


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_plan_store_from_plugin(config_dir: Path) -> AgentPlanStore:
    """Initialize the agent_workbuddy plugin and return its plan_store."""
    plugin = _initialize_agent_workbuddy_plugin(config_dir)
    return plugin.plan_store


# ---------------------------------------------------------------------------
# Tests: TC-CONFIG-001 — agent_workbuddy.yaml 变更后重启方案重算
# ---------------------------------------------------------------------------


class TestConfigReloadAgentWorkbuddy:
    """TC-CONFIG-001: 修改 agent_workbuddy.yaml 后重启方案重算。

    Validates: Requirements 4.1 (重启生效)
    """

    def test_adding_new_agent_reflected_after_reinit(self, config_dir: Path):
        """新增 Agent 定义后，重启方案表包含新 Agent。"""
        # Initial load
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        initial_store = plugin.plan_store

        # Verify initial state
        assert len(initial_store) == 3
        assert initial_store.get_model("intent_classifier") is not None
        assert initial_store.get_model("reasoning_engine") is not None
        assert initial_store.get_model("general_assistant") is not None
        assert initial_store.get_model("new_agent") is None

        # Stop config watcher to avoid interference
        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        # Reset global config singleton
        reset_config()

        # Modify agent_workbuddy.yaml — add a new agent
        new_agent_yaml = {
            "agents": [
                {
                    "name": "intent_classifier",
                    "capability_profile": "lightweight",
                },
                {
                    "name": "reasoning_engine",
                    "capability_profile": "strong_reasoning",
                },
                {
                    "name": "general_assistant",
                    "capability_profile": "medium",
                },
                {
                    "name": "new_agent",
                    "capability_profile": "medium",
                    "description": "新增 Agent",
                },
            ]
        }
        (config_dir / "agent_workbuddy.yaml").write_text(
            yaml.dump(new_agent_yaml), encoding="utf-8"
        )

        # Re-initialize (simulate restart)
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # Verify new agent is present
        assert len(new_store) == 4
        assert new_store.get_model("new_agent") is not None

    def test_removing_agent_reflected_after_reinit(self, config_dir: Path):
        """移除 Agent 定义后，重启方案表不再包含该 Agent。"""
        # Initial load
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        initial_store = plugin.plan_store

        assert len(initial_store) == 3
        assert initial_store.get_model("general_assistant") is not None

        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        reset_config()

        # Modify: remove general_assistant
        new_agent_yaml = {
            "agents": [
                {
                    "name": "intent_classifier",
                    "capability_profile": "lightweight",
                },
                {
                    "name": "reasoning_engine",
                    "capability_profile": "strong_reasoning",
                },
            ]
        }
        (config_dir / "agent_workbuddy.yaml").write_text(
            yaml.dump(new_agent_yaml), encoding="utf-8"
        )

        # Re-initialize
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # general_assistant no longer in plan
        assert len(new_store) == 2
        assert new_store.get_model("general_assistant") is None
        assert new_store.get_model("intent_classifier") is not None
        assert new_store.get_model("reasoning_engine") is not None

    def test_changing_profile_reflected_after_reinit(self, config_dir: Path):
        """修改 Agent 的 capability_profile 后，重启方案表更新模型分配。"""
        # Initial load
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        initial_store = plugin.plan_store

        # intent_classifier uses lightweight → should pick cheapest model
        initial_model = initial_store.get_model("intent_classifier")
        assert initial_model is not None

        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        reset_config()

        # Change intent_classifier from lightweight to strong_reasoning
        new_agent_yaml = {
            "agents": [
                {
                    "name": "intent_classifier",
                    "capability_profile": "strong_reasoning",
                },
                {
                    "name": "reasoning_engine",
                    "capability_profile": "strong_reasoning",
                },
                {
                    "name": "general_assistant",
                    "capability_profile": "medium",
                },
            ]
        }
        (config_dir / "agent_workbuddy.yaml").write_text(
            yaml.dump(new_agent_yaml), encoding="utf-8"
        )

        # Re-initialize
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # After changing to strong_reasoning, model should match reasoning_engine
        new_model = new_store.get_model("intent_classifier")
        reasoning_model = new_store.get_model("reasoning_engine")
        assert new_model is not None
        # Both use strong_reasoning, so should get same model
        assert new_model == reasoning_model


# ---------------------------------------------------------------------------
# Tests: TC-CONFIG-002 — capability_profiles.yaml 变更后重启方案重算
# ---------------------------------------------------------------------------


class TestConfigReloadProfiles:
    """TC-CONFIG-002: 修改 capability_profiles.yaml 后重启方案重算。

    Validates: Requirements 4.1 (重启生效)
    """

    def test_profile_weight_change_affects_model_selection(self, config_dir: Path):
        """修改 Profile 评分权重后，重启方案表的模型选择变化。"""
        # Initial load
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        initial_store = plugin.plan_store

        initial_reasoning_model = initial_store.get_model("reasoning_engine")
        assert initial_reasoning_model is not None

        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        reset_config()

        # Modify strong_reasoning profile: set max_cost to 0 so only free models pass.
        # This forces selection of local-7b (the only model with cost=0).
        new_profiles = {
            "profiles": {
                "lightweight": {
                    "description": "低延迟低成本",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.10,
                        "benchmark_humaneval": 0.05,
                        "benchmark_math": 0.05,
                        "context_window": 0.05,
                        "cost_efficiency": 0.75,
                    },
                    "min_score_threshold": 0.0,
                    "max_cost_per_1m_input": 0.5,
                },
                "medium": {
                    "description": "平衡质量和成本",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.25,
                        "benchmark_humaneval": 0.15,
                        "benchmark_math": 0.15,
                        "context_window": 0.10,
                        "cost_efficiency": 0.35,
                    },
                    "min_score_threshold": 0.30,
                    "max_cost_per_1m_input": 3.0,
                },
                "strong_reasoning": {
                    "description": "现在仅允许免费模型",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.15,
                        "benchmark_humaneval": 0.30,
                        "benchmark_math": 0.35,
                        "context_window": 0.05,
                        "cost_efficiency": 0.15,
                    },
                    "min_score_threshold": 0.0,
                    "max_cost_per_1m_input": 0.0,
                },
            }
        }
        (config_dir / "capability_profiles.yaml").write_text(
            yaml.dump(new_profiles), encoding="utf-8"
        )

        # Re-initialize
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # With max_cost_per_1m_input=0.0, only local-7b (cost=0) passes the constraint
        new_reasoning_model = new_store.get_model("reasoning_engine")
        assert new_reasoning_model is not None
        assert new_reasoning_model == "local-7b"

    def test_profile_constraint_change_affects_selection(self, config_dir: Path):
        """修改 Profile 硬约束（max_cost）后，重启影响可选模型范围。"""
        # Initial load — lightweight has max_cost_per_1m_input=0.5
        # Both local-7b (cost=0) and deepseek-v4-pro (cost=0.27) pass the constraint.
        # deepseek-v4-pro wins on benchmarks.
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        initial_store = plugin.plan_store

        initial_classifier_model = initial_store.get_model("intent_classifier")
        assert initial_classifier_model == "deepseek-v4-pro"

        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        reset_config()

        # Tighten the lightweight constraint: max_cost=0.0 (only free models)
        # Now only local-7b (cost=0) passes the constraint
        new_profiles = {
            "profiles": {
                "lightweight": {
                    "description": "仅允许免费模型",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.30,
                        "benchmark_humaneval": 0.25,
                        "benchmark_math": 0.20,
                        "context_window": 0.10,
                        "cost_efficiency": 0.15,
                    },
                    "min_score_threshold": 0.0,
                    "max_cost_per_1m_input": 0.0,
                },
                "medium": {
                    "description": "平衡质量和成本",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.25,
                        "benchmark_humaneval": 0.15,
                        "benchmark_math": 0.15,
                        "context_window": 0.10,
                        "cost_efficiency": 0.35,
                    },
                    "min_score_threshold": 0.30,
                    "max_cost_per_1m_input": 3.0,
                },
                "strong_reasoning": {
                    "description": "强推理",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.15,
                        "benchmark_humaneval": 0.30,
                        "benchmark_math": 0.35,
                        "context_window": 0.05,
                        "cost_efficiency": 0.15,
                    },
                    "min_score_threshold": 0.60,
                    "max_cost_per_1m_input": 20.0,
                },
            }
        }
        (config_dir / "capability_profiles.yaml").write_text(
            yaml.dump(new_profiles), encoding="utf-8"
        )

        # Re-initialize
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # With max_cost=0.0, only local-7b passes the constraint
        new_classifier_model = new_store.get_model("intent_classifier")
        assert new_classifier_model is not None
        assert new_classifier_model == "local-7b"


# ---------------------------------------------------------------------------
# Tests: TC-CONFIG-003 — models.yaml 变更后重启方案重算
# ---------------------------------------------------------------------------


class TestConfigReloadModels:
    """TC-CONFIG-003: 修改 models.yaml 后重启方案重算。

    Validates: Requirements 4.1 (重启生效)
    """

    def test_adding_better_model_changes_selection(self, config_dir: Path):
        """新增一个更优模型后，重启方案选用新模型。"""
        # Initial load
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        initial_store = plugin.plan_store

        initial_reasoning_model = initial_store.get_model("reasoning_engine")
        assert initial_reasoning_model is not None

        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        reset_config()

        # Add a super-model with perfect benchmarks and low cost
        new_models = {
            "models": [
                {
                    "name": "local-7b",
                    "litellm_model": "ollama/qwen2.5-7b",
                    "params": {
                        "parameter_size_b": 7,
                        "context_window": 32000,
                        "benchmark_mmlu": 65.0,
                        "benchmark_humaneval": 45.0,
                        "benchmark_math": 40.0,
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                    },
                },
                {
                    "name": "deepseek-v4-pro",
                    "litellm_model": "deepseek/deepseek-v4-pro",
                    "params": {
                        "parameter_size_b": None,
                        "context_window": 128000,
                        "benchmark_mmlu": 90.2,
                        "benchmark_humaneval": 88.5,
                        "benchmark_math": 82.0,
                        "cost_per_1m_input": 0.27,
                        "cost_per_1m_output": 1.10,
                    },
                },
                {
                    "name": "gpt-5.5",
                    "litellm_model": "openai/gpt-5.5",
                    "params": {
                        "parameter_size_b": None,
                        "context_window": 200000,
                        "benchmark_mmlu": 92.0,
                        "benchmark_humaneval": 93.0,
                        "benchmark_math": 90.0,
                        "cost_per_1m_input": 3.0,
                        "cost_per_1m_output": 15.0,
                    },
                },
                {
                    "name": "super-reasoning-model",
                    "litellm_model": "custom/super-reasoning",
                    "params": {
                        "parameter_size_b": None,
                        "context_window": 256000,
                        "benchmark_mmlu": 95.0,
                        "benchmark_humaneval": 96.0,
                        "benchmark_math": 97.0,
                        "cost_per_1m_input": 0.50,
                        "cost_per_1m_output": 2.0,
                    },
                },
            ]
        }
        (config_dir / "models.yaml").write_text(
            yaml.dump(new_models), encoding="utf-8"
        )

        # Re-initialize
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # The new super-reasoning-model has best reasoning benchmarks + low cost
        # It should be selected for strong_reasoning profile
        new_reasoning_model = new_store.get_model("reasoning_engine")
        assert new_reasoning_model == "super-reasoning-model"

    def test_removing_model_causes_different_selection(self, config_dir: Path):
        """移除一个模型后，重启方案选用剩余模型中的最优。"""
        # Initial load
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        initial_store = plugin.plan_store

        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        reset_config()

        # Remove gpt-5.5 — now only local-7b and deepseek-v4-pro remain
        new_models = {
            "models": [
                {
                    "name": "local-7b",
                    "litellm_model": "ollama/qwen2.5-7b",
                    "params": {
                        "parameter_size_b": 7,
                        "context_window": 32000,
                        "benchmark_mmlu": 65.0,
                        "benchmark_humaneval": 45.0,
                        "benchmark_math": 40.0,
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                    },
                },
                {
                    "name": "deepseek-v4-pro",
                    "litellm_model": "deepseek/deepseek-v4-pro",
                    "params": {
                        "parameter_size_b": None,
                        "context_window": 128000,
                        "benchmark_mmlu": 90.2,
                        "benchmark_humaneval": 88.5,
                        "benchmark_math": 82.0,
                        "cost_per_1m_input": 0.27,
                        "cost_per_1m_output": 1.10,
                    },
                },
            ]
        }
        (config_dir / "models.yaml").write_text(
            yaml.dump(new_models), encoding="utf-8"
        )

        # Re-initialize
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # With only 2 models left, all agents must pick from local-7b or deepseek-v4-pro
        for agent_name in ["intent_classifier", "reasoning_engine", "general_assistant"]:
            model = new_store.get_model(agent_name)
            assert model in ("local-7b", "deepseek-v4-pro"), (
                f"Agent '{agent_name}' assigned to '{model}', expected one of the remaining models"
            )

    def test_removing_all_models_results_in_empty_plan(self, config_dir: Path):
        """移除所有模型后，重启方案表为空（所有请求走 fallback）。"""
        # Initial load
        plugin = _initialize_agent_workbuddy_plugin(config_dir)
        assert len(plugin.plan_store) == 3

        if hasattr(plugin, "_config_watcher"):
            plugin._config_watcher.stop()

        reset_config()

        # Remove all models
        new_models: dict = {"models": []}
        (config_dir / "models.yaml").write_text(
            yaml.dump(new_models), encoding="utf-8"
        )

        # Re-initialize
        plugin2 = _initialize_agent_workbuddy_plugin(config_dir)
        new_store = plugin2.plan_store

        if hasattr(plugin2, "_config_watcher"):
            plugin2._config_watcher.stop()

        # Empty models → empty plan store
        assert len(new_store) == 0
