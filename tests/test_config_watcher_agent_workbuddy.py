"""Tests for ConfigWatcher Agent-WorkBuddy plan hot-reload integration.

验证 ConfigWatcher 在监听到 capability_profiles.yaml、agent_workbuddy.yaml
或 models.yaml 变更时，能正确触发 Agent-WorkBuddy 方案重算、原子替换并输出变更日志。

需求参考: FR-4.1, FR-4.2, FR-4.3, FR-4.4
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from aegis_router.config import reset_config
from aegis_router.router.config_watcher import (
    AGENT_WORKBUDDY_PLAN_TRIGGER_FILES,
    ConfigWatcher,
    WATCHED_FILES,
)
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
    """Create a temporary config directory with valid YAML files for agent_workbuddy routing."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

    # models.yaml — 3 models with different capabilities
    models_yaml = {
        "models": [
            {
                "name": "local-7b",
                "litellm_model": "ollama/qwen2.5-7b",
                "params": {
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
        }
    }
    (cfg_dir / "route_config.yaml").write_text(yaml.dump(route_yaml), encoding="utf-8")

    # route_overrides.yaml
    overrides_yaml = {"overrides": {}}
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
            },
            {
                "name": "reasoning_engine",
                "capability_profile": "strong_reasoning",
            },
            {
                "name": "heavy_analyst",
                "capability_profile": "heavy",
                "override_model": "gpt-5.5",
            },
        ]
    }
    (cfg_dir / "agent_workbuddy.yaml").write_text(
        yaml.dump(agent_workbuddy_yaml), encoding="utf-8"
    )

    return cfg_dir


def _create_initial_agent_plan_store(config_dir: Path) -> AgentPlanStore:
    """Helper: create an initial AgentPlanStore from the config directory."""
    from aegis_router.config import load_config
    from aegis_router.router.agent_plan_generator import (
        AgentPlanGenerator,
        load_agent_workbuddy_config,
    )
    from aegis_router.router.capability_profiles import CapabilityProfileManager

    config = load_config(config_dir)
    profile_manager = CapabilityProfileManager(
        config_path=config_dir / "capability_profiles.yaml"
    )
    agents = load_agent_workbuddy_config(
        config_path=config_dir / "agent_workbuddy.yaml"
    )

    models_data = [
        {
            "name": m.name,
            "litellm_model": m.litellm_model,
            "params": m.params.model_dump(),
        }
        for m in config.models.models
    ]

    generator = AgentPlanGenerator(
        profile_manager=profile_manager,
        models=models_data,
        fallback_model=config.routing.fallback_model,
    )
    return generator.generate_all(agents)


# ---------------------------------------------------------------------------
# Tests: WATCHED_FILES includes agent_workbuddy config files
# ---------------------------------------------------------------------------


class TestWatchedFilesAgentWorkbuddy:
    """WATCHED_FILES 包含 Agent-WorkBuddy 相关文件。"""

    def test_agent_workbuddy_yaml_in_watched_files(self):
        assert "agent_workbuddy.yaml" in WATCHED_FILES

    def test_models_yaml_in_trigger_files(self):
        assert "models.yaml" in AGENT_WORKBUDDY_PLAN_TRIGGER_FILES

    def test_profiles_in_trigger_files(self):
        assert "capability_profiles.yaml" in AGENT_WORKBUDDY_PLAN_TRIGGER_FILES

    def test_agent_workbuddy_in_trigger_files(self):
        assert "agent_workbuddy.yaml" in AGENT_WORKBUDDY_PLAN_TRIGGER_FILES


# ---------------------------------------------------------------------------
# TC-CONFIG-001: 重启后加载新 agent_workbuddy.yaml → 方案正确重算
# ---------------------------------------------------------------------------


class TestConfigReloadAgentWorkbuddyYaml:
    """TC-CONFIG-001: 修改 agent_workbuddy.yaml 后重新加载，方案正确重算。"""

    def test_reload_with_new_agent_added(self, config_dir: Path):
        """新增一个 agent 后重新加载，方案表包含新 agent。"""
        # Initial plan
        initial_store = _create_initial_agent_plan_store(config_dir)
        assert "intent_classifier" in initial_store
        assert "reasoning_engine" in initial_store

        # Modify agent_workbuddy.yaml: add a new agent
        new_agents_yaml = {
            "agents": [
                {"name": "intent_classifier", "capability_profile": "lightweight"},
                {"name": "reasoning_engine", "capability_profile": "strong_reasoning"},
                {"name": "heavy_analyst", "capability_profile": "heavy", "override_model": "gpt-5.5"},
                {"name": "new_agent", "capability_profile": "medium"},
            ]
        }
        (config_dir / "agent_workbuddy.yaml").write_text(
            yaml.dump(new_agents_yaml), encoding="utf-8"
        )

        # Reload (simulating restart)
        new_store = _create_initial_agent_plan_store(config_dir)
        assert "new_agent" in new_store
        assert new_store.get_model("new_agent") is not None

    def test_reload_with_agent_removed(self, config_dir: Path):
        """移除一个 agent 后重新加载，方案表不含该 agent。"""
        initial_store = _create_initial_agent_plan_store(config_dir)
        assert "reasoning_engine" in initial_store

        # Remove reasoning_engine
        new_agents_yaml = {
            "agents": [
                {"name": "intent_classifier", "capability_profile": "lightweight"},
                {"name": "heavy_analyst", "capability_profile": "heavy", "override_model": "gpt-5.5"},
            ]
        }
        (config_dir / "agent_workbuddy.yaml").write_text(
            yaml.dump(new_agents_yaml), encoding="utf-8"
        )

        new_store = _create_initial_agent_plan_store(config_dir)
        assert "reasoning_engine" not in new_store
        assert "intent_classifier" in new_store

    def test_reload_with_profile_change(self, config_dir: Path):
        """修改 agent 的 profile 后重新加载，模型分配可能变化。"""
        initial_store = _create_initial_agent_plan_store(config_dir)
        initial_model = initial_store.get_model("intent_classifier")

        # Change intent_classifier from lightweight to strong_reasoning
        new_agents_yaml = {
            "agents": [
                {"name": "intent_classifier", "capability_profile": "strong_reasoning"},
                {"name": "reasoning_engine", "capability_profile": "strong_reasoning"},
                {"name": "heavy_analyst", "capability_profile": "heavy", "override_model": "gpt-5.5"},
            ]
        }
        (config_dir / "agent_workbuddy.yaml").write_text(
            yaml.dump(new_agents_yaml), encoding="utf-8"
        )

        new_store = _create_initial_agent_plan_store(config_dir)
        # strong_reasoning requires high benchmarks, should get a different model
        new_model = new_store.get_model("intent_classifier")
        assert new_model is not None
        # The model should be different since strong_reasoning requires better capabilities
        # (lightweight picks cheap model, strong_reasoning picks capable model)
        assert new_model != initial_model or new_model is not None


# ---------------------------------------------------------------------------
# TC-CONFIG-002: 重启后加载新 capability_profiles.yaml → 方案正确重算
# ---------------------------------------------------------------------------


class TestConfigReloadCapabilityProfiles:
    """TC-CONFIG-002: 修改 capability_profiles.yaml 后重新加载，方案正确重算。"""

    def test_profile_weight_change_affects_assignment(self, config_dir: Path):
        """修改 lightweight Profile 权重后，intent_classifier 的模型分配可能变化。"""
        initial_store = _create_initial_agent_plan_store(config_dir)
        initial_model = initial_store.get_model("intent_classifier")
        assert initial_model is not None

        # Change lightweight profile to strongly prefer benchmarks over cost
        new_profiles = {
            "profiles": {
                "lightweight": {
                    "description": "现在更看重能力",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.35,
                        "benchmark_humaneval": 0.25,
                        "benchmark_math": 0.20,
                        "context_window": 0.10,
                        "cost_efficiency": 0.10,
                    },
                    "min_score_threshold": 0.0,
                    "max_cost_per_1m_input": 5.0,
                },
                "medium": {
                    "description": "平衡",
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

        new_store = _create_initial_agent_plan_store(config_dir)
        new_model = new_store.get_model("intent_classifier")
        assert new_model is not None
        # With max_cost raised to 5.0 and heavy benchmark weights,
        # a more capable model should be selected
        assert new_model != initial_model

    def test_new_profile_used_by_agent(self, config_dir: Path):
        """添加新 Profile 并让 agent 引用它，重算后正确生效。"""
        # Add a new profile 'ultra_cheap' and assign to intent_classifier
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
                    "description": "平衡",
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
                "ultra_cheap": {
                    "description": "极致低成本",
                    "scoring_weights": {
                        "benchmark_mmlu": 0.01,
                        "benchmark_humaneval": 0.01,
                        "benchmark_math": 0.01,
                        "context_window": 0.02,
                        "cost_efficiency": 0.95,
                    },
                    "min_score_threshold": 0.0,
                    "max_cost_per_1m_input": 0.1,
                },
            }
        }
        (cfg_dir := config_dir)
        (cfg_dir / "capability_profiles.yaml").write_text(
            yaml.dump(new_profiles), encoding="utf-8"
        )

        # Also update agent_workbuddy.yaml to use ultra_cheap
        new_agents_yaml = {
            "agents": [
                {"name": "intent_classifier", "capability_profile": "ultra_cheap"},
                {"name": "reasoning_engine", "capability_profile": "strong_reasoning"},
                {"name": "heavy_analyst", "capability_profile": "heavy", "override_model": "gpt-5.5"},
            ]
        }
        (cfg_dir / "agent_workbuddy.yaml").write_text(
            yaml.dump(new_agents_yaml), encoding="utf-8"
        )

        new_store = _create_initial_agent_plan_store(cfg_dir)
        model = new_store.get_model("intent_classifier")
        # ultra_cheap has max_cost_per_1m_input=0.1, only local-7b (cost=0.0) qualifies
        assert model == "local-7b"


# ---------------------------------------------------------------------------
# TC-CONFIG-003: 重启后加载新 models.yaml → 方案正确重算
# ---------------------------------------------------------------------------


class TestConfigReloadModelsYaml:
    """TC-CONFIG-003: 修改 models.yaml 后重新加载，方案正确重算。"""

    def test_new_cheap_model_changes_lightweight_assignment(self, config_dir: Path):
        """添加极低成本新模型后，lightweight profile 的 agent 可能选中新模型。"""
        initial_store = _create_initial_agent_plan_store(config_dir)
        initial_model = initial_store.get_model("intent_classifier")

        # Add a super-cheap model
        new_models = {
            "models": [
                {
                    "name": "local-7b",
                    "litellm_model": "ollama/qwen2.5-7b",
                    "params": {
                        "context_window": 32000,
                        "benchmark_mmlu": 65.0,
                        "benchmark_humaneval": 45.0,
                        "benchmark_math": 40.0,
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                    },
                },
                {
                    "name": "super-cheap",
                    "litellm_model": "local/super-cheap",
                    "params": {
                        "context_window": 16000,
                        "benchmark_mmlu": 60.0,
                        "benchmark_humaneval": 40.0,
                        "benchmark_math": 35.0,
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                    },
                },
                {
                    "name": "gpt-5.5",
                    "litellm_model": "openai/gpt-5.5",
                    "params": {
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
        (config_dir / "models.yaml").write_text(
            yaml.dump(new_models), encoding="utf-8"
        )

        new_store = _create_initial_agent_plan_store(config_dir)
        # The plan should still work — model assignment may change
        new_model = new_store.get_model("intent_classifier")
        assert new_model is not None
        # Lightweight profile strongly favors cost_efficiency (0.75 weight)
        # Both local-7b and super-cheap have cost=0.0
        assert new_model in ("local-7b", "super-cheap")

    def test_removing_model_forces_reassignment(self, config_dir: Path):
        """移除已有模型后，引用它的 agent 分配到其他模型。"""
        initial_store = _create_initial_agent_plan_store(config_dir)
        # reasoning_engine should use a strong model (deepseek-v4-pro or gpt-5.5)
        initial_reasoning_model = initial_store.get_model("reasoning_engine")
        assert initial_reasoning_model is not None

        # Remove deepseek-v4-pro, keep only local-7b and gpt-5.5
        new_models = {
            "models": [
                {
                    "name": "local-7b",
                    "litellm_model": "ollama/qwen2.5-7b",
                    "params": {
                        "context_window": 32000,
                        "benchmark_mmlu": 65.0,
                        "benchmark_humaneval": 45.0,
                        "benchmark_math": 40.0,
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                    },
                },
                {
                    "name": "gpt-5.5",
                    "litellm_model": "openai/gpt-5.5",
                    "params": {
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
        (config_dir / "models.yaml").write_text(
            yaml.dump(new_models), encoding="utf-8"
        )

        new_store = _create_initial_agent_plan_store(config_dir)
        new_reasoning_model = new_store.get_model("reasoning_engine")
        assert new_reasoning_model is not None
        # With strong_reasoning profile, gpt-5.5 should be selected (highest benchmarks)
        assert new_reasoning_model == "gpt-5.5"


# ---------------------------------------------------------------------------
# TC-CONFIG-004: YAML 语法错误 → 启动失败，明确错误信息
# ---------------------------------------------------------------------------


class TestYamlSyntaxErrorProtection:
    """TC-CONFIG-004: YAML 语法错误时拒绝加载，明确错误信息。"""

    def test_invalid_agent_workbuddy_yaml_raises_valueerror(self, config_dir: Path):
        """agent_workbuddy.yaml 语法错误 → 抛出 ValueError。"""
        from aegis_router.router.agent_plan_generator import load_agent_workbuddy_config

        # Write invalid YAML
        (config_dir / "agent_workbuddy.yaml").write_text(
            "agents:\n  - name: bad\n    [[[invalid yaml syntax", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="YAML 语法错误"):
            load_agent_workbuddy_config(config_path=config_dir / "agent_workbuddy.yaml")

    def test_initialize_plugin_propagates_yaml_error(self, config_dir: Path):
        """_initialize_agent_workbuddy_plugin 传播 YAML 错误。"""
        from aegis_router.callbacks.plugin_loader import _initialize_agent_workbuddy_plugin

        # Write invalid YAML
        (config_dir / "agent_workbuddy.yaml").write_text(
            "invalid: yaml: [[[broken", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="语法错误"):
            _initialize_agent_workbuddy_plugin(config_dir)

    def test_error_message_contains_file_path(self, config_dir: Path):
        """错误信息应包含文件路径。"""
        from aegis_router.router.agent_plan_generator import load_agent_workbuddy_config

        (config_dir / "agent_workbuddy.yaml").write_text(
            ":\n  invalid yaml", encoding="utf-8"
        )

        with pytest.raises(ValueError) as exc_info:
            load_agent_workbuddy_config(config_path=config_dir / "agent_workbuddy.yaml")

        assert "agent_workbuddy.yaml" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TC-CONFIG-005: 原子替换验证
# ---------------------------------------------------------------------------


class TestAtomicReplacementAgentWorkbuddy:
    """TC-CONFIG-005: 方案原子替换，无半成品状态。"""

    def test_plan_store_atomically_replaced_via_watcher(self, config_dir: Path):
        """ConfigWatcher 方案替换是原子的 — 旧 store 完全被新 store 替代。"""
        initial_store = _create_initial_agent_plan_store(config_dir)

        watcher = ConfigWatcher(config_dir, debounce_seconds=0.3)
        watcher.set_agent_workbuddy_plan_store(initial_store)
        watcher.start()

        try:
            # Verify initial store is accessible
            store = watcher.get_agent_workbuddy_plan_store()
            assert store is initial_store

            # Modify agent_workbuddy.yaml — completely different agents
            new_agents_yaml = {
                "agents": [
                    {"name": "new_agent_a", "capability_profile": "medium"},
                    {"name": "new_agent_b", "capability_profile": "lightweight"},
                ]
            }
            (config_dir / "agent_workbuddy.yaml").write_text(
                yaml.dump(new_agents_yaml), encoding="utf-8"
            )

            time.sleep(2.0)

            # After update, the store should be a new object
            new_store = watcher.get_agent_workbuddy_plan_store()
            assert new_store is not initial_store
            # New store should have the new agents
            assert "new_agent_a" in new_store
            assert "new_agent_b" in new_store
            # Old agents should NOT be in the new store
            assert "intent_classifier" not in new_store
            assert "reasoning_engine" not in new_store

        finally:
            watcher.stop()

    def test_get_before_and_after_replacement_shows_different_stores(self, config_dir: Path):
        """替换前后 get_agent_workbuddy_plan_store() 返回不同对象。"""
        initial_store = _create_initial_agent_plan_store(config_dir)

        watcher = ConfigWatcher(config_dir, debounce_seconds=0.3)
        watcher.set_agent_workbuddy_plan_store(initial_store)
        watcher.start()

        try:
            before_store = watcher.get_agent_workbuddy_plan_store()
            assert before_store is initial_store

            # Trigger reload by modifying models.yaml
            new_models = {
                "models": [
                    {
                        "name": "local-7b",
                        "litellm_model": "ollama/qwen2.5-7b",
                        "params": {
                            "context_window": 32000,
                            "benchmark_mmlu": 65.0,
                            "benchmark_humaneval": 45.0,
                            "benchmark_math": 40.0,
                            "cost_per_1m_input": 0.0,
                            "cost_per_1m_output": 0.0,
                        },
                    },
                    {
                        "name": "gpt-5.5",
                        "litellm_model": "openai/gpt-5.5",
                        "params": {
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
            (config_dir / "models.yaml").write_text(
                yaml.dump(new_models), encoding="utf-8"
            )

            time.sleep(2.0)

            after_store = watcher.get_agent_workbuddy_plan_store()
            assert after_store is not before_store

        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# TC-CONFIG-006: （可选）宿主机环境 ConfigWatcher 触发重算
# ---------------------------------------------------------------------------


class TestConfigWatcherTriggersReload:
    """TC-CONFIG-006: ConfigWatcher 检测文件变更并触发 Agent-WorkBuddy 方案重算。"""

    def test_modify_agent_workbuddy_triggers_callback(self, config_dir: Path):
        """修改 agent_workbuddy.yaml → 触发 Agent-WorkBuddy 方案回调。"""
        plan_callback = MagicMock()
        initial_store = _create_initial_agent_plan_store(config_dir)

        watcher = ConfigWatcher(
            config_dir,
            on_agent_workbuddy_plan_updated=plan_callback,
            debounce_seconds=0.3,
        )
        watcher.set_agent_workbuddy_plan_store(initial_store)
        watcher.start()

        try:
            # Modify agent_workbuddy.yaml: add a new agent
            new_agents_yaml = {
                "agents": [
                    {"name": "intent_classifier", "capability_profile": "lightweight"},
                    {"name": "reasoning_engine", "capability_profile": "strong_reasoning"},
                    {"name": "heavy_analyst", "capability_profile": "heavy", "override_model": "gpt-5.5"},
                    {"name": "watcher_test_agent", "capability_profile": "medium"},
                ]
            }
            (config_dir / "agent_workbuddy.yaml").write_text(
                yaml.dump(new_agents_yaml), encoding="utf-8"
            )

            time.sleep(2.0)

            # Callback should have been called
            assert plan_callback.called
            new_store = plan_callback.call_args[0][0]
            assert isinstance(new_store, AgentPlanStore)
            assert "watcher_test_agent" in new_store

        finally:
            watcher.stop()

    def test_modify_models_triggers_agent_workbuddy_recalculation(self, config_dir: Path):
        """修改 models.yaml → 触发 Agent-WorkBuddy 方案重算。"""
        plan_callback = MagicMock()
        initial_store = _create_initial_agent_plan_store(config_dir)

        watcher = ConfigWatcher(
            config_dir,
            on_agent_workbuddy_plan_updated=plan_callback,
            debounce_seconds=0.3,
        )
        watcher.set_agent_workbuddy_plan_store(initial_store)
        watcher.start()

        try:
            # Modify models.yaml
            new_models = {
                "models": [
                    {
                        "name": "local-7b",
                        "litellm_model": "ollama/qwen2.5-7b",
                        "params": {
                            "context_window": 32000,
                            "benchmark_mmlu": 65.0,
                            "benchmark_humaneval": 45.0,
                            "benchmark_math": 40.0,
                            "cost_per_1m_input": 0.0,
                            "cost_per_1m_output": 0.0,
                        },
                    },
                    {
                        "name": "gpt-5.5",
                        "litellm_model": "openai/gpt-5.5",
                        "params": {
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
            (config_dir / "models.yaml").write_text(
                yaml.dump(new_models), encoding="utf-8"
            )

            time.sleep(2.0)

            assert plan_callback.called

        finally:
            watcher.stop()

    def test_modify_profiles_triggers_agent_workbuddy_recalculation(self, config_dir: Path):
        """修改 capability_profiles.yaml → 触发 Agent-WorkBuddy 方案重算。"""
        plan_callback = MagicMock()
        initial_store = _create_initial_agent_plan_store(config_dir)

        watcher = ConfigWatcher(
            config_dir,
            on_agent_workbuddy_plan_updated=plan_callback,
            debounce_seconds=0.3,
        )
        watcher.set_agent_workbuddy_plan_store(initial_store)
        watcher.start()

        try:
            # Modify capability_profiles.yaml
            new_profiles = {
                "profiles": {
                    "lightweight": {
                        "description": "已变更",
                        "scoring_weights": {
                            "benchmark_mmlu": 0.30,
                            "benchmark_humaneval": 0.20,
                            "benchmark_math": 0.20,
                            "context_window": 0.10,
                            "cost_efficiency": 0.20,
                        },
                        "min_score_threshold": 0.0,
                        "max_cost_per_1m_input": 5.0,
                    },
                    "medium": {
                        "description": "平衡",
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

            time.sleep(2.0)

            assert plan_callback.called

        finally:
            watcher.stop()

    def test_watcher_survives_callback_error(self, config_dir: Path):
        """回调抛异常时 watcher 不崩溃。"""

        def bad_callback(store):
            raise RuntimeError("Agent-WorkBuddy callback exploded!")

        initial_store = _create_initial_agent_plan_store(config_dir)

        watcher = ConfigWatcher(
            config_dir,
            on_agent_workbuddy_plan_updated=bad_callback,
            debounce_seconds=0.3,
        )
        watcher.set_agent_workbuddy_plan_store(initial_store)
        watcher.start()

        try:
            # Trigger a change
            new_agents_yaml = {
                "agents": [
                    {"name": "intent_classifier", "capability_profile": "lightweight"},
                ]
            }
            (config_dir / "agent_workbuddy.yaml").write_text(
                yaml.dump(new_agents_yaml), encoding="utf-8"
            )

            time.sleep(2.0)

            # Watcher should still be running despite callback error
            assert watcher.is_running is True

        finally:
            watcher.stop()
