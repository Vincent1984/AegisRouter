"""Tests for aegis_router.config module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from aegis_router.config import (
    AegisConfig,
    load_config,
    get_config,
    reload_config,
    reset_config,
    resolve_env_vars,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_config():
    """Reset global config singleton before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with sample YAML files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # config.yaml
    config_yaml = {
        "model_list": [
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test",
                    "api_key": "os.environ/TEST_API_KEY",
                },
            }
        ],
        "litellm_settings": {"callbacks": "aegis_router.callbacks.smart_router_instance"},
        "general_settings": {"master_key": "os.environ/TEST_MASTER_KEY"},
    }
    (config_dir / "config.yaml").write_text(yaml.dump(config_yaml), encoding="utf-8")

    # models.yaml
    models_yaml = {
        "models": [
            {
                "name": "test-7b",
                "litellm_model": "ollama/test-7b",
                "params": {
                    "parameter_size_b": 7,
                    "context_window": 32000,
                    "benchmark_mmlu": 65.0,
                    "benchmark_humaneval": 45.0,
                    "benchmark_math": 40.0,
                    "cost_per_1m_input": 0.0,
                    "cost_per_1m_output": 0.0,
                },
            }
        ]
    }
    (config_dir / "models.yaml").write_text(yaml.dump(models_yaml), encoding="utf-8")

    # route_config.yaml
    route_yaml = {
        "routing": {
            "score_input": "masked",
            "trivial": {"enabled": True, "max_length": 30, "target_model": "test-7b"},
            "classifier": {"type": "mf"},
            "overlap_strategy": "lowest_cost",
            "fallback_model": "test-fallback",
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
    (config_dir / "route_config.yaml").write_text(yaml.dump(route_yaml), encoding="utf-8")

    # route_overrides.yaml
    overrides_yaml = {
        "overrides": {
            "test-7b": {
                "score_range": [0.0, 0.18],
                "reason": "test override",
            }
        }
    }
    (config_dir / "route_overrides.yaml").write_text(
        yaml.dump(overrides_yaml), encoding="utf-8"
    )

    return config_dir


# ---------------------------------------------------------------------------
# Tests: import and basic call
# ---------------------------------------------------------------------------


class TestLoadConfigImport:
    """Verify load_config can be imported and called."""

    def test_import_load_config(self):
        from aegis_router.config import load_config
        assert callable(load_config)

    def test_load_config_returns_aegis_config(self, tmp_config_dir: Path):
        cfg = load_config(tmp_config_dir)
        assert isinstance(cfg, AegisConfig)


# ---------------------------------------------------------------------------
# Tests: config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Verify config loads valid YAML files correctly."""

    def test_litellm_config_loaded(self, tmp_config_dir: Path):
        cfg = load_config(tmp_config_dir)
        assert len(cfg.litellm.model_list) == 1
        assert cfg.litellm.model_list[0].model_name == "test-model"
        assert cfg.litellm.model_list[0].litellm_params.model == "openai/test"

    def test_models_config_loaded(self, tmp_config_dir: Path):
        cfg = load_config(tmp_config_dir)
        assert len(cfg.models.models) == 1
        assert cfg.models.models[0].name == "test-7b"
        assert cfg.models.models[0].params.benchmark_mmlu == 65.0

    def test_routing_config_loaded(self, tmp_config_dir: Path):
        cfg = load_config(tmp_config_dir)
        assert cfg.routing.overlap_strategy == "lowest_cost"
        assert cfg.routing.fallback_model == "test-fallback"
        assert cfg.routing.trivial.enabled is True
        assert cfg.routing.scoring.range_tolerance == 0.15

    def test_overrides_config_loaded(self, tmp_config_dir: Path):
        cfg = load_config(tmp_config_dir)
        assert "test-7b" in cfg.overrides.overrides
        override = cfg.overrides.overrides["test-7b"]
        assert override.score_range == [0.0, 0.18]
        assert override.reason == "test override"

    def test_config_dir_recorded(self, tmp_config_dir: Path):
        cfg = load_config(tmp_config_dir)
        assert cfg.config_dir == str(tmp_config_dir)


# ---------------------------------------------------------------------------
# Tests: environment variable resolution
# ---------------------------------------------------------------------------


class TestEnvVarResolution:
    """Verify os.environ/VARIABLE_NAME patterns are resolved."""

    def test_resolve_env_var_set(self, tmp_config_dir: Path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        monkeypatch.setenv("TEST_MASTER_KEY", "master-456")
        cfg = load_config(tmp_config_dir)
        assert cfg.litellm.model_list[0].litellm_params.api_key == "sk-test-123"
        assert cfg.litellm.general_settings.master_key == "master-456"

    def test_resolve_env_var_not_set(self, tmp_config_dir: Path, monkeypatch):
        monkeypatch.delenv("TEST_API_KEY", raising=False)
        cfg = load_config(tmp_config_dir)
        # When env var is not set, original string is kept
        assert cfg.litellm.model_list[0].litellm_params.api_key == "os.environ/TEST_API_KEY"

    def test_resolve_env_vars_nested(self):
        data = {
            "key1": "os.environ/MY_VAR",
            "nested": {"key2": "os.environ/OTHER_VAR"},
            "list": ["os.environ/LIST_VAR", "plain_value"],
        }
        os.environ["MY_VAR"] = "resolved1"
        os.environ["OTHER_VAR"] = "resolved2"
        os.environ["LIST_VAR"] = "resolved3"
        try:
            result = resolve_env_vars(data)
            assert result["key1"] == "resolved1"
            assert result["nested"]["key2"] == "resolved2"
            assert result["list"][0] == "resolved3"
            assert result["list"][1] == "plain_value"
        finally:
            del os.environ["MY_VAR"]
            del os.environ["OTHER_VAR"]
            del os.environ["LIST_VAR"]

    def test_non_env_strings_unchanged(self):
        result = resolve_env_vars("just a normal string")
        assert result == "just a normal string"

    def test_non_string_types_unchanged(self):
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(3.14) == 3.14
        assert resolve_env_vars(True) is True
        assert resolve_env_vars(None) is None


# ---------------------------------------------------------------------------
# Tests: missing config files
# ---------------------------------------------------------------------------


class TestMissingConfigFiles:
    """Verify missing optional config files don't crash."""

    def test_empty_config_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_config"
        empty_dir.mkdir()
        cfg = load_config(empty_dir)
        assert isinstance(cfg, AegisConfig)
        assert cfg.litellm.model_list == []
        assert cfg.models.models == []
        assert cfg.routing.fallback_model == "deepseek-v3"  # default
        assert cfg.overrides.overrides == {}

    def test_nonexistent_config_dir(self, tmp_path: Path):
        nonexistent = tmp_path / "no_such_dir"
        cfg = load_config(nonexistent)
        assert isinstance(cfg, AegisConfig)
        assert cfg.litellm.model_list == []

    def test_partial_config_files(self, tmp_path: Path):
        """Only models.yaml exists — other files gracefully default."""
        config_dir = tmp_path / "partial"
        config_dir.mkdir()
        models_yaml = {
            "models": [
                {
                    "name": "only-model",
                    "litellm_model": "openai/only",
                    "params": {"context_window": 8192},
                }
            ]
        }
        (config_dir / "models.yaml").write_text(yaml.dump(models_yaml), encoding="utf-8")

        cfg = load_config(config_dir)
        assert len(cfg.models.models) == 1
        assert cfg.litellm.model_list == []
        assert cfg.overrides.overrides == {}


# ---------------------------------------------------------------------------
# Tests: config reload
# ---------------------------------------------------------------------------


class TestConfigReload:
    """Verify config reload works correctly."""

    def test_reload_picks_up_changes(self, tmp_config_dir: Path):
        cfg = load_config(tmp_config_dir)
        assert cfg.routing.fallback_model == "test-fallback"

        # Modify route config
        route_yaml = {
            "routing": {
                "overlap_strategy": "highest_capability",
                "fallback_model": "new-fallback",
                "scoring": {
                    "weights": {
                        "benchmark_mmlu": 0.25,
                        "benchmark_humaneval": 0.20,
                        "benchmark_math": 0.20,
                        "context_window": 0.10,
                        "cost_efficiency": 0.25,
                    },
                    "range_tolerance": 0.20,
                },
            }
        }
        (tmp_config_dir / "route_config.yaml").write_text(
            yaml.dump(route_yaml), encoding="utf-8"
        )

        # Reload using the stored config_dir
        new_cfg = reload_config(tmp_config_dir)
        assert new_cfg.routing.fallback_model == "new-fallback"
        assert new_cfg.routing.overlap_strategy == "highest_capability"
        assert new_cfg.routing.scoring.range_tolerance == 0.20

    def test_get_config_loads_on_first_call(self, tmp_config_dir: Path, monkeypatch):
        monkeypatch.chdir(tmp_config_dir.parent)
        # Rewrite config into ./config relative to cwd
        config_subdir = tmp_config_dir.parent / "config"
        if not config_subdir.exists():
            config_subdir.mkdir()
            # Copy files
            for f in tmp_config_dir.iterdir():
                (config_subdir / f.name).write_bytes(f.read_bytes())

        cfg = get_config()
        assert isinstance(cfg, AegisConfig)

    def test_reload_uses_previous_dir(self, tmp_config_dir: Path):
        # Load initially from the fixture dir
        load_config(tmp_config_dir)
        # Use get_config to set global
        from aegis_router.config import _global_config
        from aegis_router import config as config_module
        config_module._global_config = load_config(tmp_config_dir)

        # Reload without specifying dir
        reloaded = reload_config()
        assert reloaded.config_dir == str(tmp_config_dir)


# ---------------------------------------------------------------------------
# Tests: real project config files
# ---------------------------------------------------------------------------


class TestRealConfigFiles:
    """Verify loading from the actual project config/ directory."""

    def test_load_project_config(self):
        """Load from the real project config directory."""
        project_config = Path(__file__).parent.parent / "config"
        if not project_config.exists():
            pytest.skip("Project config/ directory not found")

        cfg = load_config(project_config)
        assert len(cfg.litellm.model_list) >= 1
        assert len(cfg.models.models) >= 1
        assert cfg.routing.overlap_strategy in (
            "lowest_cost", "highest_capability", "round_robin", "random"
        )
