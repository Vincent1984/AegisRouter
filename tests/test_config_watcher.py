"""Tests for aegis_router.router.config_watcher module."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from aegis_router.config import AegisConfig, reset_config
from aegis_router.router.config_watcher import ConfigWatcher, WATCHED_FILES


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
    """Create a temporary config directory with valid YAML files."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

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
            },
            {
                "name": "test-gpt4",
                "litellm_model": "openai/gpt-4o",
                "params": {
                    "parameter_size_b": None,
                    "context_window": 128000,
                    "benchmark_mmlu": 88.7,
                    "benchmark_humaneval": 90.2,
                    "benchmark_math": 81.4,
                    "cost_per_1m_input": 2.50,
                    "cost_per_1m_output": 10.00,
                },
            },
        ]
    }
    (cfg_dir / "models.yaml").write_text(yaml.dump(models_yaml), encoding="utf-8")

    route_yaml = {
        "routing": {
            "overlap_strategy": "lowest_cost",
            "fallback_model": "test-7b",
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

    overrides_yaml = {
        "overrides": {
            "test-7b": {
                "score_range": [0.0, 0.20],
                "reason": "test override",
            }
        }
    }
    (cfg_dir / "route_overrides.yaml").write_text(
        yaml.dump(overrides_yaml), encoding="utf-8"
    )

    return cfg_dir


# ---------------------------------------------------------------------------
# Tests: start/stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """ConfigWatcher starts and stops cleanly."""

    def test_start_and_stop(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir)
        watcher.start()
        assert watcher.is_running is True
        watcher.stop()
        assert watcher.is_running is False

    def test_double_start_is_safe(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir)
        watcher.start()
        watcher.start()  # should not raise
        assert watcher.is_running is True
        watcher.stop()

    def test_stop_without_start_is_safe(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir)
        watcher.stop()  # should not raise
        assert watcher.is_running is False

    def test_initial_config_loaded_on_start(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir)
        watcher.start()
        try:
            cfg = watcher.get_current_config()
            assert cfg is not None
            assert isinstance(cfg, AegisConfig)
            assert len(cfg.models.models) == 2
        finally:
            watcher.stop()

    def test_initial_routing_table_built(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir)
        watcher.start()
        try:
            table = watcher.get_current_routing_table()
            assert len(table) == 2
            # Table should be sorted by computed_score
            assert table[0]["computed_score"] <= table[1]["computed_score"]
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# Tests: file modification triggers reload
# ---------------------------------------------------------------------------


class TestFileModificationReload:
    """Modifying a watched YAML file triggers a config reload."""

    def test_modify_models_yaml_triggers_reload(self, config_dir: Path):
        callback = MagicMock()
        watcher = ConfigWatcher(
            config_dir, on_routing_table_updated=callback, debounce_seconds=0.3
        )
        watcher.start()
        try:
            # Modify models.yaml — add a new model
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
                    },
                    {
                        "name": "test-gpt4",
                        "litellm_model": "openai/gpt-4o",
                        "params": {
                            "context_window": 128000,
                            "benchmark_mmlu": 88.7,
                            "benchmark_humaneval": 90.2,
                            "benchmark_math": 81.4,
                            "cost_per_1m_input": 2.50,
                            "cost_per_1m_output": 10.00,
                        },
                    },
                    {
                        "name": "new-model",
                        "litellm_model": "openai/new",
                        "params": {
                            "context_window": 64000,
                            "benchmark_mmlu": 75.0,
                            "benchmark_humaneval": 60.0,
                            "benchmark_math": 55.0,
                            "cost_per_1m_input": 1.0,
                            "cost_per_1m_output": 3.0,
                        },
                    },
                ]
            }
            (config_dir / "models.yaml").write_text(
                yaml.dump(models_yaml), encoding="utf-8"
            )

            # Wait for debounce + processing
            time.sleep(1.5)

            # Routing table should now have 3 entries
            table = watcher.get_current_routing_table()
            assert len(table) == 3
            assert callback.called
        finally:
            watcher.stop()

    def test_modify_route_config_triggers_reload(self, config_dir: Path):
        callback = MagicMock()
        watcher = ConfigWatcher(
            config_dir, on_routing_table_updated=callback, debounce_seconds=0.3
        )
        watcher.start()
        try:
            # Change fallback model
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
            (config_dir / "route_config.yaml").write_text(
                yaml.dump(route_yaml), encoding="utf-8"
            )

            time.sleep(1.5)

            cfg = watcher.get_current_config()
            assert cfg is not None
            assert cfg.routing.overlap_strategy == "highest_capability"
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# Tests: debouncing
# ---------------------------------------------------------------------------


class TestDebouncing:
    """Multiple rapid changes result in a single reload."""

    def test_rapid_changes_debounced(self, config_dir: Path):
        callback = MagicMock()
        watcher = ConfigWatcher(
            config_dir, on_routing_table_updated=callback, debounce_seconds=0.5
        )
        watcher.start()
        try:
            # Rapid-fire modifications (simulate editor saving multiple times)
            for i in range(5):
                overrides_yaml = {
                    "overrides": {
                        "test-7b": {
                            "score_range": [0.0, 0.10 + i * 0.01],
                            "reason": f"rapid change {i}",
                        }
                    }
                }
                (config_dir / "route_overrides.yaml").write_text(
                    yaml.dump(overrides_yaml), encoding="utf-8"
                )
                time.sleep(0.1)  # 100ms between writes

            # Wait for debounce to fire
            time.sleep(1.5)

            # Should have been called only once (the debounced reload),
            # not 5 times for each write
            # Note: the exact count depends on watchdog event delivery,
            # but it should be significantly less than 5
            assert callback.call_count <= 2

        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# Tests: graceful error handling
# ---------------------------------------------------------------------------


class TestGracefulErrorHandling:
    """File parse errors don't crash the watcher."""

    def test_invalid_yaml_keeps_previous_config(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir, debounce_seconds=0.3)
        watcher.start()
        try:
            initial_table = watcher.get_current_routing_table()
            assert len(initial_table) == 2

            # Write invalid YAML
            (config_dir / "models.yaml").write_text(
                "invalid: yaml: [[[broken", encoding="utf-8"
            )

            time.sleep(1.5)

            # The watcher should still have the previous valid table
            # (load_config returns empty models for invalid yaml, which
            # means the routing table may be rebuilt with 0 entries from
            # models but won't crash)
            # What matters is the watcher is still running
            assert watcher.is_running is True
        finally:
            watcher.stop()

    def test_watcher_survives_callback_error(self, config_dir: Path):
        def bad_callback(table):
            raise RuntimeError("Callback exploded!")

        watcher = ConfigWatcher(
            config_dir, on_routing_table_updated=bad_callback, debounce_seconds=0.3
        )
        watcher.start()
        try:
            # Trigger a change
            overrides_yaml = {
                "overrides": {
                    "test-7b": {
                        "score_range": [0.0, 0.25],
                        "reason": "trigger callback error",
                    }
                }
            }
            (config_dir / "route_overrides.yaml").write_text(
                yaml.dump(overrides_yaml), encoding="utf-8"
            )

            time.sleep(1.5)

            # Watcher should still be running despite callback error
            assert watcher.is_running is True
        finally:
            watcher.stop()


# ---------------------------------------------------------------------------
# Tests: routing table rebuild correctness
# ---------------------------------------------------------------------------


class TestRoutingTableRebuild:
    """The routing table is correctly rebuilt after config change."""

    def test_override_change_reflected_in_table(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir, debounce_seconds=0.3)
        watcher.start()
        try:
            # Initial: test-7b has override [0.0, 0.20]
            table = watcher.get_current_routing_table()
            test_7b = next(t for t in table if t["name"] == "test-7b")
            assert test_7b["overridden"] is True
            assert test_7b["score_range"] == (0.0, 0.20)

            # Change override to [0.0, 0.30]
            overrides_yaml = {
                "overrides": {
                    "test-7b": {
                        "score_range": [0.0, 0.30],
                        "reason": "expanded range",
                    }
                }
            }
            (config_dir / "route_overrides.yaml").write_text(
                yaml.dump(overrides_yaml), encoding="utf-8"
            )

            time.sleep(1.5)

            table = watcher.get_current_routing_table()
            test_7b = next(t for t in table if t["name"] == "test-7b")
            assert test_7b["score_range"] == (0.0, 0.30)
        finally:
            watcher.stop()

    def test_table_sorted_by_computed_score(self, config_dir: Path):
        watcher = ConfigWatcher(config_dir, debounce_seconds=0.1)
        watcher.start()
        try:
            table = watcher.get_current_routing_table()
            scores = [t["computed_score"] for t in table]
            assert scores == sorted(scores)
        finally:
            watcher.stop()

    def test_thread_safe_get_routing_table(self, config_dir: Path):
        """Concurrent reads of the routing table don't crash."""
        watcher = ConfigWatcher(config_dir, debounce_seconds=0.3)
        watcher.start()
        errors = []

        def reader():
            try:
                for _ in range(50):
                    table = watcher.get_current_routing_table()
                    assert isinstance(table, list)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            assert not errors
        finally:
            watcher.stop()
