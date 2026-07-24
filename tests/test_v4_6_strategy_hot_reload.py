"""V4-6 验证测试：修改 route_config.yaml 的 overlap_strategy → 无需重启，下次请求使用新策略

验证完整的配置热更新链路:
1. ConfigWatcher 监听 route_config.yaml 变更
2. SmartRouterCallback._on_routing_table_updated() 被调用
3. RouteResolver 使用新策略重建
4. 下次路由请求使用新的 overlap_strategy

测试场景:
- 初始 overlap_strategy = lowest_cost
- 两个模型区间重叠 → lowest_cost 选便宜的 model-cheap
- 修改 route_config.yaml → overlap_strategy = highest_capability
- 等待热更新
- 同样的请求 → highest_capability 选能力更强的 model-capable
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.config import reset_config
from aegis_router.router.config_watcher import ConfigWatcher


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
    """Create a temporary config directory with two overlapping models.

    model-cheap:   low cost ($0.10), lower capability (computed_score ≈ 0.40)
    model-capable: high cost ($5.00), higher capability (computed_score ≈ 0.65)

    Both models have overlapping score_range so that a prompt score in the
    overlap zone will trigger the overlap strategy selection.
    """
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()

    # Two models designed to have overlapping score ranges with tolerance=0.15
    # model-cheap: benchmark scores moderate, very cheap → cost_efficiency high
    # model-capable: benchmark scores high, expensive → cost_efficiency low
    models_yaml = {
        "models": [
            {
                "name": "model-cheap",
                "litellm_model": "provider/model-cheap",
                "params": {
                    "parameter_size_b": 7,
                    "context_window": 32000,
                    "benchmark_mmlu": 70.0,
                    "benchmark_humaneval": 55.0,
                    "benchmark_math": 50.0,
                    "cost_per_1m_input": 0.10,
                    "cost_per_1m_output": 0.30,
                },
            },
            {
                "name": "model-capable",
                "litellm_model": "provider/model-capable",
                "params": {
                    "parameter_size_b": 70,
                    "context_window": 128000,
                    "benchmark_mmlu": 88.0,
                    "benchmark_humaneval": 85.0,
                    "benchmark_math": 80.0,
                    "cost_per_1m_input": 5.00,
                    "cost_per_1m_output": 15.00,
                },
            },
        ]
    }
    (cfg_dir / "models.yaml").write_text(yaml.dump(models_yaml), encoding="utf-8")

    # Initial strategy: lowest_cost
    route_yaml = {
        "routing": {
            "score_input": "original",
            "overlap_strategy": "lowest_cost",
            "fallback_model": "model-cheap",
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
                "range_tolerance": 0.20,
            },
        }
    }
    (cfg_dir / "route_config.yaml").write_text(yaml.dump(route_yaml), encoding="utf-8")

    # No overrides
    overrides_yaml = {"overrides": {}}
    (cfg_dir / "route_overrides.yaml").write_text(
        yaml.dump(overrides_yaml), encoding="utf-8"
    )

    return cfg_dir


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool that bypasses PII masking."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.max_connections = 10
    # Return None from pool.call to trigger bypass mode (skip PII masking)
    # so that routing still executes with original text
    pool.call = AsyncMock(return_value=None)
    return pool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStrategyHotReload:
    """V4-6: 修改 route_config.yaml 的 overlap_strategy → 无需重启，下次请求使用新策略。"""

    def test_hot_reload_changes_overlap_strategy(self, config_dir: Path, mock_pool):
        """End-to-end test: change overlap_strategy from lowest_cost to highest_capability.

        Steps:
        1. Start ConfigWatcher + SmartRouterCallback with lowest_cost
        2. Verify routing table has overlapping models
        3. Route a request in the overlap zone → picks cheaper model
        4. Modify route_config.yaml to highest_capability
        5. Wait for hot-reload
        6. Route same request → picks more capable model
        """
        # --- Step 1: Initialize ConfigWatcher ---
        # Create watcher without callback first, will wire it after SmartRouterCallback init
        watcher = ConfigWatcher(
            config_dir=config_dir,
            debounce_seconds=0.3,
        )
        watcher.start()

        try:
            # --- Step 2: Verify routing table built correctly ---
            table = watcher.get_current_routing_table()
            assert len(table) == 2, f"Expected 2 models in routing table, got {len(table)}"

            # Find the overlap zone: a score that falls within both models' ranges
            cheap_tier = next(t for t in table if t["name"] == "model-cheap")
            capable_tier = next(t for t in table if t["name"] == "model-capable")

            # Verify there IS overlap between the two models
            overlap_low = max(cheap_tier["score_range"][0], capable_tier["score_range"][0])
            overlap_high = min(cheap_tier["score_range"][1], capable_tier["score_range"][1])
            assert overlap_low < overlap_high, (
                f"Models must have overlapping ranges for this test. "
                f"cheap={cheap_tier['score_range']}, capable={capable_tier['score_range']}"
            )

            # Pick a score in the middle of the overlap zone
            overlap_score = (overlap_low + overlap_high) / 2.0

            # --- Step 3: Create SmartRouterCallback with lowest_cost ---
            # Mock the classifier to return our fixed overlap_score
            mock_classifier = MagicMock()
            mock_classify_result = MagicMock()
            mock_classify_result.score = overlap_score
            mock_classifier.aclassify = AsyncMock(return_value=mock_classify_result)

            callback = SmartRouterCallback(
                pool=mock_pool,
                enable_routing=True,
                config_watcher=watcher,
                classifier=mock_classifier,
                rule_engine=MagicMock(check=MagicMock(return_value=MagicMock(matched=False))),
            )

            # Wire the hot-reload callback so ConfigWatcher notifies SmartRouterCallback
            watcher._callback = callback._on_routing_table_updated

            # Verify initial strategy
            assert callback._route_resolver is not None
            assert callback._route_resolver.strategy == "lowest_cost"

            # Route a request in the overlap zone with lowest_cost
            data_1 = {
                "messages": [{"role": "user", "content": "write a complex report"}],
                "model": "default-model",
                "metadata": {},
            }
            asyncio.run(callback.async_pre_call_hook({}, None, data_1, "completion"))

            # lowest_cost should select the cheaper model
            assert data_1["model"] == "provider/model-cheap", (
                f"Expected lowest_cost to pick model-cheap, got {data_1['model']}"
            )

            # --- Step 4: Modify route_config.yaml to highest_capability ---
            route_yaml_updated = {
                "routing": {
                    "score_input": "original",
                    "overlap_strategy": "highest_capability",
                    "fallback_model": "model-cheap",
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
                        "range_tolerance": 0.20,
                    },
                }
            }
            (config_dir / "route_config.yaml").write_text(
                yaml.dump(route_yaml_updated), encoding="utf-8"
            )

            # --- Step 5: Wait for debounce + processing ---
            time.sleep(1.5)

            # --- Step 6: Verify strategy changed ---
            # The ConfigWatcher should have reloaded and called _on_routing_table_updated
            cfg = watcher.get_current_config()
            assert cfg is not None
            assert cfg.routing.overlap_strategy == "highest_capability"

            # The RouteResolver should now use highest_capability
            assert callback._route_resolver is not None
            assert callback._route_resolver.strategy == "highest_capability"

            # Route the same request again
            data_2 = {
                "messages": [{"role": "user", "content": "write a complex report"}],
                "model": "default-model",
                "metadata": {},
            }
            asyncio.run(callback.async_pre_call_hook({}, None, data_2, "completion"))

            # highest_capability should select the more capable (higher computed_score) model
            assert data_2["model"] == "provider/model-capable", (
                f"Expected highest_capability to pick model-capable, got {data_2['model']}"
            )

        finally:
            watcher.stop()

    def test_strategy_change_does_not_require_restart(self, config_dir: Path, mock_pool):
        """Verify that the hot-reload path does NOT require recreating the callback.

        The same SmartRouterCallback instance picks up the new strategy via
        the _on_routing_table_updated callback — no restart needed.
        """
        watcher = ConfigWatcher(
            config_dir=config_dir,
            debounce_seconds=0.3,
        )
        watcher.start()

        try:
            table = watcher.get_current_routing_table()
            cheap_tier = next(t for t in table if t["name"] == "model-cheap")
            capable_tier = next(t for t in table if t["name"] == "model-capable")
            overlap_low = max(cheap_tier["score_range"][0], capable_tier["score_range"][0])
            overlap_high = min(cheap_tier["score_range"][1], capable_tier["score_range"][1])
            overlap_score = (overlap_low + overlap_high) / 2.0

            mock_classifier = MagicMock()
            mock_classify_result = MagicMock()
            mock_classify_result.score = overlap_score
            mock_classifier.aclassify = AsyncMock(return_value=mock_classify_result)

            callback = SmartRouterCallback(
                pool=mock_pool,
                enable_routing=True,
                config_watcher=watcher,
                classifier=mock_classifier,
                rule_engine=MagicMock(check=MagicMock(return_value=MagicMock(matched=False))),
            )

            # Wire the hot-reload callback
            watcher._callback = callback._on_routing_table_updated

            # Record the callback object id — it should NOT change
            callback_id = id(callback)

            # Initial: lowest_cost
            assert callback._route_resolver.strategy == "lowest_cost"

            # Change strategy
            route_yaml_updated = {
                "routing": {
                    "score_input": "original",
                    "overlap_strategy": "highest_capability",
                    "fallback_model": "model-cheap",
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
                        "range_tolerance": 0.20,
                    },
                }
            }
            (config_dir / "route_config.yaml").write_text(
                yaml.dump(route_yaml_updated), encoding="utf-8"
            )

            time.sleep(1.5)

            # Same object, new strategy
            assert id(callback) == callback_id
            assert callback._route_resolver.strategy == "highest_capability"

        finally:
            watcher.stop()

    def test_multiple_strategy_switches(self, config_dir: Path, mock_pool):
        """Verify multiple consecutive strategy switches all take effect."""
        watcher = ConfigWatcher(
            config_dir=config_dir,
            debounce_seconds=0.3,
        )
        watcher.start()

        try:
            table = watcher.get_current_routing_table()
            cheap_tier = next(t for t in table if t["name"] == "model-cheap")
            capable_tier = next(t for t in table if t["name"] == "model-capable")
            overlap_low = max(cheap_tier["score_range"][0], capable_tier["score_range"][0])
            overlap_high = min(cheap_tier["score_range"][1], capable_tier["score_range"][1])
            overlap_score = (overlap_low + overlap_high) / 2.0

            mock_classifier = MagicMock()
            mock_classify_result = MagicMock()
            mock_classify_result.score = overlap_score
            mock_classifier.aclassify = AsyncMock(return_value=mock_classify_result)

            callback = SmartRouterCallback(
                pool=mock_pool,
                enable_routing=True,
                config_watcher=watcher,
                classifier=mock_classifier,
                rule_engine=MagicMock(check=MagicMock(return_value=MagicMock(matched=False))),
            )

            # Wire the hot-reload callback
            watcher._callback = callback._on_routing_table_updated

            # Initial: lowest_cost → picks cheap model
            assert callback._route_resolver.strategy == "lowest_cost"

            # Switch to highest_capability
            self._write_route_config(config_dir, "highest_capability")
            time.sleep(1.5)
            assert callback._route_resolver.strategy == "highest_capability"

            data = {
                "messages": [{"role": "user", "content": "test"}],
                "model": "x",
                "metadata": {},
            }
            asyncio.run(callback.async_pre_call_hook({}, None, data, "completion"))
            assert data["model"] == "provider/model-capable"

            # Switch back to lowest_cost
            self._write_route_config(config_dir, "lowest_cost")
            time.sleep(1.5)
            assert callback._route_resolver.strategy == "lowest_cost"

            data2 = {
                "messages": [{"role": "user", "content": "test"}],
                "model": "x",
                "metadata": {},
            }
            asyncio.run(callback.async_pre_call_hook({}, None, data2, "completion"))
            assert data2["model"] == "provider/model-cheap"

        finally:
            watcher.stop()

    @staticmethod
    def _write_route_config(config_dir: Path, strategy: str) -> None:
        """Helper to write route_config.yaml with given strategy."""
        route_yaml = {
            "routing": {
                "score_input": "original",
                "overlap_strategy": strategy,
                "fallback_model": "model-cheap",
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
                    "range_tolerance": 0.20,
                },
            }
        }
        (config_dir / "route_config.yaml").write_text(
            yaml.dump(route_yaml), encoding="utf-8"
        )
