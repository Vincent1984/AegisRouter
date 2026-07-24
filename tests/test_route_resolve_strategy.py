"""路由匹配与重叠策略测试

基于真实配置文件（models.yaml + route_config.yaml + route_overrides.yaml）
构建路由表，验证 RouteResolver 的区间匹配和重叠策略逻辑。

测试用例:
- TC-RESOLVE-001: prompt score 在重叠区 → lowest_cost 选最便宜的
- TC-RESOLVE-002: prompt score 在另一重叠区 → lowest_cost 选较便宜的
- TC-RESOLVE-003: prompt score 仅命中单个模型 → 单候选直接返回
- TC-RESOLVE-004: highest_capability → 重叠时选 computed_score 最高的模型
- TC-RESOLVE-005: round_robin → 连续 10 次相同分数请求均匀分布到候选模型
- TC-RESOLVE-006: 无候选命中 → 返回 fallback_model
- TC-RESOLVE-007: 配置热更新后路由表实时生效
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import pytest
import yaml

from aegis_router.config import load_config
from aegis_router.router.config_watcher import ConfigWatcher
from aegis_router.router.model_scorer import ModelScorer, build_routing_table
from aegis_router.router.route_resolver import RouteResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).parent.parent / "config"


@pytest.fixture
def real_routing_table():
    """从真实配置文件构建路由表。"""
    config = load_config(CONFIG_DIR)

    scoring_cfg = config.routing.scoring
    weights = scoring_cfg.weights.model_dump()
    normalization = scoring_cfg.normalization.model_dump()
    tolerance = scoring_cfg.range_tolerance

    scorer = ModelScorer(
        weights=weights,
        normalization=normalization,
        tolerance=tolerance,
    )

    models_data = [
        {
            "name": m.name,
            "litellm_model": m.litellm_model,
            "params": m.params.model_dump(),
        }
        for m in config.models.models
    ]

    overrides_data = {
        name: override.model_dump()
        for name, override in config.overrides.overrides.items()
    }

    return build_routing_table(models_data, scorer, overrides_data)


def _get_candidates(tiers, score):
    """获取给定分数命中的候选模型名列表。"""
    return [
        tier["name"]
        for tier in tiers
        if tier["score_range"][0] <= score <= tier["score_range"][1]
    ]


# ---------------------------------------------------------------------------
# TC-RESOLVE-001: prompt score 在 local-7b 独占区 → lowest_cost 选 local-7b
#
# 真实路由表:
#   local-7b:      (0.0, 0.18)  cost=0.0    [overridden]
#   o1:            (~0.55, ~0.85) cost=15.0
#   deepseek-v3:   (~0.62, ~0.92) cost=0.27
#   gpt-4o:        (0.50, 0.82) cost=2.5    [overridden]
#   gemini-1.5-pro:(~0.64, ~0.94) cost=1.25
#
# score=0.12 仅命中 local-7b，单候选直接返回
# ---------------------------------------------------------------------------


class TestResolve001:
    """TC-RESOLVE-001: prompt score 0.12 → 命中 local-7b，lowest_cost 选 local-7b。"""

    def test_score_012_hits_local_7b(self, real_routing_table):
        """验证 score=0.12 落在 local-7b 的区间 [0.0, 0.18] 内。"""
        candidates = _get_candidates(real_routing_table, 0.12)
        assert "local-7b" in candidates

    def test_lowest_cost_selects_local_7b(self, real_routing_table):
        """lowest_cost 策略应选择 local-7b（cost=0）。"""
        resolver = RouteResolver(
            real_routing_table, strategy="lowest_cost", fallback_model="deepseek-v3"
        )
        result = resolver.resolve(0.12)
        assert result["model"] == "ollama/qwen2-7b"
        # local-7b 是此区间唯一候选
        assert result["reason"] == "single_match"


# ---------------------------------------------------------------------------
# TC-RESOLVE-002: prompt score 在多模型重叠区 → lowest_cost 选最便宜的
#
# score=0.65 命中: o1(~0.55,~0.85), deepseek-v3(~0.62,~0.92),
#                  gpt-4o(0.50,0.82), gemini-1.5-pro(~0.64,~0.94)
# lowest_cost → 选 deepseek-v3 (cost=0.27)
# ---------------------------------------------------------------------------


class TestResolve002:
    """TC-RESOLVE-002: prompt score 0.65 → 多候选重叠区，lowest_cost 选 deepseek-v3。"""

    def test_score_065_hits_multiple_candidates(self, real_routing_table):
        """验证 score=0.65 同时命中多个模型。"""
        candidates = _get_candidates(real_routing_table, 0.65)
        # 至少命中 gpt-4o 和 deepseek-v3
        assert "gpt-4o" in candidates
        assert "deepseek-v3" in candidates
        assert len(candidates) >= 2

    def test_lowest_cost_selects_cheapest(self, real_routing_table):
        """lowest_cost 策略选 cost 最低的 deepseek-v3（cost=0.27）。"""
        resolver = RouteResolver(
            real_routing_table, strategy="lowest_cost", fallback_model="deepseek-v3"
        )
        result = resolver.resolve(0.65)
        assert result["model"] == "deepseek/deepseek-chat"
        assert "lowest_cost" in result["reason"]


# ---------------------------------------------------------------------------
# TC-RESOLVE-003: prompt score 在边缘区域 → 仅命中少数模型
#
# score=0.95 → 仅命中 gemini-1.5-pro (~0.64, ~0.94) 的上界附近
# 实际 gemini 上界约 0.94，所以 0.95 不命中任何模型 → fallback
# 改用 score=0.93 → 命中 deepseek-v3(~0.62,~0.92上界) + gemini-1.5-pro(~0.64,~0.94)
# 或找一个只命中一个模型的分数
#
# score=0.50 只命中 gpt-4o (0.50, 0.82) 的下边界
# 但 o1 的下界 ~0.55 所以 0.50 确实只命中 gpt-4o
# ---------------------------------------------------------------------------


class TestResolve003:
    """TC-RESOLVE-003: 仅命中单个候选时直接返回。"""

    def test_single_candidate_direct_return(self, real_routing_table):
        """找到一个只命中单个模型的分数，验证 single_match。"""
        # score=0.50 正好在 gpt-4o 的下界 (overridden [0.50, 0.82])
        # 需确认 0.50 是否也落在其他模型区间
        candidates = _get_candidates(real_routing_table, 0.50)

        # 如果只命中一个，验证 single_match
        if len(candidates) == 1:
            resolver = RouteResolver(
                real_routing_table, strategy="lowest_cost", fallback_model="deepseek-v3"
            )
            result = resolver.resolve(0.50)
            assert result["reason"] == "single_match"
        else:
            # 尝试 score=0.10 只命中 local-7b
            candidates_010 = _get_candidates(real_routing_table, 0.10)
            assert len(candidates_010) == 1
            assert candidates_010[0] == "local-7b"

            resolver = RouteResolver(
                real_routing_table, strategy="lowest_cost", fallback_model="deepseek-v3"
            )
            result = resolver.resolve(0.10)
            assert result["model"] == "ollama/qwen2-7b"
            assert result["reason"] == "single_match"

    def test_score_010_single_match_local_7b(self, real_routing_table):
        """score=0.10 仅命中 local-7b [0.0, 0.18]，单候选直接返回。"""
        resolver = RouteResolver(
            real_routing_table, strategy="lowest_cost", fallback_model="deepseek-v3"
        )
        result = resolver.resolve(0.10)
        assert result["model"] == "ollama/qwen2-7b"
        assert result["reason"] == "single_match"
        assert result["candidates"] == ["ollama/qwen2-7b"]


# ---------------------------------------------------------------------------
# TC-RESOLVE-004: highest_capability → 重叠时选 computed_score 最高的模型
# ---------------------------------------------------------------------------


class TestResolve004:
    """TC-RESOLVE-004: highest_capability 策略在重叠时选 computed_score 最高的模型。"""

    def test_highest_capability_selects_highest_score(self, real_routing_table):
        """score=0.70 命中多个模型，highest_capability 选 computed_score 最高的。"""
        candidates = _get_candidates(real_routing_table, 0.70)
        assert len(candidates) >= 2, "需要至少两个候选才能测试重叠策略"

        resolver = RouteResolver(
            real_routing_table, strategy="highest_capability", fallback_model="deepseek-v3"
        )
        result = resolver.resolve(0.70)

        # 在候选中找到 computed_score 最高的模型
        candidate_tiers = [
            t for t in real_routing_table if t["name"] in candidates
        ]
        best = max(candidate_tiers, key=lambda t: t["computed_score"])

        assert result["model"] == best["model"]
        assert "highest_capability" in result["reason"]

    def test_highest_capability_prefers_gemini_over_others_in_overlap(
        self, real_routing_table
    ):
        """在重叠区域，gemini-1.5-pro 的 computed_score 最高（约 0.79），应被选中。"""
        # score=0.70: 应命中 o1, deepseek-v3, gpt-4o, gemini-1.5-pro
        resolver = RouteResolver(
            real_routing_table, strategy="highest_capability", fallback_model="deepseek-v3"
        )
        result = resolver.resolve(0.70)

        # gemini-1.5-pro 的 computed_score 最高 (~0.79)
        assert result["model"] == "gemini/gemini-1.5-pro"


# ---------------------------------------------------------------------------
# TC-RESOLVE-005: round_robin → 连续 10 次相同分数请求均匀分布
# ---------------------------------------------------------------------------


class TestResolve005:
    """TC-RESOLVE-005: round_robin 策略连续 10 次相同分数请求均匀分布到候选模型。"""

    def test_round_robin_distributes_across_candidates(self, real_routing_table):
        """连续 10 次相同分数请求，应轮询分布到所有候选模型。"""
        resolver = RouteResolver(
            real_routing_table, strategy="round_robin", fallback_model="deepseek-v3"
        )

        # score=0.70 命中多个候选
        candidates = _get_candidates(real_routing_table, 0.70)
        n_candidates = len(candidates)
        assert n_candidates >= 2

        # 运行 10 次
        results = [resolver.resolve(0.70)["model"] for _ in range(10)]
        counter = Counter(results)

        # 验证所有候选模型都被轮询到
        unique_models = set(results)
        assert len(unique_models) == n_candidates

        # 验证均匀分布：每个模型出现次数差不超过 1
        counts = list(counter.values())
        assert max(counts) - min(counts) <= 1

    def test_round_robin_cycles_correctly(self, real_routing_table):
        """验证 round_robin 严格按顺序循环。"""
        resolver = RouteResolver(
            real_routing_table, strategy="round_robin", fallback_model="deepseek-v3"
        )

        candidates = _get_candidates(real_routing_table, 0.70)
        n_candidates = len(candidates)

        # 运行 n_candidates * 2 次，验证第二轮与第一轮相同
        results = [resolver.resolve(0.70)["model"] for _ in range(n_candidates * 2)]

        first_cycle = results[:n_candidates]
        second_cycle = results[n_candidates:]
        assert first_cycle == second_cycle


# ---------------------------------------------------------------------------
# TC-RESOLVE-006: 无候选命中 → 返回 fallback_model
# ---------------------------------------------------------------------------


class TestResolve006:
    """TC-RESOLVE-006: 无候选命中时返回 fallback_model。"""

    def test_score_beyond_all_ranges_returns_fallback(self, real_routing_table):
        """score 超出所有模型区间时返回 fallback。"""
        # 找到一个不在任何区间内的 score
        # local-7b 上界 0.18，其他模型下界最低约 0.50
        # 所以 score=0.30 应该落在间隙中
        candidates = _get_candidates(real_routing_table, 0.30)
        assert len(candidates) == 0, "score=0.30 应不命中任何模型"

        resolver = RouteResolver(
            real_routing_table, strategy="lowest_cost", fallback_model="deepseek-v3"
        )
        result = resolver.resolve(0.30)
        assert result["model"] == "deepseek-v3"
        assert result["reason"] == "no_match_fallback"
        assert result["candidates"] == []

    def test_empty_routing_table_returns_fallback(self):
        """空路由表对任何分数都返回 fallback。"""
        resolver = RouteResolver(
            [], strategy="lowest_cost", fallback_model="deepseek-v3"
        )
        result = resolver.resolve(0.5)
        assert result["model"] == "deepseek-v3"
        assert result["reason"] == "no_match_fallback"
        assert result["candidates"] == []

    def test_custom_fallback_model_name(self, real_routing_table):
        """验证自定义 fallback_model 名称正确返回。"""
        resolver = RouteResolver(
            real_routing_table,
            strategy="lowest_cost",
            fallback_model="my-custom-fallback",
        )
        # score=0.30 在间隙中
        result = resolver.resolve(0.30)
        assert result["model"] == "my-custom-fallback"


# ---------------------------------------------------------------------------
# TC-RESOLVE-007: 配置热更新后路由表实时生效
# ---------------------------------------------------------------------------


class TestResolve007:
    """TC-RESOLVE-007: 配置热更新后路由表实时生效（改 yaml → 下次请求走新配置）。"""

    def test_config_watcher_reloads_on_file_change(self):
        """修改 route_overrides.yaml 后，ConfigWatcher 应重建路由表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 复制所有配置文件到临时目录
            for filename in ["models.yaml", "route_config.yaml", "route_overrides.yaml"]:
                src = CONFIG_DIR / filename
                dst = tmp_path / filename
                shutil.copy2(src, dst)

            # 也复制 config.yaml（load_config 需要）
            config_yaml = CONFIG_DIR / "config.yaml"
            if config_yaml.exists():
                shutil.copy2(config_yaml, tmp_path / "config.yaml")

            # 记录回调触发
            callback_tables = []

            def on_table_updated(new_table):
                callback_tables.append(new_table)

            # 创建并启动 ConfigWatcher，使用极短的 debounce 加速测试
            watcher = ConfigWatcher(
                config_dir=tmp_path,
                on_routing_table_updated=on_table_updated,
                debounce_seconds=0.1,
            )
            watcher.start()

            try:
                # 获取初始路由表
                initial_table = watcher.get_current_routing_table()
                assert len(initial_table) > 0

                # 验证初始状态：local-7b 的区间为 (0.0, 0.18)
                local_7b_initial = next(
                    t for t in initial_table if t["name"] == "local-7b"
                )
                assert local_7b_initial["score_range"] == (0.0, 0.18)

                # 修改 route_overrides.yaml — 扩大 local-7b 的区间
                overrides_path = tmp_path / "route_overrides.yaml"
                new_overrides = {
                    "overrides": {
                        "gpt-4o": {
                            "score_range": [0.50, 0.82],
                            "reason": "实测表现优于 benchmark",
                        },
                        "local-7b": {
                            "score_range": [0.0, 0.30],
                            "reason": "扩大本地模型覆盖范围",
                        },
                    }
                }
                with open(overrides_path, "w", encoding="utf-8") as f:
                    yaml.dump(new_overrides, f, allow_unicode=True)

                # 等待 debounce + reload 完成
                time.sleep(1.0)

                # 验证路由表已更新
                updated_table = watcher.get_current_routing_table()
                local_7b_updated = next(
                    t for t in updated_table if t["name"] == "local-7b"
                )
                assert local_7b_updated["score_range"] == (0.0, 0.30)

                # 验证回调被触发
                assert len(callback_tables) >= 1

            finally:
                watcher.stop()

    def test_updated_routing_table_affects_resolution(self):
        """更新后的路由表应影响 RouteResolver 的决策。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 复制配置文件
            for filename in ["models.yaml", "route_config.yaml", "route_overrides.yaml"]:
                src = CONFIG_DIR / filename
                dst = tmp_path / filename
                shutil.copy2(src, dst)

            config_yaml = CONFIG_DIR / "config.yaml"
            if config_yaml.exists():
                shutil.copy2(config_yaml, tmp_path / "config.yaml")

            # 启动 watcher
            watcher = ConfigWatcher(
                config_dir=tmp_path,
                debounce_seconds=0.1,
            )
            watcher.start()

            try:
                # 初始路由表：score=0.25 应该不在 local-7b 的 [0.0, 0.18] 内
                initial_table = watcher.get_current_routing_table()
                resolver_before = RouteResolver(
                    initial_table, strategy="lowest_cost", fallback_model="deepseek-v3"
                )
                result_before = resolver_before.resolve(0.25)

                # 0.25 > 0.18，不命中 local-7b，也不命中其他模型 → fallback
                assert result_before["model"] == "deepseek-v3"
                assert result_before["reason"] == "no_match_fallback"

                # 修改配置：扩大 local-7b 区间到 [0.0, 0.30]
                overrides_path = tmp_path / "route_overrides.yaml"
                new_overrides = {
                    "overrides": {
                        "gpt-4o": {
                            "score_range": [0.50, 0.82],
                            "reason": "保持不变",
                        },
                        "local-7b": {
                            "score_range": [0.0, 0.30],
                            "reason": "扩大覆盖",
                        },
                    }
                }
                with open(overrides_path, "w", encoding="utf-8") as f:
                    yaml.dump(new_overrides, f, allow_unicode=True)

                # 等待热更新
                time.sleep(1.0)

                # 用新路由表构建 resolver
                updated_table = watcher.get_current_routing_table()
                resolver_after = RouteResolver(
                    updated_table, strategy="lowest_cost", fallback_model="deepseek-v3"
                )
                result_after = resolver_after.resolve(0.25)

                # 更新后 local-7b 覆盖 [0.0, 0.30]，score=0.25 命中 local-7b
                assert result_after["model"] == "ollama/qwen2-7b"
                assert result_after["reason"] == "single_match"

            finally:
                watcher.stop()
