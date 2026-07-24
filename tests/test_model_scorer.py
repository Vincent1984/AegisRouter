"""模型能力评分引擎单元测试

覆盖测试用例:
- TC-SCORE-001: 验证 local-7b 的 computed_score 最低
- TC-SCORE-002: 验证 o1 的 computed_score 排名 (benchmark 最高但成本惩罚)
- TC-SCORE-003: 修改权重配置 → 重新计算分数变化符合预期
- TC-SCORE-004: 模型参数缺失某字段（如 parameter_size=null）→ 使用默认中位值，不报错
- TC-SCORE-005: score_range 自动生成 = computed_score ± tolerance
- TC-SCORE-006: 人工覆盖 (route_overrides.yaml) 优先于自动计算
"""

import pytest

from aegis_router.router.model_scorer import ModelScorer, build_routing_table


# --- 测试用配置 ---

WEIGHTS = {
    "benchmark_mmlu": 0.25,
    "benchmark_humaneval": 0.20,
    "benchmark_math": 0.20,
    "context_window": 0.10,
    "cost_efficiency": 0.25,
}

NORMALIZATION = {
    "benchmark_mmlu": [50, 95],
    "benchmark_humaneval": [30, 95],
    "benchmark_math": [20, 95],
    "context_window": [4096, 2000000],
    "cost_per_1m_input": [0, 20],
}

TOLERANCE = 0.15


@pytest.fixture
def scorer():
    return ModelScorer(weights=WEIGHTS, normalization=NORMALIZATION, tolerance=TOLERANCE)


# --- models.yaml 中的模型参数 ---

LOCAL_7B_PARAMS = {
    "context_window": 32000,
    "benchmark_mmlu": 65.0,
    "benchmark_humaneval": 45.0,
    "benchmark_math": 40.0,
    "cost_per_1m_input": 0.0,
}

DEEPSEEK_V3_PARAMS = {
    "context_window": 128000,
    "benchmark_mmlu": 87.1,
    "benchmark_humaneval": 82.6,
    "benchmark_math": 75.3,
    "cost_per_1m_input": 0.27,
}

GEMINI_15_PRO_PARAMS = {
    "context_window": 2000000,
    "benchmark_mmlu": 85.9,
    "benchmark_humaneval": 71.9,
    "benchmark_math": 67.7,
    "cost_per_1m_input": 1.25,
}

GPT_4O_PARAMS = {
    "context_window": 128000,
    "benchmark_mmlu": 88.7,
    "benchmark_humaneval": 90.2,
    "benchmark_math": 81.4,
    "cost_per_1m_input": 2.50,
}

O1_PARAMS = {
    "context_window": 200000,
    "benchmark_mmlu": 91.8,
    "benchmark_humaneval": 94.2,
    "benchmark_math": 94.8,
    "cost_per_1m_input": 15.00,
}


# ============================================================
# 归一化测试
# ============================================================


class TestNormalize:
    """归一化方法测试"""

    def test_none_returns_midpoint(self, scorer):
        """None 值应返回 0.5"""
        assert scorer.normalize(None, 50, 95) == 0.5

    def test_value_at_min_returns_zero(self, scorer):
        """最小值应返回 0.0"""
        assert scorer.normalize(50, 50, 95) == 0.0

    def test_value_at_max_returns_one(self, scorer):
        """最大值应返回 1.0"""
        assert scorer.normalize(95, 50, 95) == 1.0

    def test_value_below_min_clamped_to_zero(self, scorer):
        """低于最小值应 clamp 到 0.0"""
        assert scorer.normalize(30, 50, 95) == 0.0

    def test_value_above_max_clamped_to_one(self, scorer):
        """高于最大值应 clamp 到 1.0"""
        assert scorer.normalize(100, 50, 95) == 1.0

    def test_midpoint_value(self, scorer):
        """中间值应返回 0.5 左右"""
        result = scorer.normalize(72.5, 50, 95)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_boundary_zero_value(self, scorer):
        """值为 0 且 min=0 时应归一化为 0.0"""
        assert scorer.normalize(0, 0, 20) == 0.0

    def test_large_range(self, scorer):
        """大范围归一化 (context_window)"""
        result = scorer.normalize(128000, 4096, 2000000)
        assert 0.0 < result < 1.0


# ============================================================
# 评分计算测试
# ============================================================


class TestComputeScore:
    """compute_score 方法测试"""

    def test_tc_score_001_local_7b_lowest_score(self, scorer):
        """TC-SCORE-001: local-7b 应该是所有 5 个模型中 computed_score 最低的"""
        scores = {
            "local-7b": scorer.compute_score(LOCAL_7B_PARAMS),
            "deepseek-v3": scorer.compute_score(DEEPSEEK_V3_PARAMS),
            "gemini-1.5-pro": scorer.compute_score(GEMINI_15_PRO_PARAMS),
            "gpt-4o": scorer.compute_score(GPT_4O_PARAMS),
            "o1": scorer.compute_score(O1_PARAMS),
        }
        # local-7b 必须是绝对最低分（低 benchmark + 虽然免费但 benchmark 太弱）
        assert scores["local-7b"] == min(scores.values())
        # 额外验证: local-7b 低于所有其他模型
        for name, score in scores.items():
            if name != "local-7b":
                assert scores["local-7b"] < score, f"local-7b should be lower than {name}"

    def test_tc_score_002_o1_benchmark_vs_cost_tradeoff(self, scorer):
        """TC-SCORE-002: 验证 o1 的 computed_score 排名。

        o1 拥有最高的 benchmark 分数，但成本极高 ($15/M input)。
        在默认权重下 cost_efficiency=0.25，o1 的高成本会显著拉低综合评分。
        实际排名: gemini-1.5-pro > gpt-4o ≈ deepseek-v3 > o1 > local-7b
        这验证了评分算法正确权衡了质量与成本的 tradeoff。
        """
        scores = {
            "local-7b": scorer.compute_score(LOCAL_7B_PARAMS),
            "deepseek-v3": scorer.compute_score(DEEPSEEK_V3_PARAMS),
            "gemini-1.5-pro": scorer.compute_score(GEMINI_15_PRO_PARAMS),
            "gpt-4o": scorer.compute_score(GPT_4O_PARAMS),
            "o1": scorer.compute_score(O1_PARAMS),
        }
        # o1 的 benchmark 确实最高，但综合分并非最高
        # 验证: o1 高于 local-7b 但低于 deepseek-v3（成本优势模型）
        assert scores["o1"] > scores["local-7b"]
        assert scores["o1"] < scores["deepseek-v3"], (
            "o1 cost penalty ($15) should make its overall score lower than deepseek-v3 (cost=$0.27)"
        )
        # 验证: 当仅考虑 benchmark 时 (去掉 cost_efficiency), o1 应该最高
        benchmark_only_weights = {
            "benchmark_mmlu": 0.35,
            "benchmark_humaneval": 0.30,
            "benchmark_math": 0.25,
            "context_window": 0.10,
            "cost_efficiency": 0.0,  # 去除成本因素
        }
        scorer_no_cost = ModelScorer(
            weights=benchmark_only_weights, normalization=NORMALIZATION, tolerance=TOLERANCE
        )
        scores_no_cost = {
            name: scorer_no_cost.compute_score(params)
            for name, params in [
                ("local-7b", LOCAL_7B_PARAMS),
                ("deepseek-v3", DEEPSEEK_V3_PARAMS),
                ("gemini-1.5-pro", GEMINI_15_PRO_PARAMS),
                ("gpt-4o", GPT_4O_PARAMS),
                ("o1", O1_PARAMS),
            ]
        }
        assert scores_no_cost["o1"] == max(scores_no_cost.values()), (
            "Without cost penalty, o1 should have the highest score due to superior benchmarks"
        )

    def test_tc_score_003_weight_modification_changes_scores(self, scorer):
        """TC-SCORE-003: 修改权重配置后重新计算，分数变化符合预期。

        当增加 cost_efficiency 权重时:
        - 低成本模型 (local-7b, deepseek-v3) 的分数应该上升
        - 高成本模型 (o1) 的分数应该下降
        """
        # 原始分数
        original_scores = {
            "local-7b": scorer.compute_score(LOCAL_7B_PARAMS),
            "deepseek-v3": scorer.compute_score(DEEPSEEK_V3_PARAMS),
            "o1": scorer.compute_score(O1_PARAMS),
        }

        # 修改权重: 大幅增加 cost_efficiency 权重
        cost_heavy_weights = {
            "benchmark_mmlu": 0.15,
            "benchmark_humaneval": 0.10,
            "benchmark_math": 0.10,
            "context_window": 0.05,
            "cost_efficiency": 0.60,  # 从 0.25 提升到 0.60
        }
        cost_heavy_scorer = ModelScorer(
            weights=cost_heavy_weights, normalization=NORMALIZATION, tolerance=TOLERANCE
        )
        new_scores = {
            "local-7b": cost_heavy_scorer.compute_score(LOCAL_7B_PARAMS),
            "deepseek-v3": cost_heavy_scorer.compute_score(DEEPSEEK_V3_PARAMS),
            "o1": cost_heavy_scorer.compute_score(O1_PARAMS),
        }

        # 低成本模型分数应该上升（local-7b cost=0, 性价比最高）
        assert new_scores["local-7b"] > original_scores["local-7b"], (
            "local-7b (free) should score higher when cost_efficiency weight increases"
        )
        # deepseek-v3 几乎免费，性价比极高，分数也应上升
        assert new_scores["deepseek-v3"] > original_scores["deepseek-v3"], (
            "deepseek-v3 (cost=0.27) should score higher when cost_efficiency weight increases"
        )
        # 高成本模型分数应该下降
        assert new_scores["o1"] < original_scores["o1"], (
            "o1 (cost=15) should score lower when cost_efficiency weight increases"
        )

    def test_tc_score_003_benchmark_weight_increase(self, scorer):
        """TC-SCORE-003 补充: 增加 benchmark_mmlu 权重时，高 MMLU 模型分数应上升。"""
        original_o1 = scorer.compute_score(O1_PARAMS)
        original_local = scorer.compute_score(LOCAL_7B_PARAMS)

        # 增加 benchmark_mmlu 权重
        mmlu_heavy_weights = {
            "benchmark_mmlu": 0.50,  # 从 0.25 提升到 0.50
            "benchmark_humaneval": 0.15,
            "benchmark_math": 0.15,
            "context_window": 0.05,
            "cost_efficiency": 0.15,  # 降低
        }
        mmlu_scorer = ModelScorer(
            weights=mmlu_heavy_weights, normalization=NORMALIZATION, tolerance=TOLERANCE
        )
        new_o1 = mmlu_scorer.compute_score(O1_PARAMS)
        new_local = mmlu_scorer.compute_score(LOCAL_7B_PARAMS)

        # o1 (mmlu=91.8) 应该在增加 mmlu 权重后分数上升
        assert new_o1 > original_o1, "o1 (high MMLU) should improve with higher MMLU weight"
        # local-7b (mmlu=65) 相对差异应该扩大
        gap_original = original_o1 - original_local
        gap_new = new_o1 - new_local
        assert gap_new > gap_original, (
            "Score gap between o1 and local-7b should widen with higher MMLU weight"
        )

    def test_tc_score_004_null_parameter_uses_default_midpoint(self, scorer):
        """TC-SCORE-004: 模型参数缺失 (parameter_size_b=null) → 使用默认中位值 0.5，不报错。

        gemini-1.5-pro 的 parameter_size_b 为 null，但 scorer 不使用此字段。
        更重要的是测试 benchmark 等字段为 None 时的行为。
        """
        # 模拟 gemini-1.5-pro 的真实场景: parameter_size_b=null 不影响计算
        # (scorer 不使用 parameter_size_b，所以主要验证其他 None 字段)
        params_with_none_fields = {
            "parameter_size_b": None,  # 不参与评分计算
            "context_window": 2000000,
            "benchmark_mmlu": 85.9,
            "benchmark_humaneval": 71.9,
            "benchmark_math": 67.7,
            "cost_per_1m_input": 1.25,
        }
        # 不应抛出异常
        score = scorer.compute_score(params_with_none_fields)
        assert 0.0 <= score <= 1.0

        # 与正常参数的 gemini 分数一致（parameter_size_b 不参与计算）
        assert score == pytest.approx(scorer.compute_score(GEMINI_15_PRO_PARAMS), abs=0.001)

    def test_tc_score_004_benchmark_field_none_uses_midpoint(self, scorer):
        """TC-SCORE-004 补充: benchmark 字段为 None 时归一化为 0.5 (中位值)。"""
        # 仅 benchmark_mmlu 为 None，其余正常
        params_mmlu_none = {
            "context_window": 128000,
            "benchmark_mmlu": None,  # 应归一化为 0.5
            "benchmark_humaneval": 80.0,
            "benchmark_math": 70.0,
            "cost_per_1m_input": 1.0,
        }
        # 不应抛出异常
        score = scorer.compute_score(params_mmlu_none)
        assert 0.0 <= score <= 1.0

        # 对比 benchmark_mmlu 设为中间值的情况
        # normalize(None, 50, 95) = 0.5 → 等效于 benchmark_mmlu = 72.5 (midpoint of [50,95])
        params_mmlu_midpoint = {
            "context_window": 128000,
            "benchmark_mmlu": 72.5,  # (50+95)/2 = 72.5 → normalize = 0.5
            "benchmark_humaneval": 80.0,
            "benchmark_math": 70.0,
            "cost_per_1m_input": 1.0,
        }
        score_midpoint = scorer.compute_score(params_mmlu_midpoint)
        assert score == pytest.approx(score_midpoint, abs=0.001)

    def test_score_in_valid_range(self, scorer):
        """所有模型分数应在 [0, 1] 范围内"""
        for params in [LOCAL_7B_PARAMS, DEEPSEEK_V3_PARAMS, GEMINI_15_PRO_PARAMS, GPT_4O_PARAMS, O1_PARAMS]:
            score = scorer.compute_score(params)
            assert 0.0 <= score <= 1.0

    def test_empty_params_uses_defaults(self, scorer):
        """空参数应使用默认值"""
        score = scorer.compute_score({})
        # benchmark 全部 None → 0.5, context_window 默认 4096, cost 默认 0
        assert 0.0 <= score <= 1.0

    def test_cost_efficiency_inverse(self, scorer):
        """成本越低，性价比分数越高"""
        low_cost_params = {"cost_per_1m_input": 0.0}
        high_cost_params = {"cost_per_1m_input": 20.0}
        # 只看 cost_efficiency 维度
        low_cost_score = scorer.compute_score(low_cost_params)
        high_cost_score = scorer.compute_score(high_cost_params)
        assert low_cost_score > high_cost_score

    def test_missing_benchmark_treated_as_none(self, scorer):
        """缺失的 benchmark 字段应视为 None (归一化为 0.5)"""
        # 使用 90.0 归一化为 (90-50)/(95-50) = 0.889，不同于默认的 0.5
        params_with_mmlu = {"benchmark_mmlu": 90.0}
        params_without_mmlu = {}
        score_with = scorer.compute_score(params_with_mmlu)
        score_without = scorer.compute_score(params_without_mmlu)
        assert score_with != score_without


# ============================================================
# 路由区间测试
# ============================================================


class TestComputeScoreRange:
    """compute_score_range 方法测试 — TC-SCORE-005"""

    def test_tc_score_005_range_equals_score_plus_minus_tolerance(self, scorer):
        """TC-SCORE-005: score_range = (max(0, computed_score - tolerance), min(1, computed_score + tolerance))"""
        for params in [DEEPSEEK_V3_PARAMS, GPT_4O_PARAMS, O1_PARAMS]:
            score = scorer.compute_score(params)
            score_range = scorer.compute_score_range(params)
            expected_lower = max(0.0, score - TOLERANCE)
            expected_upper = min(1.0, score + TOLERANCE)
            assert score_range[0] == pytest.approx(expected_lower, abs=1e-9)
            assert score_range[1] == pytest.approx(expected_upper, abs=1e-9)

    def test_tc_score_005_range_clamped_at_zero(self, scorer):
        """TC-SCORE-005: 当 computed_score - tolerance < 0 时，下界 clamp 到 0"""
        # local-7b score ≈ 0.43, tolerance=0.15, so lower = 0.28 (不触发 clamp)
        # 创建一个极低分模型来触发 lower clamp
        very_low_params = {
            "benchmark_mmlu": 50.0,  # min
            "benchmark_humaneval": 30.0,  # min
            "benchmark_math": 20.0,  # min
            "context_window": 4096,  # min
            "cost_per_1m_input": 20.0,  # max cost → cost_efficiency = 0
        }
        score = scorer.compute_score(very_low_params)
        assert score < TOLERANCE, f"Score {score} should be less than tolerance {TOLERANCE}"
        score_range = scorer.compute_score_range(very_low_params)
        assert score_range[0] == 0.0, "Lower bound should be clamped to 0"

    def test_tc_score_005_range_clamped_at_one(self, scorer):
        """TC-SCORE-005: 当 computed_score + tolerance > 1 时，上界 clamp 到 1"""
        # 创建一个极高分模型（低成本 + 高 benchmark）
        very_high_params = {
            "benchmark_mmlu": 95.0,  # max
            "benchmark_humaneval": 95.0,  # max
            "benchmark_math": 95.0,  # max
            "context_window": 2000000,  # max
            "cost_per_1m_input": 0.0,  # free → cost_efficiency = 1.0
        }
        score = scorer.compute_score(very_high_params)
        assert score > (1.0 - TOLERANCE), f"Score {score} should be > {1.0 - TOLERANCE}"
        score_range = scorer.compute_score_range(very_high_params)
        assert score_range[1] == 1.0, "Upper bound should be clamped to 1.0"

    def test_range_width_is_double_tolerance(self, scorer):
        """区间宽度应为 2 * tolerance（除非被 clamp）"""
        score_range = scorer.compute_score_range(DEEPSEEK_V3_PARAMS)
        width = score_range[1] - score_range[0]
        # 如果没有被 clamp，宽度应为 0.30
        assert width <= 2 * TOLERANCE + 0.001

    def test_range_contains_score(self, scorer):
        """分数应在区间内"""
        for params in [LOCAL_7B_PARAMS, DEEPSEEK_V3_PARAMS, O1_PARAMS]:
            score = scorer.compute_score(params)
            score_range = scorer.compute_score_range(params)
            assert score_range[0] <= score <= score_range[1]


# ============================================================
# 路由表构建测试
# ============================================================


MODELS_CONFIG = [
    {"name": "local-7b", "litellm_model": "ollama/qwen2-7b", "params": LOCAL_7B_PARAMS},
    {"name": "deepseek-v3", "litellm_model": "deepseek/deepseek-chat", "params": DEEPSEEK_V3_PARAMS},
    {"name": "gemini-1.5-pro", "litellm_model": "gemini/gemini-1.5-pro", "params": GEMINI_15_PRO_PARAMS},
    {"name": "gpt-4o", "litellm_model": "openai/gpt-4o", "params": GPT_4O_PARAMS},
    {"name": "o1", "litellm_model": "openai/o1", "params": O1_PARAMS},
]


class TestBuildRoutingTable:
    """build_routing_table 函数测试 — 包含 TC-SCORE-006"""

    def test_sorted_by_computed_score_ascending(self, scorer):
        """路由表应按 computed_score 升序排列"""
        table = build_routing_table(MODELS_CONFIG, scorer, overrides={})
        scores = [tier["computed_score"] for tier in table]
        assert scores == sorted(scores)

    def test_all_models_present(self, scorer):
        """所有模型都应出现在路由表中"""
        table = build_routing_table(MODELS_CONFIG, scorer, overrides={})
        names = {tier["name"] for tier in table}
        assert names == {"local-7b", "deepseek-v3", "gemini-1.5-pro", "gpt-4o", "o1"}

    def test_without_overrides_all_not_overridden(self, scorer):
        """无覆盖时所有模型 overridden 应为 False"""
        table = build_routing_table(MODELS_CONFIG, scorer, overrides={})
        for tier in table:
            assert tier["overridden"] is False

    def test_tc_score_006_overrides_take_priority(self, scorer):
        """TC-SCORE-006: 人工覆盖 (route_overrides.yaml) 优先于自动计算。

        当 overrides 字典传入时，对应模型应使用覆盖的 score_range，
        而非 computed_score ± tolerance 自动生成的 score_range。
        """
        overrides = {
            "gpt-4o": {"score_range": [0.50, 0.82]},
            "local-7b": {"score_range": [0.0, 0.18]},
        }
        table = build_routing_table(MODELS_CONFIG, scorer, overrides=overrides)

        for tier in table:
            if tier["name"] == "gpt-4o":
                # 覆盖值应被使用
                assert tier["score_range"] == (0.50, 0.82)
                assert tier["overridden"] is True
                # computed_score 仍然是自动计算的（不受覆盖影响）
                expected_score = scorer.compute_score(GPT_4O_PARAMS)
                assert tier["computed_score"] == pytest.approx(expected_score)
                # 覆盖的 range 不等于自动计算的 range
                auto_range = scorer.compute_score_range(GPT_4O_PARAMS)
                assert tier["score_range"] != auto_range
            elif tier["name"] == "local-7b":
                assert tier["score_range"] == (0.0, 0.18)
                assert tier["overridden"] is True
            else:
                # 未被覆盖的模型使用自动计算
                assert tier["overridden"] is False
                expected_range = scorer.compute_score_range(tier["name"] == "deepseek-v3" and DEEPSEEK_V3_PARAMS or tier["name"] == "gemini-1.5-pro" and GEMINI_15_PRO_PARAMS or O1_PARAMS)
                assert tier["score_range"][0] == pytest.approx(expected_range[0], abs=1e-9)
                assert tier["score_range"][1] == pytest.approx(expected_range[1], abs=1e-9)

    def test_tc_score_006_override_does_not_affect_computed_score(self, scorer):
        """TC-SCORE-006 补充: 覆盖 score_range 不应影响 computed_score 的计算。"""
        overrides = {"o1": {"score_range": [0.80, 1.0]}}
        table = build_routing_table(MODELS_CONFIG, scorer, overrides=overrides)

        o1_tier = next(t for t in table if t["name"] == "o1")
        # computed_score 不受覆盖影响
        assert o1_tier["computed_score"] == pytest.approx(scorer.compute_score(O1_PARAMS))
        # score_range 使用覆盖值
        assert o1_tier["score_range"] == (0.80, 1.0)
        assert o1_tier["overridden"] is True

    def test_tier_structure(self, scorer):
        """每个 tier 应包含所有必需字段"""
        table = build_routing_table(MODELS_CONFIG, scorer, overrides={})
        required_keys = {"name", "model", "computed_score", "score_range", "cost_per_1m_input", "overridden"}
        for tier in table:
            assert set(tier.keys()) == required_keys

    def test_score_range_is_tuple(self, scorer):
        """score_range 应为 tuple"""
        table = build_routing_table(MODELS_CONFIG, scorer, overrides={})
        for tier in table:
            assert isinstance(tier["score_range"], tuple)
            assert len(tier["score_range"]) == 2

    def test_local_7b_is_first_in_table(self, scorer):
        """local-7b 应排在路由表最前面（分数最低）"""
        table = build_routing_table(MODELS_CONFIG, scorer, overrides={})
        assert table[0]["name"] == "local-7b"

    def test_cost_per_1m_input_preserved(self, scorer):
        """cost_per_1m_input 应正确保留"""
        table = build_routing_table(MODELS_CONFIG, scorer, overrides={})
        cost_map = {tier["name"]: tier["cost_per_1m_input"] for tier in table}
        assert cost_map["local-7b"] == 0.0
        assert cost_map["gpt-4o"] == 2.50
        assert cost_map["o1"] == 15.00

    def test_empty_models_config(self, scorer):
        """空模型列表应返回空路由表"""
        table = build_routing_table([], scorer, overrides={})
        assert table == []
