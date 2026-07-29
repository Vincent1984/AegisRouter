"""CapabilityProfile 和 CapabilityProfileManager 单元测试

覆盖:
- V2-1: 配置文件不存在时使用内置默认 Profile，无报错
- Profile 加载和评分计算
- 硬约束过滤逻辑
- 未知 Profile 名称降级为 'medium' 并发出警告
"""

import pytest
import tempfile
import os
from pathlib import Path

from aegis_router.router.capability_profiles import (
    CapabilityProfile,
    CapabilityProfileManager,
    DEFAULT_PROFILES,
    DEFAULT_NORMALIZATION,
)


# --- 测试用模型数据 ---

LOCAL_7B = {
    "name": "local-7b",
    "params": {
        "context_window": 32000,
        "benchmark_mmlu": 65.0,
        "benchmark_humaneval": 45.0,
        "benchmark_math": 40.0,
        "cost_per_1m_input": 0.0,
    },
}

DEEPSEEK_V4_PRO = {
    "name": "deepseek-v4-pro",
    "params": {
        "context_window": 128000,
        "benchmark_mmlu": 90.2,
        "benchmark_humaneval": 88.5,
        "benchmark_math": 82.0,
        "cost_per_1m_input": 0.27,
    },
}

GPT_55 = {
    "name": "gpt-5.5",
    "params": {
        "context_window": 1050000,
        "benchmark_mmlu": 92.0,
        "benchmark_humaneval": 93.5,
        "benchmark_math": 88.0,
        "cost_per_1m_input": 5.00,
    },
}

GPT_56_SOL = {
    "name": "gpt-5.6-sol",
    "params": {
        "context_window": 1050000,
        "benchmark_mmlu": 94.5,
        "benchmark_humaneval": 96.0,
        "benchmark_math": 95.0,
        "cost_per_1m_input": 15.00,
    },
}

CODEX_MINI = {
    "name": "codex-mini",
    "params": {
        "context_window": 200000,
        "benchmark_mmlu": 75.0,
        "benchmark_humaneval": 92.0,
        "benchmark_math": 70.0,
        "cost_per_1m_input": 0.50,
    },
}

GEMINI_25_FLASH = {
    "name": "gemini-2.5-flash",
    "params": {
        "context_window": 1048576,
        "benchmark_mmlu": 86.0,
        "benchmark_humaneval": 78.0,
        "benchmark_math": 75.0,
        "cost_per_1m_input": 0.15,
    },
}

GEMINI_25_PRO = {
    "name": "gemini-2.5-pro",
    "params": {
        "context_window": 2097152,
        "benchmark_mmlu": 89.0,
        "benchmark_humaneval": 84.0,
        "benchmark_math": 80.0,
        "cost_per_1m_input": 1.25,
    },
}

GEMINI_31_PRO = {
    "name": "gemini-3.1-pro",
    "params": {
        "context_window": 1048576,
        "benchmark_mmlu": 91.5,
        "benchmark_humaneval": 89.0,
        "benchmark_math": 85.0,
        "cost_per_1m_input": 2.00,
    },
}

GPT_52 = {
    "name": "gpt-5.2",
    "params": {
        "context_window": 400000,
        "benchmark_mmlu": 90.0,
        "benchmark_humaneval": 91.5,
        "benchmark_math": 83.0,
        "cost_per_1m_input": 2.50,
    },
}

GPT_54_MINI = {
    "name": "gpt-5.4-mini",
    "params": {
        "context_window": 400000,
        "benchmark_mmlu": 87.0,
        "benchmark_humaneval": 85.0,
        "benchmark_math": 78.0,
        "cost_per_1m_input": 0.80,
    },
}

CLAUDE_SONNET = {
    "name": "claude-sonnet",
    "params": {
        "context_window": 200000,
        "benchmark_mmlu": 89.5,
        "benchmark_humaneval": 92.0,
        "benchmark_math": 80.5,
        "cost_per_1m_input": 3.00,
    },
}

ALL_MODELS = [
    LOCAL_7B, DEEPSEEK_V4_PRO, GPT_52, GPT_54_MINI, GPT_55, GPT_56_SOL,
    CODEX_MINI, GEMINI_25_FLASH, GEMINI_25_PRO, GEMINI_31_PRO, CLAUDE_SONNET,
]


# ============================================================
# CapabilityProfile 数据类测试
# ============================================================


class TestCapabilityProfile:
    """CapabilityProfile dataclass 测试"""

    def test_basic_creation(self):
        """Profile 可以正确创建"""
        profile = CapabilityProfile(
            name="test",
            description="测试 Profile",
            scoring_weights={"benchmark_mmlu": 0.5, "cost_efficiency": 0.5},
        )
        assert profile.name == "test"
        assert profile.description == "测试 Profile"
        assert profile.min_score_threshold == 0.0
        assert profile.max_cost_per_1m_input == 60.0
        assert profile.min_context_window is None
        assert profile.prefer_models == []

    def test_default_profiles_complete(self):
        """内置默认 Profile 应包含 6 种"""
        assert len(DEFAULT_PROFILES) == 6
        expected_names = {"lightweight", "medium", "strong_reasoning", "code_specialist", "long_context", "heavy"}
        assert set(DEFAULT_PROFILES.keys()) == expected_names

    def test_default_profile_weights_sum_to_one(self):
        """每个默认 Profile 的权重之和应为 1.0"""
        for name, profile in DEFAULT_PROFILES.items():
            total = sum(profile.scoring_weights.values())
            assert total == pytest.approx(1.0, abs=0.01), (
                f"Profile '{name}' weights sum to {total}, expected 1.0"
            )


# ============================================================
# CapabilityProfileManager 加载测试
# ============================================================


class TestProfileManagerLoading:
    """Profile 加载逻辑测试"""

    def test_v2_1_file_not_found_uses_defaults(self):
        """V2-1: 配置文件不存在时使用内置默认 Profile，无报错"""
        manager = CapabilityProfileManager(
            config_path="nonexistent/path/capability_profiles.yaml"
        )
        assert len(manager.profiles) == 6
        assert "lightweight" in manager.profiles
        assert "medium" in manager.profiles
        assert "heavy" in manager.profiles

    def test_empty_yaml_uses_defaults(self, tmp_path):
        """空 YAML 文件使用内置默认 Profile"""
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text("")
        manager = CapabilityProfileManager(config_path=str(config_file))
        assert len(manager.profiles) == 6

    def test_yaml_without_profiles_key_uses_defaults(self, tmp_path):
        """YAML 文件没有 'profiles' 键时使用内置默认值"""
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text("other_key: value\n")
        manager = CapabilityProfileManager(config_path=str(config_file))
        assert len(manager.profiles) == 6

    def test_load_from_yaml(self, tmp_path):
        """从 YAML 文件正确加载 Profile"""
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text(
            """
profiles:
  custom:
    description: "Custom Profile"
    scoring_weights:
      benchmark_mmlu: 0.50
      cost_efficiency: 0.50
    min_score_threshold: 0.40
    max_cost_per_1m_input: 5.0
    min_context_window: 100000
    prefer_models: [model-a, model-b]
""",
            encoding="utf-8",
        )
        manager = CapabilityProfileManager(config_path=str(config_file))
        assert len(manager.profiles) == 1
        assert "custom" in manager.profiles
        profile = manager.profiles["custom"]
        assert profile.name == "custom"
        assert profile.description == "Custom Profile"
        assert profile.scoring_weights == {"benchmark_mmlu": 0.50, "cost_efficiency": 0.50}
        assert profile.min_score_threshold == 0.40
        assert profile.max_cost_per_1m_input == 5.0
        assert profile.min_context_window == 100000
        assert profile.prefer_models == ["model-a", "model-b"]

    def test_invalid_yaml_uses_defaults(self, tmp_path):
        """无效 YAML 内容时使用内置默认值"""
        config_file = tmp_path / "profiles.yaml"
        config_file.write_text("{{{{invalid yaml content!!!!")
        manager = CapabilityProfileManager(config_path=str(config_file))
        assert len(manager.profiles) == 6


# ============================================================
# get_profile 测试
# ============================================================


class TestGetProfile:
    """get_profile 方法测试"""

    def test_get_existing_profile(self):
        """获取已存在的 Profile"""
        manager = CapabilityProfileManager(config_path="nonexistent_path.yaml")
        profile = manager.get_profile("lightweight")
        assert profile.name == "lightweight"
        assert profile.max_cost_per_1m_input == 0.5

    def test_unknown_profile_falls_back_to_medium(self, caplog):
        """未知 Profile 降级为 'medium' 并发出警告"""
        manager = CapabilityProfileManager(config_path="nonexistent_path.yaml")
        import logging

        with caplog.at_level(logging.WARNING):
            profile = manager.get_profile("nonexistent_profile")

        assert profile.name == "medium"
        assert "not found" in caplog.text
        assert "fallback" in caplog.text.lower()

    def test_get_all_default_profiles(self):
        """所有 6 种默认 Profile 均可获取"""
        manager = CapabilityProfileManager(config_path="nonexistent_path.yaml")
        for name in ["lightweight", "medium", "strong_reasoning", "code_specialist", "long_context", "heavy"]:
            profile = manager.get_profile(name)
            assert profile.name == name


# ============================================================
# score_model 测试
# ============================================================


class TestScoreModel:
    """score_model 方法测试"""

    @pytest.fixture
    def manager(self):
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    def test_score_in_valid_range(self, manager):
        """所有模型分数应在 [0, 1] 范围内"""
        for profile_name in DEFAULT_PROFILES:
            profile = manager.get_profile(profile_name)
            for model in ALL_MODELS:
                score = manager.score_model(model, profile)
                assert 0.0 <= score <= 1.0, (
                    f"Model '{model['name']}' with profile '{profile_name}' "
                    f"scored {score}, out of [0, 1]"
                )

    def test_lightweight_favors_low_cost(self, manager):
        """lightweight Profile 应偏好低成本模型"""
        profile = manager.get_profile("lightweight")
        score_local = manager.score_model(LOCAL_7B, profile)
        score_gpt56 = manager.score_model(GPT_56_SOL, profile)
        # local-7b 免费, gpt-5.6-sol $15 — lightweight 偏好低成本
        assert score_local > score_gpt56

    def test_heavy_favors_high_capability(self, manager):
        """heavy Profile 应偏好高能力模型"""
        profile = manager.get_profile("heavy")
        score_gpt56 = manager.score_model(GPT_56_SOL, profile)
        score_local = manager.score_model(LOCAL_7B, profile)
        # gpt-5.6-sol benchmark 最高
        assert score_gpt56 > score_local

    def test_code_specialist_favors_humaneval(self, manager):
        """code_specialist Profile 应偏好 humaneval 高分模型"""
        profile = manager.get_profile("code_specialist")
        score_codex = manager.score_model(CODEX_MINI, profile)
        score_flash = manager.score_model(GEMINI_25_FLASH, profile)
        # codex-mini humaneval=92 vs flash humaneval=78
        assert score_codex > score_flash

    def test_long_context_favors_large_window(self, manager):
        """long_context Profile 应偏好大上下文窗口模型"""
        profile = manager.get_profile("long_context")
        score_gpt55 = manager.score_model(GPT_55, profile)
        score_local = manager.score_model(LOCAL_7B, profile)
        # gpt-5.5 context=1050000 vs local-7b context=32000
        assert score_gpt55 > score_local

    def test_empty_params_does_not_crash(self, manager):
        """空参数模型不应崩溃"""
        profile = manager.get_profile("medium")
        model = {"name": "empty", "params": {}}
        score = manager.score_model(model, profile)
        assert 0.0 <= score <= 1.0

    def test_none_benchmark_uses_midpoint(self, manager):
        """None 值的 benchmark 应使用中位值 0.5"""
        profile = manager.get_profile("medium")
        model_with_none = {
            "name": "test",
            "params": {
                "context_window": 128000,
                "benchmark_mmlu": None,
                "benchmark_humaneval": 80.0,
                "benchmark_math": 70.0,
                "cost_per_1m_input": 1.0,
            },
        }
        score = manager.score_model(model_with_none, profile)
        assert 0.0 <= score <= 1.0

    def test_v2_2_gemini_25_pro_long_context_high_score(self, manager):
        """V2-2: gemini-2.5-pro 因 context_window 权重高而在 long_context Profile 中得最高分

        long_context Profile:
          - context_window 权重 = 0.50 (50%)
          - min_context_window = 500000
          - max_cost_per_1m_input = 10.0

        gemini-2.5-pro:
          - context_window = 2,097,152 (模型池最大值，归一化到 1.0)
          - cost_per_1m_input = 1.25 (满足成本约束)

        因此 gemini-2.5-pro 应在所有模型中获得 long_context 最高分。
        """
        profile = manager.get_profile("long_context")

        # 对所有模型打分
        scores = {
            model["name"]: manager.score_model(model, profile)
            for model in ALL_MODELS
        }

        gemini_pro_score = scores["gemini-2.5-pro"]

        # gemini-2.5-pro 应获得所有模型中的最高分
        for model_name, score in scores.items():
            if model_name != "gemini-2.5-pro":
                assert gemini_pro_score > score, (
                    f"gemini-2.5-pro (score={gemini_pro_score:.4f}) should score higher "
                    f"than {model_name} (score={score:.4f}) for long_context profile"
                )

        # 分数应当较高（> 0.6），因为 context_window 归一化到 1.0 且权重 50%
        assert gemini_pro_score > 0.6, (
            f"gemini-2.5-pro score={gemini_pro_score:.4f} should be > 0.6 for long_context"
        )


# ============================================================
# filter_by_constraints 测试
# ============================================================


class TestFilterByConstraints:
    """filter_by_constraints 方法测试"""

    @pytest.fixture
    def manager(self):
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    def test_lightweight_filters_high_cost(self, manager):
        """lightweight (max_cost=0.5) 应过滤掉高成本模型"""
        profile = manager.get_profile("lightweight")
        candidates = manager.filter_by_constraints(ALL_MODELS, profile)
        candidate_names = [m["name"] for m in candidates]
        # local-7b cost=0, gemini-2.5-flash cost=0.15, deepseek cost=0.27
        # codex-mini cost=0.50 (边界), gpt-5.5 cost=5.00, gpt-5.6-sol cost=15.00
        assert "gpt-5.5" not in candidate_names
        assert "gpt-5.6-sol" not in candidate_names
        # 免费和低成本模型应保留
        assert "local-7b" in candidate_names

    def test_long_context_filters_small_window(self, manager):
        """long_context (min_context_window=500000) 应过滤小窗口模型"""
        profile = manager.get_profile("long_context")
        candidates = manager.filter_by_constraints(ALL_MODELS, profile)
        candidate_names = [m["name"] for m in candidates]
        # local-7b context=32000, deepseek context=128000, codex context=200000
        # 都不满足 500000 的要求
        assert "local-7b" not in candidate_names
        assert "deepseek-v4-pro" not in candidate_names
        assert "codex-mini" not in candidate_names

    def test_heavy_filters_low_score_models(self, manager):
        """heavy (min_score_threshold=0.75) 应过滤低分模型"""
        profile = manager.get_profile("heavy")
        candidates = manager.filter_by_constraints(ALL_MODELS, profile)
        candidate_names = [m["name"] for m in candidates]
        # local-7b 能力太弱，不应通过 heavy 的门槛
        assert "local-7b" not in candidate_names

    def test_no_constraints_profile_keeps_all(self, manager):
        """无约束 Profile 应保留所有模型"""
        # 创建一个几乎无约束的 Profile
        no_constraint_profile = CapabilityProfile(
            name="no_constraint",
            description="无约束",
            scoring_weights={
                "benchmark_mmlu": 0.20,
                "benchmark_humaneval": 0.20,
                "benchmark_math": 0.20,
                "context_window": 0.20,
                "cost_efficiency": 0.20,
            },
            min_score_threshold=0.0,
            max_cost_per_1m_input=100.0,
            min_context_window=None,
        )
        candidates = manager.filter_by_constraints(ALL_MODELS, no_constraint_profile)
        assert len(candidates) == len(ALL_MODELS)

    def test_empty_model_list(self, manager):
        """空模型列表应返回空列表"""
        profile = manager.get_profile("medium")
        candidates = manager.filter_by_constraints([], profile)
        assert candidates == []

    def test_constraint_checks_all_three_conditions(self, manager):
        """约束检查应同时检查分数、成本和上下文长度"""
        # 创建一个严格的 Profile: 需要高分、低成本、大窗口
        strict_profile = CapabilityProfile(
            name="strict",
            description="严格约束",
            scoring_weights={
                "benchmark_mmlu": 0.30,
                "benchmark_humaneval": 0.30,
                "benchmark_math": 0.30,
                "context_window": 0.05,
                "cost_efficiency": 0.05,
            },
            min_score_threshold=0.80,
            max_cost_per_1m_input=6.0,
            min_context_window=500000,
        )
        candidates = manager.filter_by_constraints(ALL_MODELS, strict_profile)
        # 需要满足: score >= 0.80, cost <= 6.0, context >= 500000
        # gpt-5.5: cost=5.0 ✓, context=1050000 ✓, 高 benchmark 应达到 0.80
        candidate_names = [m["name"] for m in candidates]
        # local-7b: context=32000 ✗
        assert "local-7b" not in candidate_names
        # gpt-5.6-sol: cost=15 ✗
        assert "gpt-5.6-sol" not in candidate_names


# ============================================================
# 归一化测试
# ============================================================


class TestNormalize:
    """_normalize 静态方法测试"""

    def test_none_returns_midpoint(self):
        """None 值应返回 0.5"""
        assert CapabilityProfileManager._normalize(None, 50, 95) == 0.5

    def test_min_returns_zero(self):
        """最小值应返回 0.0"""
        assert CapabilityProfileManager._normalize(50, 50, 95) == 0.0

    def test_max_returns_one(self):
        """最大值应返回 1.0"""
        assert CapabilityProfileManager._normalize(95, 50, 95) == 1.0

    def test_below_min_clamped(self):
        """低于最小值应 clamp 到 0.0"""
        assert CapabilityProfileManager._normalize(30, 50, 95) == 0.0

    def test_above_max_clamped(self):
        """高于最大值应 clamp 到 1.0"""
        assert CapabilityProfileManager._normalize(100, 50, 95) == 1.0

    def test_equal_min_max_returns_midpoint(self):
        """min == max 时返回 0.5"""
        assert CapabilityProfileManager._normalize(50, 50, 50) == 0.5


# ============================================================
# select_best_model 测试 (V2-4: prefer_models 优先选中)
# ============================================================


class TestSelectBestModel:
    """select_best_model 方法测试 — V2-4 验证"""

    @pytest.fixture
    def manager(self):
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    def test_v2_4_prefer_models_prioritized(self, manager):
        """V2-4: prefer_models 列表中的模型在满足约束时优先选中

        code_specialist Profile:
          - prefer_models = ["codex-mini", "gpt-5.5"]
          - max_cost_per_1m_input = 10.0
          - min_score_threshold = 0.50

        场景: codex-mini 不是最高分模型，但因在 prefer_models 列表中而被优先选中。
        """
        profile = manager.get_profile("code_specialist")

        # 确认 prefer_models 设置正确
        assert "codex-mini" in profile.prefer_models
        assert "gpt-5.5" in profile.prefer_models

        # 使用包含多个模型的集合，其中有得分更高的非偏好模型
        models = [CODEX_MINI, GPT_52, CLAUDE_SONNET, DEEPSEEK_V4_PRO, GEMINI_31_PRO]

        # 验证 codex-mini 不是最高分（有其他模型得分更高）
        scores = {
            m["name"]: manager.score_model(m, profile) for m in models
        }
        codex_score = scores["codex-mini"]
        has_higher_scorer = any(
            s > codex_score for name, s in scores.items() if name != "codex-mini"
        )
        assert has_higher_scorer, (
            "Test setup error: codex-mini should NOT be the highest-scored model"
        )

        # 执行选择 — codex-mini 应该因 prefer_models 而被优先选中
        selected = manager.select_best_model(models, profile)
        assert selected == "codex-mini", (
            f"Expected 'codex-mini' (preferred), got '{selected}'. "
            f"Scores: {scores}"
        )

    def test_v2_4_prefer_models_fallback_when_none_satisfy_constraints(self, manager):
        """当 prefer_models 中的模型都不满足约束时，回退到最高分模型

        场景: 使用 lightweight Profile (max_cost=0.5)，codex-mini (cost=0.50) 刚好在边界。
        创建一个严格约束 Profile 使 prefer_models 中的模型都被淘汰。
        """
        # 创建一个有 prefer_models 但约束很严格的 Profile
        strict_profile = CapabilityProfile(
            name="strict_prefer",
            description="严格约束 + 偏好模型",
            scoring_weights={
                "benchmark_mmlu": 0.20,
                "benchmark_humaneval": 0.40,
                "benchmark_math": 0.20,
                "context_window": 0.10,
                "cost_efficiency": 0.10,
            },
            min_score_threshold=0.0,
            max_cost_per_1m_input=0.3,  # 非常低的成本上限
            prefer_models=["codex-mini", "gpt-5.5"],  # 这两个都超过 0.3 的成本限制
        )

        # codex-mini cost=0.50 > 0.3, gpt-5.5 cost=5.00 > 0.3 — 都被淘汰
        models = [LOCAL_7B, DEEPSEEK_V4_PRO, CODEX_MINI, GPT_55, GEMINI_25_FLASH]

        selected = manager.select_best_model(models, strict_profile)

        # 偏好模型都不满足约束，应回退到最高分模型
        assert selected is not None
        assert selected != "codex-mini"
        assert selected != "gpt-5.5"

        # 验证选中的是通过约束的最高分模型
        candidates = manager.filter_by_constraints(models, strict_profile)
        candidate_names = [m["name"] for m in candidates]
        assert selected in candidate_names

        # 确认选中的确实是最高分
        scored_candidates = [
            (m["name"], manager.score_model(m, strict_profile))
            for m in candidates
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        assert selected == scored_candidates[0][0]

    def test_select_best_model_empty_candidates(self, manager):
        """没有模型满足约束时返回 None"""
        # 使用非常严格的约束
        impossible_profile = CapabilityProfile(
            name="impossible",
            description="不可能满足的约束",
            scoring_weights={
                "benchmark_mmlu": 0.50,
                "benchmark_humaneval": 0.50,
            },
            min_score_threshold=0.99,  # 几乎不可能达到的门槛
            max_cost_per_1m_input=0.01,  # 几乎免费
            min_context_window=10000000,  # 超大窗口
        )

        selected = manager.select_best_model(ALL_MODELS, impossible_profile)
        assert selected is None

    def test_select_best_model_no_prefer_models_uses_highest_score(self, manager):
        """无 prefer_models 时选择最高分模型"""
        # medium Profile 没有 prefer_models
        profile = manager.get_profile("medium")
        assert profile.prefer_models == []

        models = [LOCAL_7B, DEEPSEEK_V4_PRO, GEMINI_25_FLASH]

        selected = manager.select_best_model(models, profile)

        # 应选择通过约束后得分最高的模型
        candidates = manager.filter_by_constraints(models, profile)
        if candidates:
            scored = [
                (m["name"], manager.score_model(m, profile))
                for m in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            assert selected == scored[0][0]


# ============================================================
# TC-PROFILE 评分单元测试 (Task 18)
# ============================================================


class TestProfileScoring:
    """TC-PROFILE-001 ~ TC-PROFILE-007: Profile 评分与选择逻辑"""

    @pytest.fixture
    def manager(self):
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    def test_tc_profile_001_lightweight_local_7b_highest(self, manager):
        """TC-PROFILE-001: lightweight Profile → local-7b 在满足成本约束的模型中得最高分（成本权重 75%）

        lightweight Profile:
          - cost_efficiency 权重 = 0.75 (75%)
          - max_cost_per_1m_input = 0.5
          - prefer_models = ["local-7b"]

        local-7b:
          - cost_per_1m_input = 0.0 (免费，成本效率归一化到 1.0)

        通过 select_best_model 流程，local-7b 作为 prefer_models 中的模型，
        在满足硬约束后被优先选中。
        """
        profile = manager.get_profile("lightweight")

        # 验证 local-7b 有最高的成本效率分数（cost=0 → cost_efficiency=1.0）
        # cost_efficiency 贡献 = weight(0.75) * (1.0 - normalize(0, 0, 20)) = 0.75
        local_score = manager.score_model(LOCAL_7B, profile)
        assert local_score > 0.75, (
            f"local-7b score={local_score:.4f} should be > 0.75 "
            f"(cost_efficiency alone contributes 0.75)"
        )

        # 通过 select_best_model 选择，local-7b 作为 prefer_models 被优先选中
        selected = manager.select_best_model(ALL_MODELS, profile)
        assert selected == "local-7b", (
            f"Expected 'local-7b' (preferred model in lightweight), got '{selected}'"
        )

    def test_tc_profile_002_long_context_gemini_25_pro_highest(self, manager):
        """TC-PROFILE-002: long_context Profile → gemini-2.5-pro 得最高分（上下文权重 50%）

        long_context Profile:
          - context_window 权重 = 0.50 (50%)
          - min_context_window = 500000

        gemini-2.5-pro:
          - context_window = 2,097,152 (模型池最大值，归一化接近 1.0)
          - cost_per_1m_input = 1.25 (低成本)

        因此 gemini-2.5-pro 应在所有模型中获得 long_context 最高分。
        """
        profile = manager.get_profile("long_context")

        scores = {
            model["name"]: manager.score_model(model, profile)
            for model in ALL_MODELS
        }

        gemini_pro_score = scores["gemini-2.5-pro"]

        for model_name, score in scores.items():
            if model_name != "gemini-2.5-pro":
                assert gemini_pro_score > score, (
                    f"gemini-2.5-pro (score={gemini_pro_score:.4f}) should score higher "
                    f"than {model_name} (score={score:.4f}) for long_context profile"
                )

    def test_tc_profile_003_strong_reasoning_gpt55_or_gpt56_highest(self, manager):
        """TC-PROFILE-003: strong_reasoning Profile → gpt-5.5/gpt-5.6-sol 得最高分

        strong_reasoning Profile:
          - benchmark_math 权重 = 0.35
          - benchmark_humaneval 权重 = 0.30
          - min_score_threshold = 0.60
          - max_cost_per_1m_input = 20.0

        gpt-5.5: math=88.0, humaneval=93.5, cost=5.00
        gpt-5.6-sol: math=95.0, humaneval=96.0, cost=15.00

        两者都有最高的 math+humaneval，具体谁赢取决于 cost_efficiency 权重影响。
        """
        profile = manager.get_profile("strong_reasoning")

        scores = {
            model["name"]: manager.score_model(model, profile)
            for model in ALL_MODELS
        }

        # 最高分应是 gpt-5.5 或 gpt-5.6-sol 之一
        top_model = max(scores, key=scores.get)
        assert top_model in ("gpt-5.5", "gpt-5.6-sol"), (
            f"Expected top model to be gpt-5.5 or gpt-5.6-sol, "
            f"got '{top_model}' (score={scores[top_model]:.4f}). "
            f"Scores: gpt-5.5={scores['gpt-5.5']:.4f}, gpt-5.6-sol={scores['gpt-5.6-sol']:.4f}"
        )

    def test_tc_profile_004_code_specialist_prefer_codex_mini(self, manager):
        """TC-PROFILE-004: code_specialist Profile + prefer_models=[codex-mini] → codex-mini 被选中

        code_specialist Profile:
          - prefer_models = ["codex-mini", "gpt-5.5"]
          - max_cost_per_1m_input = 10.0
          - min_score_threshold = 0.50

        codex-mini:
          - humaneval = 92.0 (高)
          - cost = 0.50 (满足 <= 10.0)

        场景: 当候选模型不包含 gpt-5.5 时，codex-mini 作为 prefer_models
        中唯一满足约束的模型被优先选中（即使非最高分）。
        """
        profile = manager.get_profile("code_specialist")

        # 排除 gpt-5.5，只保留 codex-mini 作为唯一 prefer_model 候选
        models_without_gpt55 = [m for m in ALL_MODELS if m["name"] != "gpt-5.5"]

        # 验证 codex-mini 满足约束
        candidates = manager.filter_by_constraints(models_without_gpt55, profile)
        candidate_names = [m["name"] for m in candidates]
        assert "codex-mini" in candidate_names, (
            "codex-mini should pass code_specialist constraints"
        )

        # 验证 codex-mini 不是最高分模型（有其他模型分数更高）
        scores = {
            m["name"]: manager.score_model(m, profile)
            for m in candidates
        }
        codex_score = scores["codex-mini"]
        has_higher = any(s > codex_score for n, s in scores.items() if n != "codex-mini")
        assert has_higher, "Test setup: codex-mini should not be the highest scorer"

        # select_best_model 应优先选择 prefer_models 中的 codex-mini
        selected = manager.select_best_model(models_without_gpt55, profile)
        assert selected == "codex-mini", (
            f"Expected 'codex-mini' (preferred model), got '{selected}'. "
            f"Scores: {scores}"
        )

    def test_tc_profile_005_long_context_filters_small_context_window(self, manager):
        """TC-PROFILE-005: 硬约束过滤 — context_window<500000 的模型被 long_context 淘汰

        long_context Profile:
          - min_context_window = 500000

        应被过滤的模型 (context_window < 500000):
          - local-7b: 32000
          - deepseek-v4-pro: 128000
          - codex-mini: 200000
          - claude-sonnet: 200000
          - gpt-5.2: 400000
          - gpt-5.4-mini: 400000
        """
        profile = manager.get_profile("long_context")
        candidates = manager.filter_by_constraints(ALL_MODELS, profile)
        candidate_names = [m["name"] for m in candidates]

        # 这些模型 context_window < 500000，应被过滤
        filtered_models = [
            "local-7b",        # 32000
            "deepseek-v4-pro", # 128000
            "codex-mini",      # 200000
            "claude-sonnet",   # 200000
            "gpt-5.2",        # 400000
            "gpt-5.4-mini",   # 400000
        ]
        for model_name in filtered_models:
            assert model_name not in candidate_names, (
                f"'{model_name}' should be filtered out by min_context_window=500000 "
                f"but was found in candidates: {candidate_names}"
            )

    def test_tc_profile_006_long_context_filters_high_cost(self, manager):
        """TC-PROFILE-006: 硬约束过滤 — cost>$10 的模型被 long_context 淘汰

        long_context Profile:
          - max_cost_per_1m_input = 10.0

        应被过滤的模型 (cost > 10.0):
          - gpt-5.6-sol: cost = 15.00
        """
        profile = manager.get_profile("long_context")
        candidates = manager.filter_by_constraints(ALL_MODELS, profile)
        candidate_names = [m["name"] for m in candidates]

        # gpt-5.6-sol cost=15.00 > 10.0，应被过滤
        assert "gpt-5.6-sol" not in candidate_names, (
            f"'gpt-5.6-sol' (cost=15.00) should be filtered out by max_cost=10.0 "
            f"but was found in candidates: {candidate_names}"
        )

    def test_tc_profile_007_nonexistent_profile_fallback_to_medium(self, manager, caplog):
        """TC-PROFILE-007: Profile 不存在时降级为 medium

        调用 get_profile("nonexistent_profile_xyz") 应:
        1. 返回 medium Profile
        2. 记录警告日志
        """
        import logging

        with caplog.at_level(logging.WARNING):
            profile = manager.get_profile("nonexistent_profile_xyz")

        # 应返回 medium Profile
        assert profile.name == "medium", (
            f"Expected fallback to 'medium', got '{profile.name}'"
        )

        # 应记录警告日志
        assert "nonexistent_profile_xyz" in caplog.text, (
            "Warning log should mention the missing profile name"
        )
        assert "not found" in caplog.text.lower() or "fallback" in caplog.text.lower(), (
            f"Warning log should mention 'not found' or 'fallback', got: {caplog.text}"
        )
