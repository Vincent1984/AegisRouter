"""V3-1 验证检查点: generate_all() 为 resume_screening 4 个 Agent 选出正确模型

使用完整 11 模型池 + 默认内置 Profile，验证 resume_screening 模板中
每个 Agent 的模型分配结果符合 Profile 评分逻辑预期。

Agent 及其 capability_profile:
1. intent_classifier   — lightweight      (cost_efficiency 权重 0.75)
2. resume_parser       — long_context     (context_window 权重 0.50, min_context=500000)
3. skill_matcher       — strong_reasoning (math 0.35 + humaneval 0.30)
4. compliance_checker  — medium           (平衡型, max_cost=3.0)
"""

import pytest

from aegis_router.router.capability_profiles import CapabilityProfileManager
from aegis_router.router.template_models import AgentDef, TemplateDef
from aegis_router.router.template_plan_generator import TemplatePlanGenerator


# --- 完整 11 模型池（来自 config/models.yaml） ---

FULL_MODEL_POOL = [
    {
        "name": "local-7b",
        "params": {
            "context_window": 32000,
            "benchmark_mmlu": 65.0,
            "benchmark_humaneval": 45.0,
            "benchmark_math": 40.0,
            "cost_per_1m_input": 0.0,
        },
    },
    {
        "name": "deepseek-v4-pro",
        "params": {
            "context_window": 128000,
            "benchmark_mmlu": 90.2,
            "benchmark_humaneval": 88.5,
            "benchmark_math": 82.0,
            "cost_per_1m_input": 0.27,
        },
    },
    {
        "name": "claude-sonnet",
        "params": {
            "context_window": 200000,
            "benchmark_mmlu": 89.5,
            "benchmark_humaneval": 92.0,
            "benchmark_math": 80.5,
            "cost_per_1m_input": 3.00,
        },
    },
    {
        "name": "gpt-5.2",
        "params": {
            "context_window": 400000,
            "benchmark_mmlu": 90.0,
            "benchmark_humaneval": 91.5,
            "benchmark_math": 83.0,
            "cost_per_1m_input": 2.50,
        },
    },
    {
        "name": "gpt-5.4-mini",
        "params": {
            "context_window": 400000,
            "benchmark_mmlu": 87.0,
            "benchmark_humaneval": 85.0,
            "benchmark_math": 78.0,
            "cost_per_1m_input": 0.80,
        },
    },
    {
        "name": "gpt-5.5",
        "params": {
            "context_window": 1050000,
            "benchmark_mmlu": 92.0,
            "benchmark_humaneval": 93.5,
            "benchmark_math": 88.0,
            "cost_per_1m_input": 5.00,
        },
    },
    {
        "name": "gpt-5.6-sol",
        "params": {
            "context_window": 1050000,
            "benchmark_mmlu": 94.5,
            "benchmark_humaneval": 96.0,
            "benchmark_math": 95.0,
            "cost_per_1m_input": 15.00,
        },
    },
    {
        "name": "codex-mini",
        "params": {
            "context_window": 200000,
            "benchmark_mmlu": 75.0,
            "benchmark_humaneval": 92.0,
            "benchmark_math": 70.0,
            "cost_per_1m_input": 0.50,
        },
    },
    {
        "name": "gemini-2.5-flash",
        "params": {
            "context_window": 1048576,
            "benchmark_mmlu": 86.0,
            "benchmark_humaneval": 78.0,
            "benchmark_math": 75.0,
            "cost_per_1m_input": 0.15,
        },
    },
    {
        "name": "gemini-2.5-pro",
        "params": {
            "context_window": 2097152,
            "benchmark_mmlu": 89.0,
            "benchmark_humaneval": 84.0,
            "benchmark_math": 80.0,
            "cost_per_1m_input": 1.25,
        },
    },
    {
        "name": "gemini-3.1-pro",
        "params": {
            "context_window": 1048576,
            "benchmark_mmlu": 91.5,
            "benchmark_humaneval": 89.0,
            "benchmark_math": 85.0,
            "cost_per_1m_input": 2.00,
        },
    },
]

FALLBACK_MODEL = "local-7b"


# --- resume_screening 模板（4 Agent，完整定义） ---

RESUME_SCREENING_TEMPLATE = {
    "resume_screening": TemplateDef(
        name="resume_screening",
        description="简历筛选流程",
        agents=[
            AgentDef(name="intent_classifier", capability_profile="lightweight"),
            AgentDef(name="resume_parser", capability_profile="long_context"),
            AgentDef(name="skill_matcher", capability_profile="strong_reasoning"),
            AgentDef(name="compliance_checker", capability_profile="medium"),
        ],
    ),
}


@pytest.fixture
def profile_manager():
    """使用内置默认 Profile（指定不存在的路径触发默认值）"""
    return CapabilityProfileManager(config_path="nonexistent_path.yaml")


@pytest.fixture
def generator(profile_manager):
    """使用完整 11 模型池的方案生成器"""
    return TemplatePlanGenerator(
        profile_manager=profile_manager,
        models=FULL_MODEL_POOL,
        fallback_model=FALLBACK_MODEL,
    )


class TestV3_1_ResumeScreeningFullModelPool:
    """V3-1: generate_all() 为 resume_screening 的 4 个 Agent 选出正确模型

    使用完整 11 模型池和默认内置 Profile 验证评分逻辑的正确性。
    """

    def test_all_four_agents_assigned(self, generator):
        """所有 4 个 Agent 均获得模型分配（非 None）"""
        store = generator.generate_all(RESUME_SCREENING_TEMPLATE)

        for agent_name in [
            "intent_classifier",
            "resume_parser",
            "skill_matcher",
            "compliance_checker",
        ]:
            model = store.get_model("resume_screening", agent_name)
            assert model is not None, (
                f"Agent '{agent_name}' should have a model assigned"
            )

    def test_intent_classifier_selects_lowest_cost(self, generator):
        """lightweight Profile: 成本权重 75%, max_cost=0.5

        满足 cost ≤ 0.5 的模型:
        - local-7b (cost=0.0) → cost_efficiency = 1.0
        - gemini-2.5-flash (cost=0.15) → cost_efficiency = 0.9925
        - deepseek-v4-pro (cost=0.27) → cost_efficiency = 0.9865
        - codex-mini (cost=0.50) → cost_efficiency = 0.975

        local-7b 的 cost_efficiency 分数最高 (1.0 × 0.75 = 0.75 单项)
        即使其他维度弱，75%的成本权重使其总分最高 → 选 local-7b
        """
        store = generator.generate_all(RESUME_SCREENING_TEMPLATE)
        model = store.get_model("resume_screening", "intent_classifier")
        assert model == "local-7b", (
            f"lightweight profile should select local-7b (lowest cost), got '{model}'"
        )

    def test_resume_parser_selects_large_context(self, generator):
        """long_context Profile: context_window 权重 50%, min_context=500000, max_cost=10

        硬约束过滤 (min_context_window=500000, max_cost≤10):
        - gpt-5.5 (1050000, cost=5.0) ✓
        - gpt-5.6-sol (1050000, cost=15.0) ✗ (cost>10)
        - gemini-2.5-flash (1048576, cost=0.15) ✓
        - gemini-2.5-pro (2097152, cost=1.25) ✓
        - gemini-3.1-pro (1048576, cost=2.0) ✓

        context_window 归一化（[4096, 2000000]）:
        - gemini-2.5-pro: (2097152-4096)/(2000000-4096) ≈ 1.0 (clamped)
        - gpt-5.5: (1050000-4096)/(2000000-4096) ≈ 0.524
        - gemini-2.5-flash: (1048576-4096)/(2000000-4096) ≈ 0.523
        - gemini-3.1-pro: same as flash ≈ 0.523

        gemini-2.5-pro context score = 1.0 × 0.50 = 0.50 (context alone)
        → gemini-2.5-pro should have highest total score
        """
        store = generator.generate_all(RESUME_SCREENING_TEMPLATE)
        model = store.get_model("resume_screening", "resume_parser")
        assert model == "gemini-2.5-pro", (
            f"long_context profile should select gemini-2.5-pro (largest context), "
            f"got '{model}'"
        )

    def test_skill_matcher_selects_strong_reasoning_model(self, generator):
        """strong_reasoning Profile: math=0.35, humaneval=0.30, mmlu=0.15,
        context=0.05, cost_efficiency=0.15, min_score_threshold=0.60, max_cost≤20

        All models with cost ≤ 20 pass the cost constraint. Key candidates:
        - gpt-5.6-sol (math=95, humaneval=96, cost=15) — highest reasoning scores
        - gpt-5.5 (math=88, humaneval=93.5, cost=5)

        gpt-5.6-sol should score highest on math+humaneval but lower on cost_efficiency.
        Need to check if it passes the min_score_threshold of 0.60.

        The scoring will determine which model wins. With math at 0.35 and humaneval at
        0.30, gpt-5.6-sol's superior benchmark scores should win despite higher cost.
        """
        store = generator.generate_all(RESUME_SCREENING_TEMPLATE)
        model = store.get_model("resume_screening", "skill_matcher")
        # gpt-5.6-sol or gpt-5.5 should win — both are strong reasoning models
        assert model in ["gpt-5.6-sol", "gpt-5.5"], (
            f"strong_reasoning profile should select a top reasoning model "
            f"(gpt-5.6-sol or gpt-5.5), got '{model}'"
        )

    def test_compliance_checker_selects_balanced_model(self, generator):
        """medium Profile: mmlu=0.25, humaneval=0.15, math=0.15, context=0.10,
        cost_efficiency=0.35, min_score=0.30, max_cost≤3.0

        硬约束过滤 (cost ≤ 3.0):
        - local-7b (0.0) ✓
        - deepseek-v4-pro (0.27) ✓
        - gemini-2.5-flash (0.15) ✓
        - gemini-2.5-pro (1.25) ✓
        - gemini-3.1-pro (2.00) ✓
        - gpt-5.4-mini (0.80) ✓
        - gpt-5.2 (2.50) ✓
        - claude-sonnet (3.00) ✓
        - codex-mini (0.50) ✓
        - gpt-5.5 (5.0) ✗
        - gpt-5.6-sol (15.0) ✗

        Medium profile balances benchmarks (55% total) with cost_efficiency (35%).
        deepseek-v4-pro has strong benchmarks AND very low cost (0.27) → should score well.
        """
        store = generator.generate_all(RESUME_SCREENING_TEMPLATE)
        model = store.get_model("resume_screening", "compliance_checker")
        # With cost_efficiency at 35% weight and max_cost=3.0 constraint,
        # deepseek-v4-pro (cost=0.27, strong benchmarks) should be the winner
        assert model == "deepseek-v4-pro", (
            f"medium profile should select deepseek-v4-pro "
            f"(strong benchmarks + low cost), got '{model}'"
        )

    def test_all_four_agents_get_distinct_sensible_models(self, generator):
        """验证 4 个 Agent 基于不同 Profile 选出合理且差异化的模型"""
        store = generator.generate_all(RESUME_SCREENING_TEMPLATE)

        assignments = {
            "intent_classifier": store.get_model("resume_screening", "intent_classifier"),
            "resume_parser": store.get_model("resume_screening", "resume_parser"),
            "skill_matcher": store.get_model("resume_screening", "skill_matcher"),
            "compliance_checker": store.get_model("resume_screening", "compliance_checker"),
        }

        # All assignments should be non-None
        for agent, model in assignments.items():
            assert model is not None, f"{agent} has no model assigned"

        # At least 3 distinct models (lightweight and medium might not overlap,
        # but strong_reasoning and long_context should definitely differ)
        unique_models = set(assignments.values())
        assert len(unique_models) >= 3, (
            f"Expected at least 3 distinct models across 4 different profiles, "
            f"got {len(unique_models)}: {assignments}"
        )

    def test_scoring_details_for_transparency(self, generator, profile_manager):
        """辅助测试: 输出各 Agent 的候选模型得分用于调试和验证"""
        profiles_to_check = ["lightweight", "long_context", "strong_reasoning", "medium"]

        for profile_name in profiles_to_check:
            profile = profile_manager.get_profile(profile_name)
            candidates = profile_manager.filter_by_constraints(FULL_MODEL_POOL, profile)

            # Verify we have candidates after filtering
            assert len(candidates) > 0, (
                f"Profile '{profile_name}' should have at least one candidate model"
            )

            # Score and sort
            scored = [
                (m["name"], profile_manager.score_model(m, profile))
                for m in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            # The top model should have a reasonable score
            top_name, top_score = scored[0]
            assert top_score > 0.3, (
                f"Profile '{profile_name}': top model '{top_name}' score "
                f"{top_score:.4f} is unexpectedly low"
            )
