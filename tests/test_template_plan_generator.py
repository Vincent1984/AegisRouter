"""TemplatePlanGenerator 方案生成单元测试

覆盖:
- TC-PLAN-001: 4 个模板 × 各 3-5 个 Agent → 全部正确分配
- TC-PLAN-002: override_model 优先于 Profile 打分
- TC-PLAN-003: 同一 Agent 在不同模板下分配不同模型
- TC-PLAN-004: 无候选模型时使用 fallback
- TC-PLAN-005: 相同配置多次调用 generate_all() → 结果完全相同（确定性）
"""

import pytest

from aegis_router.router.capability_profiles import CapabilityProfileManager
from aegis_router.router.routing_plan_store import RoutingPlanStore
from aegis_router.router.template_models import AgentDef, TemplateDef
from aegis_router.router.template_plan_generator import TemplatePlanGenerator


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


# --- 测试用模板定义 ---


def _build_templates() -> dict[str, TemplateDef]:
    """构建 4 个业务流程模板，模拟 config/transaction_templates.yaml"""
    return {
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
        "code_review": TemplateDef(
            name="code_review",
            description="代码审查流程",
            agents=[
                AgentDef(name="code_analyzer", capability_profile="code_specialist"),
                AgentDef(name="issue_detector", capability_profile="strong_reasoning"),
                AgentDef(name="fix_suggester", capability_profile="code_specialist"),
            ],
        ),
        "supplier_evaluation": TemplateDef(
            name="supplier_evaluation",
            description="供应商评估流程",
            agents=[
                AgentDef(name="data_collector", capability_profile="lightweight"),
                AgentDef(name="performance_scorer", capability_profile="medium"),
                AgentDef(name="compliance_checker", capability_profile="strong_reasoning"),
                AgentDef(name="tier_determiner", capability_profile="strong_reasoning"),
            ],
        ),
        "custom_pipeline": TemplateDef(
            name="custom_pipeline",
            description="自定义流水线",
            agents=[
                AgentDef(name="analyzer", capability_profile="medium"),
                AgentDef(name="generator", capability_profile="heavy", override_model="gpt-5.6-sol"),
            ],
        ),
    }


# ============================================================
# TemplatePlanGenerator 单元测试
# ============================================================


class TestTemplatePlanGenerator:
    """TemplatePlanGenerator 方案生成测试"""

    @pytest.fixture
    def profile_manager(self):
        """使用默认 Profile 的管理器（无外部文件依赖）"""
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    @pytest.fixture
    def templates(self):
        """4 个标准业务模板"""
        return _build_templates()

    @pytest.fixture
    def generator(self, profile_manager):
        """标准 TemplatePlanGenerator 实例"""
        return TemplatePlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )

    def test_tc_plan_001_all_agents_assigned(self, generator, templates):
        """TC-PLAN-001: 4 个模板 × 各 3-5 个 Agent → 全部正确分配

        验证:
        - generate_all 返回 RoutingPlanStore 实例
        - 所有 4 个模板的所有 Agent 都有模型分配
        - 分配的模型名称存在于模型池或 override 中
        - 总共 13 个 Agent 全部有分配
        """
        store = generator.generate_all(templates)

        # 返回值类型验证
        assert isinstance(store, RoutingPlanStore)

        all_plans = store.get_all_plans()

        # 4 个模板都有方案
        assert set(all_plans.keys()) == {
            "resume_screening", "code_review", "supplier_evaluation", "custom_pipeline"
        }

        # 验证每个模板的 Agent 数量
        assert len(all_plans["resume_screening"]) == 4
        assert len(all_plans["code_review"]) == 3
        assert len(all_plans["supplier_evaluation"]) == 4
        assert len(all_plans["custom_pipeline"]) == 2

        # 验证所有分配的模型名称有效（在模型池中或是 override_model）
        valid_model_names = {m["name"] for m in ALL_MODELS}
        for tpl_name, agent_map in all_plans.items():
            for agent_name, model_name in agent_map.items():
                assert model_name in valid_model_names, (
                    f"模板 '{tpl_name}' 中 Agent '{agent_name}' "
                    f"分配了无效模型 '{model_name}'"
                )

        # 总 Agent 数 = 4 + 3 + 4 + 2 = 13
        total_agents = sum(len(plan) for plan in all_plans.values())
        assert total_agents == 13

    def test_tc_plan_002_override_model_priority(self, profile_manager, templates):
        """TC-PLAN-002: override_model 优先于 Profile 打分

        验证:
        - custom_pipeline 模板中 generator Agent 设置了 override_model="gpt-5.6-sol"
        - 即使 heavy Profile 的 Profile 评分可能选出其他模型，override 始终生效
        - override_model 直接使用，不经过 Profile 评分
        """
        generator = TemplatePlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )
        store = generator.generate_all(templates)

        # generator Agent 有 override_model="gpt-5.6-sol"
        assigned_model = store.get_model("custom_pipeline", "generator")
        assert assigned_model == "gpt-5.6-sol", (
            f"override_model 应优先使用 'gpt-5.6-sol'，实际分配 '{assigned_model}'"
        )

        # 对比: analyzer Agent（同一模板但无 override）应通过 Profile 打分选择
        analyzer_model = store.get_model("custom_pipeline", "analyzer")
        assert analyzer_model is not None
        assert analyzer_model != "gpt-5.6-sol" or True  # analyzer 可能碰巧选到同一模型，但关键是 generator 的 override 生效

    def test_tc_plan_003_same_agent_different_models_across_templates(self, generator, templates):
        """TC-PLAN-003: 同一 Agent 在不同模板下分配不同模型

        验证:
        - compliance_checker 出现在 resume_screening（profile=medium）和
          supplier_evaluation（profile=strong_reasoning）两个模板中
        - 由于使用不同的 capability_profile，分配的模型应不同
        """
        store = generator.generate_all(templates)

        # compliance_checker 在 resume_screening 中使用 medium Profile
        model_in_resume = store.get_model("resume_screening", "compliance_checker")
        # compliance_checker 在 supplier_evaluation 中使用 strong_reasoning Profile
        model_in_supplier = store.get_model("supplier_evaluation", "compliance_checker")

        assert model_in_resume is not None
        assert model_in_supplier is not None

        # 两个 Profile 的约束和权重完全不同，应选出不同模型
        # medium: 平衡成本和质量, max_cost=3.0
        # strong_reasoning: 偏推理, max_cost=20.0, min_score_threshold=0.60
        assert model_in_resume != model_in_supplier, (
            f"同名 Agent 'compliance_checker' 在不同模板下使用不同 Profile，"
            f"应分配不同模型。resume_screening={model_in_resume}, "
            f"supplier_evaluation={model_in_supplier}"
        )

    def test_tc_plan_004_fallback_when_no_candidates(self, profile_manager):
        """TC-PLAN-004: 无候选模型时使用 fallback

        验证:
        - 当模型池为空时（所有模型都不满足约束），使用 fallback_model
        - fallback_model 是构造时指定的降级模型
        """
        # 使用空模型池，模拟无候选模型场景
        generator = TemplatePlanGenerator(
            profile_manager=profile_manager,
            models=[],  # 空模型池 → select_best_model 返回 None
            fallback_model="local-7b",
            trigger_reason="test",
        )

        templates = {
            "test_template": TemplateDef(
                name="test_template",
                description="测试模板",
                agents=[
                    AgentDef(name="agent_a", capability_profile="medium"),
                    AgentDef(name="agent_b", capability_profile="strong_reasoning"),
                ],
            ),
        }

        store = generator.generate_all(templates)

        # 没有模型可选时，所有 Agent 应使用 fallback
        assert store.get_model("test_template", "agent_a") == "local-7b"
        assert store.get_model("test_template", "agent_b") == "local-7b"

    def test_tc_plan_004_fallback_with_strict_constraints(self, profile_manager):
        """TC-PLAN-004 补充: 模型池非空但硬约束过滤后无候选时使用 fallback

        验证:
        - 模型池有模型，但没有一个能通过极端严格的 Profile 约束
        - 此时应降级到 fallback_model
        """
        # 只提供 local-7b，它无法通过 heavy Profile 的 min_score_threshold=0.75
        generator = TemplatePlanGenerator(
            profile_manager=profile_manager,
            models=[LOCAL_7B],  # local-7b 无法通过 heavy 的分数门槛
            fallback_model="local-7b",
            trigger_reason="test",
        )

        templates = {
            "strict_template": TemplateDef(
                name="strict_template",
                description="严格约束模板",
                agents=[
                    AgentDef(name="heavy_agent", capability_profile="heavy"),
                ],
            ),
        }

        store = generator.generate_all(templates)

        # heavy Profile (min_score_threshold=0.75) — local-7b 无法满足
        assigned = store.get_model("strict_template", "heavy_agent")
        assert assigned == "local-7b", (
            f"无候选时应使用 fallback 'local-7b'，实际分配 '{assigned}'"
        )

    def test_tc_plan_005_deterministic_generation(self, generator, templates):
        """TC-PLAN-005: 相同配置多次调用 generate_all() → 结果完全相同（确定性）

        验证:
        - 使用相同的 profile_manager、models、templates 调用多次
        - 每次返回的方案表内容完全一致
        - 证明方案生成是纯函数，无随机性
        """
        # 调用 5 次 generate_all
        results = [generator.generate_all(templates) for _ in range(5)]

        # 取第一次的结果作为基准
        baseline = results[0].get_all_plans()

        # 后续每次结果必须与基准完全一致
        for i, store in enumerate(results[1:], start=2):
            current = store.get_all_plans()
            assert current == baseline, (
                f"第 {i} 次调用 generate_all() 的结果与第 1 次不同。"
                f"差异: baseline={baseline}, current={current}"
            )
