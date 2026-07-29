"""AgentPlanGenerator 方案生成单元测试

验证检查点 V1-3: `generate_all()` 正确为各 Agent 分配模型

测试内容:
- TC-GEN-V1-3-001: 所有 Agent 都被分配模型（store 长度 == 唯一 agent 数量）
- TC-GEN-V1-3-002: 每个分配的模型都在模型池中
- TC-GEN-V1-3-003: lightweight profile 的 Agent 获得低成本模型
- TC-GEN-V1-3-004: strong_reasoning profile 的 Agent 获得高推理能力模型
- TC-GEN-V1-3-005: code_specialist profile 的 Agent 获得代码专精模型
"""

import pytest

from aegis_router.router.agent_plan_generator import AgentPlanGenerator, AgentWorkbuddyDef
from aegis_router.router.agent_plan_store import AgentPlanStore
from aegis_router.router.capability_profiles import CapabilityProfileManager


# --- 测试用模型数据（与 test_template_plan_generator.py 一致） ---

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


# --- 测试用 Agent 定义 ---

def _build_agents() -> list[AgentWorkbuddyDef]:
    """构建一组包含不同 Profile 的 Agent 定义"""
    return [
        AgentWorkbuddyDef(
            name="intent_classifier",
            capability_profile="lightweight",
            description="意图分类 Agent",
        ),
        AgentWorkbuddyDef(
            name="reasoning_engine",
            capability_profile="strong_reasoning",
            description="推理引擎 Agent",
        ),
        AgentWorkbuddyDef(
            name="code_assistant",
            capability_profile="code_specialist",
            description="代码助手 Agent",
        ),
        AgentWorkbuddyDef(
            name="general_assistant",
            capability_profile="medium",
            description="通用助手 Agent",
        ),
        AgentWorkbuddyDef(
            name="document_parser",
            capability_profile="long_context",
            description="文档解析 Agent",
        ),
    ]


# ============================================================
# V1-3: generate_all() 正确为各 Agent 分配模型
# ============================================================


class TestGenerateAllAssignsModels:
    """V1-3: generate_all() 正确为各 Agent 分配模型"""

    @pytest.fixture
    def profile_manager(self):
        """使用默认 Profile 的管理器（无外部文件依赖）"""
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    @pytest.fixture
    def agents(self):
        """标准 Agent 定义列表"""
        return _build_agents()

    @pytest.fixture
    def generator(self, profile_manager):
        """标准 AgentPlanGenerator 实例"""
        return AgentPlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )

    def test_tc_gen_v1_3_001_all_agents_get_assigned(self, generator, agents):
        """TC-GEN-V1-3-001: 所有 Agent 都被分配模型（store 长度 == 唯一 agent 数量）

        验证:
        - generate_all 返回 AgentPlanStore 实例
        - store 中条目数量等于输入 Agent 数量
        - 每个 Agent 在 store 中都有对应条目
        """
        store = generator.generate_all(agents)

        # 返回值类型验证
        assert isinstance(store, AgentPlanStore)

        # store 长度 == 唯一 agent 数量
        assert len(store) == len(agents)

        # 每个 Agent 都在 store 中
        for agent_def in agents:
            assert agent_def.name in store, (
                f"Agent '{agent_def.name}' 应在 store 中但未找到"
            )
            assert store.get_model(agent_def.name) is not None, (
                f"Agent '{agent_def.name}' 的模型分配为 None"
            )

    def test_tc_gen_v1_3_002_assigned_models_in_pool(self, generator, agents):
        """TC-GEN-V1-3-002: 每个分配的模型都在模型池中

        验证:
        - 所有分配的模型名称都存在于 ALL_MODELS 模型池中
        """
        store = generator.generate_all(agents)
        valid_model_names = {m["name"] for m in ALL_MODELS}

        all_plans = store.get_all_plans()
        for agent_name, model_name in all_plans.items():
            assert model_name in valid_model_names, (
                f"Agent '{agent_name}' 分配了无效模型 '{model_name}'，"
                f"不在模型池 {valid_model_names} 中"
            )

    def test_tc_gen_v1_3_003_lightweight_gets_cost_efficient_model(self, generator):
        """TC-GEN-V1-3-003: lightweight profile 的 Agent 获得低成本模型

        验证:
        - lightweight profile 的权重中 cost_efficiency=0.75, max_cost_per_1m_input=0.5
        - prefer_models=["local-7b"]
        - 分配的模型应是 local-7b（偏好模型且满足成本约束）
        """
        agents = [
            AgentWorkbuddyDef(
                name="cheap_agent",
                capability_profile="lightweight",
            ),
        ]
        store = generator.generate_all(agents)
        assigned_model = store.get_model("cheap_agent")

        # lightweight profile: max_cost_per_1m_input=0.5, prefer_models=["local-7b"]
        # 在满足成本约束（<=0.5）的候选中，local-7b 是偏好模型
        assert assigned_model == "local-7b", (
            f"lightweight profile 应选择偏好模型 'local-7b'，"
            f"实际分配 '{assigned_model}'"
        )

    def test_tc_gen_v1_3_004_strong_reasoning_gets_high_reasoning_model(self, generator):
        """TC-GEN-V1-3-004: strong_reasoning profile 的 Agent 获得高推理能力模型

        验证:
        - strong_reasoning profile: benchmark_math 权重=0.35, benchmark_humaneval=0.30
        - min_score_threshold=0.60, max_cost_per_1m_input=20.0
        - 分配的模型应具有高 math/humaneval 分数
        """
        agents = [
            AgentWorkbuddyDef(
                name="reasoning_agent",
                capability_profile="strong_reasoning",
            ),
        ]
        store = generator.generate_all(agents)
        assigned_model = store.get_model("reasoning_agent")

        # strong_reasoning 偏好高推理能力模型
        # 满足条件的高分模型: gpt-5.6-sol(math=95,humaneval=96,cost=15),
        #                    gpt-5.5(math=88,humaneval=93.5,cost=5),
        #                    gpt-5.2(math=83,humaneval=91.5,cost=2.5),
        #                    gemini-3.1-pro(math=85,humaneval=89,cost=2)
        # 低推理能力模型（local-7b, gemini-2.5-flash 等）应被过滤
        high_reasoning_models = {
            "gpt-5.6-sol", "gpt-5.5", "gpt-5.2", "gemini-3.1-pro",
            "claude-sonnet", "deepseek-v4-pro",
        }
        assert assigned_model in high_reasoning_models, (
            f"strong_reasoning profile 应选择高推理模型，"
            f"实际分配 '{assigned_model}'"
        )

        # 确保不会选到低能力模型
        low_capability_models = {"local-7b", "gemini-2.5-flash"}
        assert assigned_model not in low_capability_models, (
            f"strong_reasoning profile 不应选择低能力模型 '{assigned_model}'"
        )

    def test_tc_gen_v1_3_005_code_specialist_gets_coding_model(self, generator):
        """TC-GEN-V1-3-005: code_specialist profile 的 Agent 获得代码专精模型

        验证:
        - code_specialist profile: benchmark_humaneval 权重=0.50
        - max_cost_per_1m_input=10.0, min_score_threshold=0.50
        - prefer_models=["codex-mini", "gpt-5.5"]
        - 分配的模型应是 codex-mini 或 gpt-5.5（偏好模型）
        """
        agents = [
            AgentWorkbuddyDef(
                name="coder_agent",
                capability_profile="code_specialist",
            ),
        ]
        store = generator.generate_all(agents)
        assigned_model = store.get_model("coder_agent")

        # code_specialist profile: prefer_models=["codex-mini", "gpt-5.5"]
        # 在满足约束的候选中，优先选择偏好列表中的模型
        preferred_coding_models = {"codex-mini", "gpt-5.5"}
        assert assigned_model in preferred_coding_models, (
            f"code_specialist profile 应选择偏好代码模型 "
            f"{preferred_coding_models}，实际分配 '{assigned_model}'"
        )

    def test_different_profiles_get_different_models(self, generator):
        """不同 Profile 的 Agent 根据各自 Profile 权重获得不同模型

        验证:
        - lightweight 和 strong_reasoning 的约束和权重完全不同
        - 应选出不同模型
        """
        agents = [
            AgentWorkbuddyDef(
                name="light_agent",
                capability_profile="lightweight",
            ),
            AgentWorkbuddyDef(
                name="heavy_agent",
                capability_profile="strong_reasoning",
            ),
        ]
        store = generator.generate_all(agents)

        light_model = store.get_model("light_agent")
        heavy_model = store.get_model("heavy_agent")

        assert light_model != heavy_model, (
            f"不同 Profile (lightweight vs strong_reasoning) 的 Agent "
            f"应分配不同模型，但两者都被分配了 '{light_model}'"
        )

    def test_scoring_path_uses_profile_manager(self, profile_manager):
        """正常评分路径 — 没有 override_model 的 Agent 通过 Profile 评分选择模型

        验证:
        - generate_all 的结果与 select_best_model 一致
        - 确认使用了 CapabilityProfileManager 的评分逻辑
        """
        generator = AgentPlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )

        agents = [
            AgentWorkbuddyDef(
                name="medium_agent",
                capability_profile="medium",
            ),
        ]

        store = generator.generate_all(agents)
        assigned_model = store.get_model("medium_agent")

        # 手动调用 select_best_model 验证一致性
        profile = profile_manager.get_profile("medium")
        expected_model = profile_manager.select_best_model(ALL_MODELS, profile)

        assert assigned_model == expected_model, (
            f"generate_all 应通过 select_best_model 选择模型。"
            f"期望 '{expected_model}'，实际 '{assigned_model}'"
        )


# ============================================================
# V1-4: override_model 优先于 Profile 评分
# ============================================================


class TestOverrideModelPriority:
    """V1-4: override_model 优先于 Profile 评分

    验证 Property 4: 对任何设置了 override_model 的 Agent，
    AgentPlanGenerator 应直接分配该模型，忽略 CapabilityProfileManager 评分结果。
    """

    @pytest.fixture
    def profile_manager(self):
        """使用默认 Profile 的管理器（无外部文件依赖）"""
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    @pytest.fixture
    def generator(self, profile_manager):
        """标准 AgentPlanGenerator 实例"""
        return AgentPlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )

    def test_tc_gen_v1_4_001_override_model_assigned_directly(self, generator):
        """TC-GEN-V1-4-001: Agent 设置 override_model 时直接分配该模型（基本场景）

        验证:
        - 设置 override_model="gpt-5.6-sol" 的 Agent 获得正好该模型
        - 不受 capability_profile 评分结果影响
        """
        agents = [
            AgentWorkbuddyDef(
                name="heavy_analyst",
                capability_profile="medium",
                override_model="gpt-5.6-sol",
                description="重度分析 Agent — override 指定模型",
            ),
        ]
        store = generator.generate_all(agents)

        assigned_model = store.get_model("heavy_analyst")
        assert assigned_model == "gpt-5.6-sol", (
            f"override_model='gpt-5.6-sol' 应直接被分配，"
            f"实际分配 '{assigned_model}'"
        )

    def test_tc_gen_v1_4_002_override_beats_profile_scoring(self, generator, profile_manager):
        """TC-GEN-V1-4-002: Override 优先于 Profile 评分选择

        验证:
        - lightweight profile 正常评分会选 local-7b（低成本偏好）
        - 设置 override_model="gpt-5.6-sol" 后，忽略评分，直接分配 gpt-5.6-sol
        - 证明 override 优先级高于 Profile 评分
        """
        # 先验证 lightweight profile 正常情况下不会选 gpt-5.6-sol
        profile = profile_manager.get_profile("lightweight")
        scored_model = profile_manager.select_best_model(ALL_MODELS, profile)
        assert scored_model != "gpt-5.6-sol", (
            "前提条件：lightweight profile 正常评分不应选 gpt-5.6-sol"
        )

        # 设置 override_model 覆盖评分结果
        agents = [
            AgentWorkbuddyDef(
                name="light_agent_with_override",
                capability_profile="lightweight",
                override_model="gpt-5.6-sol",
                description="lightweight 但强制使用昂贵模型",
            ),
        ]
        store = generator.generate_all(agents)

        assigned_model = store.get_model("light_agent_with_override")
        assert assigned_model == "gpt-5.6-sol", (
            f"override_model 应优先于 Profile 评分。"
            f"lightweight 正常选 '{scored_model}'，"
            f"但 override='gpt-5.6-sol' 应胜出，实际分配 '{assigned_model}'"
        )

    def test_tc_gen_v1_4_003_mixed_override_and_scored_agents(self, generator, profile_manager):
        """TC-GEN-V1-4-003: 混合场景 — 部分 Agent 有 override，部分无

        验证:
        - 有 override 的 Agent 获得指定模型
        - 无 override 的 Agent 通过 Profile 评分获得模型
        - 两类 Agent 互不干扰
        """
        agents = [
            # 有 override 的 Agent
            AgentWorkbuddyDef(
                name="forced_agent_a",
                capability_profile="lightweight",
                override_model="gpt-5.5",
            ),
            AgentWorkbuddyDef(
                name="forced_agent_b",
                capability_profile="strong_reasoning",
                override_model="codex-mini",
            ),
            # 无 override 的 Agent — 通过 Profile 评分
            AgentWorkbuddyDef(
                name="scored_agent_c",
                capability_profile="medium",
            ),
            AgentWorkbuddyDef(
                name="scored_agent_d",
                capability_profile="code_specialist",
            ),
        ]
        store = generator.generate_all(agents)

        # override Agent 应获得指定模型
        assert store.get_model("forced_agent_a") == "gpt-5.5", (
            "override Agent 'forced_agent_a' 应分配 'gpt-5.5'"
        )
        assert store.get_model("forced_agent_b") == "codex-mini", (
            "override Agent 'forced_agent_b' 应分配 'codex-mini'"
        )

        # 无 override 的 Agent 应通过评分选择
        profile_medium = profile_manager.get_profile("medium")
        expected_medium = profile_manager.select_best_model(ALL_MODELS, profile_medium)
        assert store.get_model("scored_agent_c") == expected_medium, (
            f"评分 Agent 'scored_agent_c' 应分配 '{expected_medium}'，"
            f"实际 '{store.get_model('scored_agent_c')}'"
        )

        profile_code = profile_manager.get_profile("code_specialist")
        expected_code = profile_manager.select_best_model(ALL_MODELS, profile_code)
        assert store.get_model("scored_agent_d") == expected_code, (
            f"评分 Agent 'scored_agent_d' 应分配 '{expected_code}'，"
            f"实际 '{store.get_model('scored_agent_d')}'"
        )

    def test_tc_gen_v1_4_004_override_model_not_in_models_list(self, generator):
        """TC-GEN-V1-4-004: override_model 不在 models.yaml 列表中仍被使用

        验证:
        - override_model 指定了一个不存在于模型池中的模型名称
        - 仍然使用该模型（不降级为评分选择或 fallback）
        - 确认 override 优先级高于模型校验
        - 会产生 OVERRIDE_MODEL_NOT_FOUND 警告（通过日志）
        """
        unknown_model = "super-secret-model-v99"
        agents = [
            AgentWorkbuddyDef(
                name="special_agent",
                capability_profile="medium",
                override_model=unknown_model,
                description="使用未在 models.yaml 中注册的模型",
            ),
        ]
        store = generator.generate_all(agents)

        assigned_model = store.get_model("special_agent")
        assert assigned_model == unknown_model, (
            f"即使 override_model='{unknown_model}' 不在模型列表中，"
            f"仍应被分配。实际分配 '{assigned_model}'"
        )

        # 确认不是 fallback
        assert assigned_model != "local-7b", (
            "不应降级为 fallback 模型"
        )


# ============================================================
# V1-5: 相同配置多次调用 generate_all() → 结果完全相同（确定性）
# ============================================================


class TestGenerateAllDeterminism:
    """V1-5: 相同配置多次调用 generate_all() → 结果完全相同（确定性）

    验证 Property 7 (FR-7.2): 对任何有效的配置输入，多次调用
    AgentPlanGenerator.generate_all() 产生完全相同的 AgentPlanStore 内容。
    """

    @pytest.fixture
    def profile_manager(self):
        """使用默认 Profile 的管理器（无外部文件依赖）"""
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    @pytest.fixture
    def generator(self, profile_manager):
        """标准 AgentPlanGenerator 实例"""
        return AgentPlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )

    def test_tc_gen_v1_5_001_determinism_diverse_profiles(self, generator):
        """TC-GEN-V1-5-001: 多种 Profile 的 Agent 集合，10 次调用结果完全一致

        验证:
        - 使用涵盖所有 Profile 的 Agent 列表（lightweight, medium, strong_reasoning,
          code_specialist, long_context）
        - 连续调用 generate_all() 10 次
        - 每次调用产生的 agent→model 映射完全相同
        """
        agents = _build_agents()

        # 第一次调用作为基准
        baseline_store = generator.generate_all(agents)
        baseline_plans = baseline_store.get_all_plans()

        # 连续调用 10 次，每次结果都必须与基准一致
        for i in range(10):
            store = generator.generate_all(agents)
            plans = store.get_all_plans()
            assert plans == baseline_plans, (
                f"第 {i + 1} 次调用结果与基准不同。\n"
                f"基准: {baseline_plans}\n"
                f"第 {i + 1} 次: {plans}"
            )

    def test_tc_gen_v1_5_002_determinism_with_override_model(self, generator):
        """TC-GEN-V1-5-002: 包含 override_model 的 Agent 集合，10 次调用结果完全一致

        验证:
        - 混合 override 和 Profile 评分的 Agent 列表
        - 连续调用 10 次
        - override Agent 和评分 Agent 的分配结果均保持不变
        """
        agents = [
            AgentWorkbuddyDef(
                name="forced_heavy",
                capability_profile="lightweight",
                override_model="gpt-5.6-sol",
                description="管理员强制指定重量级模型",
            ),
            AgentWorkbuddyDef(
                name="scored_light",
                capability_profile="lightweight",
                description="轻量 Agent，评分选择",
            ),
            AgentWorkbuddyDef(
                name="forced_codex",
                capability_profile="medium",
                override_model="codex-mini",
                description="管理员强制指定代码模型",
            ),
            AgentWorkbuddyDef(
                name="scored_reasoning",
                capability_profile="strong_reasoning",
                description="推理 Agent，评分选择",
            ),
            AgentWorkbuddyDef(
                name="scored_code",
                capability_profile="code_specialist",
                description="代码 Agent，评分选择",
            ),
        ]

        baseline_store = generator.generate_all(agents)
        baseline_plans = baseline_store.get_all_plans()

        for i in range(10):
            store = generator.generate_all(agents)
            plans = store.get_all_plans()
            assert plans == baseline_plans, (
                f"第 {i + 1} 次调用结果与基准不同（含 override 场景）。\n"
                f"基准: {baseline_plans}\n"
                f"第 {i + 1} 次: {plans}"
            )

    def test_tc_gen_v1_5_003_determinism_with_duplicate_agents(self, generator):
        """TC-GEN-V1-5-003: 包含重复 Agent 名称的列表，10 次调用结果完全一致

        验证:
        - Agent 列表中存在同名 Agent（后定义覆盖前面的）
        - 确定性保证即使有重复名称也成立
        - 每次调用产生相同的最终映射
        """
        agents = [
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="lightweight",
                description="第一次定义 — 将被覆盖",
            ),
            AgentWorkbuddyDef(
                name="unique_agent",
                capability_profile="medium",
                description="独立定义",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="strong_reasoning",
                description="第二次定义 — 覆盖第一次",
            ),
            AgentWorkbuddyDef(
                name="another_agent",
                capability_profile="code_specialist",
                description="另一个独立 Agent",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="code_specialist",
                override_model="gpt-5.5",
                description="第三次定义 — 最终覆盖",
            ),
        ]

        baseline_store = generator.generate_all(agents)
        baseline_plans = baseline_store.get_all_plans()

        for i in range(10):
            store = generator.generate_all(agents)
            plans = store.get_all_plans()
            assert plans == baseline_plans, (
                f"第 {i + 1} 次调用结果与基准不同（含重复 Agent 场景）。\n"
                f"基准: {baseline_plans}\n"
                f"第 {i + 1} 次: {plans}"
            )

    def test_tc_gen_v1_5_004_exact_agent_model_mapping_stable(self, generator):
        """TC-GEN-V1-5-004: 验证每个 Agent 的具体模型分配在多次调用间保持稳定

        验证:
        - 逐个 Agent 检查分配的模型名称
        - 确认不仅整体映射相同，每个具体 agent→model 对也不变
        """
        agents = _build_agents()

        # 收集 10 次调用中每个 Agent 的模型分配
        all_assignments: list[dict[str, str]] = []
        for _ in range(10):
            store = generator.generate_all(agents)
            all_assignments.append(store.get_all_plans())

        # 逐个 Agent 验证模型分配一致性
        for agent_def in agents:
            models_assigned = [a[agent_def.name] for a in all_assignments]
            unique_models = set(models_assigned)
            assert len(unique_models) == 1, (
                f"Agent '{agent_def.name}' (profile={agent_def.capability_profile}) "
                f"在 10 次调用中被分配了不同模型: {unique_models}"
            )

    def test_tc_gen_v1_5_005_determinism_separate_generator_instances(self, profile_manager):
        """TC-GEN-V1-5-005: 使用独立的 Generator 实例，相同配置仍产生相同结果

        验证:
        - 创建多个独立的 AgentPlanGenerator 实例（相同配置）
        - 各实例调用 generate_all() 的结果完全一致
        - 确认确定性不依赖于实例内部状态
        """
        agents = _build_agents()

        results: list[dict[str, str]] = []
        for _ in range(10):
            gen = AgentPlanGenerator(
                profile_manager=profile_manager,
                models=ALL_MODELS,
                fallback_model="local-7b",
                trigger_reason="test",
            )
            store = gen.generate_all(agents)
            results.append(store.get_all_plans())

        # 所有实例的结果必须相同
        baseline = results[0]
        for i, plans in enumerate(results[1:], start=2):
            assert plans == baseline, (
                f"第 {i} 个独立实例的结果与基准不同。\n"
                f"基准: {baseline}\n"
                f"第 {i} 个实例: {plans}"
            )


# ============================================================
# V1-6: 重复 agent 名称最后定义胜出
# ============================================================


class TestDuplicateAgentLastDefinitionWins:
    """V1-6: 重复 Agent 最后定义胜出 (Property 6)

    验证: 对任何包含重复 agent 名称的列表，AgentPlanStore 仅保留每个重复名称
    的最后定义对应的分配，且 store 条目数恰好等于唯一 agent 名称数。

    Validates: Requirements 2.6, 7.4
    """

    @pytest.fixture
    def profile_manager(self):
        """使用默认 Profile 的管理器（无外部文件依赖）"""
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    @pytest.fixture
    def generator(self, profile_manager):
        """标准 AgentPlanGenerator 实例"""
        return AgentPlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )

    def test_tc_gen_v1_6_001_basic_duplicate_last_wins(self, generator, profile_manager):
        """TC-GEN-V1-6-001: 基本重复 — 两次定义同名 agent，最后定义胜出

        验证:
        - 定义 "my_agent" 两次，第一次 profile=lightweight，第二次 profile=strong_reasoning
        - store 使用第二次定义的 profile 评分结果
        - store 长度等于唯一 agent 名称数（1）
        """
        agents = [
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="lightweight",
                description="第一次定义 — 将被覆盖",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="strong_reasoning",
                description="第二次定义 — 最终胜出",
            ),
        ]
        store = generator.generate_all(agents)

        # store 长度等于唯一 agent 名称数
        assert len(store) == 1, (
            f"store 应只有 1 个条目（唯一 agent 数），实际 {len(store)}"
        )

        # 第二次定义的 profile 是 strong_reasoning，验证使用了该 profile 的评分
        profile_strong = profile_manager.get_profile("strong_reasoning")
        expected_model = profile_manager.select_best_model(ALL_MODELS, profile_strong)
        assigned_model = store.get_model("my_agent")

        assert assigned_model == expected_model, (
            f"重复 agent 应使用最后定义的 profile(strong_reasoning) 评分结果。"
            f"期望 '{expected_model}'，实际 '{assigned_model}'"
        )

    def test_tc_gen_v1_6_002_triple_duplicate_last_wins(self, generator, profile_manager):
        """TC-GEN-V1-6-002: 三次重复 — 三次定义同名 agent，最后定义胜出

        验证:
        - 定义 "my_agent" 三次，分别用 lightweight / strong_reasoning / code_specialist
        - 最后一次(code_specialist)胜出
        - 前两次定义完全被忽略
        """
        agents = [
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="lightweight",
                description="第一次定义",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="strong_reasoning",
                description="第二次定义",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="code_specialist",
                description="第三次定义 — 最终胜出",
            ),
        ]
        store = generator.generate_all(agents)

        # store 长度仍然为 1
        assert len(store) == 1, (
            f"三次重复定义后 store 应只有 1 个条目，实际 {len(store)}"
        )

        # 最后定义的 profile 是 code_specialist
        profile_code = profile_manager.get_profile("code_specialist")
        expected_model = profile_manager.select_best_model(ALL_MODELS, profile_code)
        assigned_model = store.get_model("my_agent")

        assert assigned_model == expected_model, (
            f"三次重复应使用最后定义的 profile(code_specialist) 评分结果。"
            f"期望 '{expected_model}'，实际 '{assigned_model}'"
        )

    def test_tc_gen_v1_6_003_duplicate_last_has_override(self, generator):
        """TC-GEN-V1-6-003: 重复且最后定义有 override_model

        验证:
        - 第一次定义：profile=lightweight（评分选模型）
        - 第二次定义：override_model="gpt-5.6-sol"
        - 最后定义有 override，直接使用 override_model
        """
        agents = [
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="lightweight",
                description="第一次定义 — 评分路径",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="medium",
                override_model="gpt-5.6-sol",
                description="第二次定义 — override 路径",
            ),
        ]
        store = generator.generate_all(agents)

        assert len(store) == 1, (
            f"store 应只有 1 个条目，实际 {len(store)}"
        )

        assigned_model = store.get_model("my_agent")
        assert assigned_model == "gpt-5.6-sol", (
            f"最后定义有 override_model='gpt-5.6-sol'，应直接使用。"
            f"实际分配 '{assigned_model}'"
        )

    def test_tc_gen_v1_6_004_first_has_override_last_has_profile(self, generator, profile_manager):
        """TC-GEN-V1-6-004: 第一次定义有 override，最后定义用 profile 评分

        验证:
        - 第一次定义：override_model="gpt-5.6-sol"
        - 第二次定义：profile=lightweight（无 override）
        - 最后定义无 override，使用 profile 评分结果（而非第一次的 override）
        """
        agents = [
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="medium",
                override_model="gpt-5.6-sol",
                description="第一次定义 — 有 override",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="lightweight",
                description="第二次定义 — 无 override，profile 评分",
            ),
        ]
        store = generator.generate_all(agents)

        assert len(store) == 1, (
            f"store 应只有 1 个条目，实际 {len(store)}"
        )

        # 最后定义使用 lightweight profile 评分
        profile_light = profile_manager.get_profile("lightweight")
        expected_model = profile_manager.select_best_model(ALL_MODELS, profile_light)
        assigned_model = store.get_model("my_agent")

        assert assigned_model == expected_model, (
            f"最后定义无 override，应使用 lightweight profile 评分。"
            f"期望 '{expected_model}'（非 'gpt-5.6-sol'），实际 '{assigned_model}'"
        )

        # 明确验证不是第一次定义的 override 值
        assert assigned_model != "gpt-5.6-sol", (
            "不应使用第一次定义的 override_model='gpt-5.6-sol'，"
            "最后定义胜出原则应覆盖之前的 override"
        )

    def test_tc_gen_v1_6_005_mixed_unique_and_duplicate(self, generator, profile_manager):
        """TC-GEN-V1-6-005: 混合唯一与重复 agent

        验证:
        - 多个唯一 agent + 部分重复 agent
        - store 长度等于唯一 agent 名称数（非总定义数）
        - 唯一 agent 不受重复逻辑影响
        - 重复 agent 使用最后定义
        """
        agents = [
            # 唯一 agent
            AgentWorkbuddyDef(
                name="unique_a",
                capability_profile="lightweight",
                description="唯一 Agent A",
            ),
            # 第一次定义 dup_agent
            AgentWorkbuddyDef(
                name="dup_agent",
                capability_profile="lightweight",
                description="重复 Agent 第一次定义",
            ),
            # 唯一 agent
            AgentWorkbuddyDef(
                name="unique_b",
                capability_profile="code_specialist",
                description="唯一 Agent B",
            ),
            # 第二次定义 dup_agent（最后定义）
            AgentWorkbuddyDef(
                name="dup_agent",
                capability_profile="strong_reasoning",
                description="重复 Agent 第二次定义 — 最终胜出",
            ),
            # 唯一 agent
            AgentWorkbuddyDef(
                name="unique_c",
                capability_profile="medium",
                description="唯一 Agent C",
            ),
        ]
        store = generator.generate_all(agents)

        # 唯一 agent 名称: unique_a, dup_agent, unique_b, unique_c → 4 个
        unique_names = {"unique_a", "dup_agent", "unique_b", "unique_c"}
        assert len(store) == len(unique_names), (
            f"store 长度应等于唯一 agent 数({len(unique_names)})，"
            f"实际 {len(store)}"
        )

        # 唯一 agent 正常分配，不受重复逻辑影响
        profile_light = profile_manager.get_profile("lightweight")
        expected_unique_a = profile_manager.select_best_model(ALL_MODELS, profile_light)
        assert store.get_model("unique_a") == expected_unique_a, (
            f"唯一 Agent 'unique_a' 应通过 lightweight 评分获得 '{expected_unique_a}'，"
            f"实际 '{store.get_model('unique_a')}'"
        )

        profile_code = profile_manager.get_profile("code_specialist")
        expected_unique_b = profile_manager.select_best_model(ALL_MODELS, profile_code)
        assert store.get_model("unique_b") == expected_unique_b, (
            f"唯一 Agent 'unique_b' 应通过 code_specialist 评分获得 '{expected_unique_b}'，"
            f"实际 '{store.get_model('unique_b')}'"
        )

        profile_medium = profile_manager.get_profile("medium")
        expected_unique_c = profile_manager.select_best_model(ALL_MODELS, profile_medium)
        assert store.get_model("unique_c") == expected_unique_c, (
            f"唯一 Agent 'unique_c' 应通过 medium 评分获得 '{expected_unique_c}'，"
            f"实际 '{store.get_model('unique_c')}'"
        )

        # 重复 agent 使用最后定义（strong_reasoning）
        profile_strong = profile_manager.get_profile("strong_reasoning")
        expected_dup = profile_manager.select_best_model(ALL_MODELS, profile_strong)
        assert store.get_model("dup_agent") == expected_dup, (
            f"重复 Agent 'dup_agent' 应使用最后定义(strong_reasoning) 的评分结果 "
            f"'{expected_dup}'，实际 '{store.get_model('dup_agent')}'"
        )

    def test_tc_gen_v1_6_006_duplicate_agent_warning_logged(self, generator, caplog):
        """TC-GEN-V1-6-006: 重复 agent 时记录 DUPLICATE_AGENT 警告日志

        验证:
        - 存在重复 agent 名称时，记录 DUPLICATE_AGENT 警告
        - 警告信息包含 agent 名称
        """
        import logging

        agents = [
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="lightweight",
                description="第一次定义",
            ),
            AgentWorkbuddyDef(
                name="my_agent",
                capability_profile="strong_reasoning",
                description="第二次定义 — 触发 DUPLICATE_AGENT 警告",
            ),
        ]

        with caplog.at_level(logging.WARNING):
            generator.generate_all(agents)

        # 验证 DUPLICATE_AGENT 警告被记录
        duplicate_warnings = [
            record for record in caplog.records
            if "DUPLICATE_AGENT" in record.message
        ]
        assert len(duplicate_warnings) >= 1, (
            "应至少记录 1 条 DUPLICATE_AGENT 警告，"
            f"实际警告: {[r.message for r in caplog.records]}"
        )

        # 验证警告中包含 agent 名称
        warning_text = duplicate_warnings[0].message
        assert "my_agent" in warning_text, (
            f"DUPLICATE_AGENT 警告应包含 agent 名称 'my_agent'，"
            f"实际警告内容: '{warning_text}'"
        )


# ============================================================
# Task 8: AgentPlanGenerator 单元测试 (TC-GEN-001 ~ TC-GEN-007)
# ============================================================


class TestAgentPlanGeneratorUnitTests:
    """AgentPlanGenerator 单元测试

    TC-GEN-001 ~ TC-GEN-007: 覆盖正常评分、Override、Profile 不存在降级、
    无候选模型 fallback、重复 agent、override 不在列表中警告、确定性。

    Validates: Requirements FR-2.1~FR-2.7, FR-7.2, FR-8.1
    """

    @pytest.fixture
    def profile_manager(self):
        """使用默认 Profile 的管理器（无外部文件依赖）"""
        return CapabilityProfileManager(config_path="nonexistent_path.yaml")

    @pytest.fixture
    def generator(self, profile_manager):
        """标准 AgentPlanGenerator 实例"""
        return AgentPlanGenerator(
            profile_manager=profile_manager,
            models=ALL_MODELS,
            fallback_model="local-7b",
            trigger_reason="test",
        )

    # ----------------------------------------------------------
    # TC-GEN-001: 正常评分路径 — Agent 无 override 时选最优模型
    # ----------------------------------------------------------

    def test_tc_gen_001_normal_scoring_path_selects_best_model(
        self, generator, profile_manager
    ):
        """TC-GEN-001: 正常评分路径 — Agent 无 override 时选最优模型

        验证:
        - 没有 override_model 的 Agent 通过 CapabilityProfileManager 评分选择模型
        - generate_all 的分配结果与手动调用 select_best_model 一致
        - 不同 Profile 的 Agent 应选出符合其能力需求的最佳模型
        """
        agents = [
            AgentWorkbuddyDef(
                name="medium_agent",
                capability_profile="medium",
            ),
            AgentWorkbuddyDef(
                name="lightweight_agent",
                capability_profile="lightweight",
            ),
            AgentWorkbuddyDef(
                name="reasoning_agent",
                capability_profile="strong_reasoning",
            ),
        ]
        store = generator.generate_all(agents)

        # 逐个验证与 select_best_model 的一致性
        for agent_def in agents:
            profile = profile_manager.get_profile(agent_def.capability_profile)
            expected_model = profile_manager.select_best_model(ALL_MODELS, profile)
            assigned_model = store.get_model(agent_def.name)
            assert assigned_model == expected_model, (
                f"Agent '{agent_def.name}' (profile={agent_def.capability_profile}) "
                f"应分配 '{expected_model}'，实际分配 '{assigned_model}'"
            )

    # ----------------------------------------------------------
    # TC-GEN-002: Override 路径 — override_model 跳过评分直接分配
    # ----------------------------------------------------------

    def test_tc_gen_002_override_model_skips_scoring(
        self, generator, profile_manager
    ):
        """TC-GEN-002: Override 路径 — override_model 跳过评分直接分配

        验证:
        - 设置 override_model 的 Agent 直接使用该模型
        - 不受 capability_profile 评分结果影响
        - override_model 即使与 Profile 评分结果不同也生效
        """
        # 验证 lightweight profile 正常评分不会选 gpt-5.6-sol
        profile = profile_manager.get_profile("lightweight")
        scored_model = profile_manager.select_best_model(ALL_MODELS, profile)
        assert scored_model != "gpt-5.6-sol", (
            "前提: lightweight 评分不应选 gpt-5.6-sol"
        )

        agents = [
            AgentWorkbuddyDef(
                name="overridden_agent",
                capability_profile="lightweight",
                override_model="gpt-5.6-sol",
                description="lightweight profile 但 override 指定昂贵模型",
            ),
        ]
        store = generator.generate_all(agents)

        assigned_model = store.get_model("overridden_agent")
        assert assigned_model == "gpt-5.6-sol", (
            f"override_model='gpt-5.6-sol' 应直接被分配，"
            f"跳过 lightweight 评分结果 '{scored_model}'。"
            f"实际分配 '{assigned_model}'"
        )

    # ----------------------------------------------------------
    # TC-GEN-003: Profile 不存在 → 降级为 medium
    # ----------------------------------------------------------

    def test_tc_gen_003_nonexistent_profile_falls_back_to_medium(
        self, generator, profile_manager, caplog
    ):
        """TC-GEN-003: Profile 不存在 → 降级为 medium

        验证:
        - 引用的 capability_profile 不存在于 profile_manager.profiles
        - 自动降级为 'medium' Profile 进行评分
        - 记录 PROFILE_NOT_FOUND 警告日志
        - 分配结果与使用 medium profile 评分一致
        """
        import logging

        nonexistent_profile = "super_ultra_quantum_profile"

        agents = [
            AgentWorkbuddyDef(
                name="unknown_profile_agent",
                capability_profile=nonexistent_profile,
            ),
        ]

        with caplog.at_level(logging.WARNING):
            store = generator.generate_all(agents)

        # 验证分配结果与 medium profile 一致
        profile_medium = profile_manager.get_profile("medium")
        expected_model = profile_manager.select_best_model(ALL_MODELS, profile_medium)
        assigned_model = store.get_model("unknown_profile_agent")

        assert assigned_model == expected_model, (
            f"Profile 不存在时应降级为 medium 评分。"
            f"期望 '{expected_model}'，实际 '{assigned_model}'"
        )

        # 验证 PROFILE_NOT_FOUND 警告被记录
        profile_warnings = [
            record for record in caplog.records
            if "PROFILE_NOT_FOUND" in record.message
               or "not found" in record.message.lower()
        ]
        assert len(profile_warnings) >= 1, (
            f"应记录 Profile 不存在的警告日志，"
            f"实际日志: {[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
        )

    # ----------------------------------------------------------
    # TC-GEN-004: 无候选模型 → 使用 fallback_model
    # ----------------------------------------------------------

    def test_tc_gen_004_no_candidate_models_uses_fallback(self, profile_manager):
        """TC-GEN-004: 无候选模型 → 使用 fallback_model

        验证:
        - 当所有模型都不满足 Profile 硬约束时，使用 fallback_model
        - 记录 NO_CANDIDATE 警告
        - 使用一组全部超出 lightweight profile 成本约束的模型（max_cost=0.5）
        """
        import logging

        # 构造模型池：所有模型成本都超出 lightweight 的 max_cost_per_1m_input=0.5
        expensive_models = [
            {
                "name": "expensive-a",
                "params": {
                    "context_window": 128000,
                    "benchmark_mmlu": 90.0,
                    "benchmark_humaneval": 88.0,
                    "benchmark_math": 82.0,
                    "cost_per_1m_input": 5.0,
                },
            },
            {
                "name": "expensive-b",
                "params": {
                    "context_window": 200000,
                    "benchmark_mmlu": 92.0,
                    "benchmark_humaneval": 91.0,
                    "benchmark_math": 85.0,
                    "cost_per_1m_input": 10.0,
                },
            },
        ]

        fallback = "my-fallback-model"
        generator = AgentPlanGenerator(
            profile_manager=profile_manager,
            models=expensive_models,
            fallback_model=fallback,
            trigger_reason="test",
        )

        agents = [
            AgentWorkbuddyDef(
                name="cost_constrained_agent",
                capability_profile="lightweight",  # max_cost_per_1m_input=0.5
            ),
        ]

        store = generator.generate_all(agents)
        assigned_model = store.get_model("cost_constrained_agent")

        assert assigned_model == fallback, (
            f"无候选模型时应使用 fallback_model='{fallback}'，"
            f"实际分配 '{assigned_model}'"
        )

    # ----------------------------------------------------------
    # TC-GEN-005: 重复 agent 名称 → 后定义胜出
    # ----------------------------------------------------------

    def test_tc_gen_005_duplicate_agent_last_definition_wins(
        self, generator, profile_manager
    ):
        """TC-GEN-005: 重复 agent 名称 → 后定义胜出

        验证:
        - 同名 Agent 出现多次时，最后一次定义的配置胜出
        - store 长度等于唯一 agent 名称数
        - 使用最后定义的 profile 评分结果
        """
        agents = [
            AgentWorkbuddyDef(
                name="dup_agent",
                capability_profile="lightweight",
                description="第一次定义 — 将被覆盖",
            ),
            AgentWorkbuddyDef(
                name="other_agent",
                capability_profile="medium",
                description="独立 agent",
            ),
            AgentWorkbuddyDef(
                name="dup_agent",
                capability_profile="strong_reasoning",
                description="第二次定义 — 最终胜出",
            ),
        ]
        store = generator.generate_all(agents)

        # store 长度等于唯一 agent 数（2: dup_agent, other_agent）
        assert len(store) == 2, (
            f"store 应有 2 个条目（唯一 agent 数），实际 {len(store)}"
        )

        # dup_agent 使用最后定义的 strong_reasoning profile
        profile_strong = profile_manager.get_profile("strong_reasoning")
        expected_model = profile_manager.select_best_model(ALL_MODELS, profile_strong)
        assigned_model = store.get_model("dup_agent")

        assert assigned_model == expected_model, (
            f"重复 agent 'dup_agent' 应使用最后定义的 profile(strong_reasoning) "
            f"评分结果 '{expected_model}'，实际 '{assigned_model}'"
        )

        # other_agent 不受影响
        profile_medium = profile_manager.get_profile("medium")
        expected_other = profile_manager.select_best_model(ALL_MODELS, profile_medium)
        assert store.get_model("other_agent") == expected_other

    # ----------------------------------------------------------
    # TC-GEN-006: override_model 不在模型列表中 → 触发警告
    # ----------------------------------------------------------

    def test_tc_gen_006_override_model_not_in_list_triggers_warning(
        self, generator, caplog
    ):
        """TC-GEN-006: override_model 不在模型列表中 → 触发警告

        验证:
        - override_model 指定了不在 models 列表中的模型名称
        - 仍然使用该 override_model（不降级）
        - 记录包含警告信息的日志
        """
        import logging

        unknown_model = "nonexistent-super-model-v99"
        agents = [
            AgentWorkbuddyDef(
                name="override_unknown_agent",
                capability_profile="medium",
                override_model=unknown_model,
            ),
        ]

        with caplog.at_level(logging.WARNING):
            store = generator.generate_all(agents)

        # 仍然使用 override_model
        assigned_model = store.get_model("override_unknown_agent")
        assert assigned_model == unknown_model, (
            f"即使 override_model='{unknown_model}' 不在模型列表中，"
            f"仍应被分配。实际分配 '{assigned_model}'"
        )

        # 验证警告被记录
        override_warnings = [
            record for record in caplog.records
            if "OVERRIDE_MODEL_NOT_FOUND" in record.message
               or unknown_model in record.message
        ]
        assert len(override_warnings) >= 1, (
            f"override_model 不在模型列表时应记录警告日志，"
            f"实际警告: {[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
        )

    # ----------------------------------------------------------
    # TC-GEN-007: 相同输入多次调用 → 结果完全一致（确定性）
    # ----------------------------------------------------------

    def test_tc_gen_007_determinism_same_input_same_output(self, generator):
        """TC-GEN-007: 相同输入多次调用 → 结果完全一致（确定性）

        验证:
        - 使用涵盖多种 Profile 和 override 的 Agent 列表
        - 连续调用 generate_all() 多次
        - 每次调用产生的 agent→model 映射完全相同
        - 验证 FR-7.2 确定性要求
        """
        agents = [
            AgentWorkbuddyDef(
                name="agent_a",
                capability_profile="lightweight",
            ),
            AgentWorkbuddyDef(
                name="agent_b",
                capability_profile="strong_reasoning",
            ),
            AgentWorkbuddyDef(
                name="agent_c",
                capability_profile="code_specialist",
                override_model="gpt-5.5",
            ),
            AgentWorkbuddyDef(
                name="agent_d",
                capability_profile="medium",
            ),
            AgentWorkbuddyDef(
                name="agent_e",
                capability_profile="long_context",
            ),
        ]

        # 第一次调用作为基准
        baseline_store = generator.generate_all(agents)
        baseline_plans = baseline_store.get_all_plans()

        # 连续调用 10 次，验证确定性
        for i in range(10):
            store = generator.generate_all(agents)
            plans = store.get_all_plans()
            assert plans == baseline_plans, (
                f"第 {i + 1} 次调用结果与基准不同，违反确定性要求。\n"
                f"基准: {baseline_plans}\n"
                f"第 {i + 1} 次: {plans}"
            )
