"""V4 验证测试 — 使用真实配置文件验证端到端路由结果。

V4-2: 请求 {"template": "resume_screening", "agent": "resume_parser"} → 路由到 gemini-2.5-pro
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.capability_profiles import CapabilityProfileManager
from aegis_router.router.template_models import load_templates
from aegis_router.router.template_plan_generator import TemplatePlanGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_models_from_yaml() -> list[dict]:
    """从真实 config/models.yaml 加载模型列表。"""
    models_path = CONFIG_DIR / "models.yaml"
    with open(models_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["models"]


def _build_plan_store_from_real_config():
    """使用真实配置文件构建 RoutingPlanStore。

    步骤:
    1. 从 config/capability_profiles.yaml 加载 Profile
    2. 从 config/models.yaml 加载模型池
    3. 从 config/transaction_templates.yaml 加载模板
    4. 通过 TemplatePlanGenerator 预计算方案表
    """
    profiles_path = CONFIG_DIR / "capability_profiles.yaml"
    templates_path = CONFIG_DIR / "transaction_templates.yaml"

    # 加载 Profile 管理器
    profile_manager = CapabilityProfileManager(config_path=profiles_path)

    # 加载模型池
    models = _load_models_from_yaml()

    # 加载模板
    templates = load_templates(config_path=templates_path)

    # 生成方案表
    generator = TemplatePlanGenerator(
        profile_manager=profile_manager,
        models=models,
        fallback_model="deepseek-v3",
    )
    plan_store = generator.generate_all(templates)
    return plan_store


@pytest.fixture
def real_plan_store():
    """使用真实配置文件生成的方案表。"""
    return _build_plan_store_from_real_config()


@pytest.fixture
def mock_pool():
    """Mock ClawVaultPool，模拟 PII 管道正常工作。"""
    pool = MagicMock(spec=ClawVaultPool)
    pool.max_connections = 10

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            return {"masked_text": params["text"], "entities_found": []}
        elif method == "restore":
            return {"restored_text": params["text"]}
        elif method == "get_mapping":
            return {"mapping": {}}
        return None

    pool.call = AsyncMock(side_effect=mock_call)
    return pool


# ---------------------------------------------------------------------------
# V4-2: resume_screening / resume_parser → gemini-2.5-pro
# ---------------------------------------------------------------------------


class TestV4_2_ResumeParserRouting:
    """V4-2: 验证 resume_screening 模板中 resume_parser agent 路由到 gemini-2.5-pro。

    逻辑链:
    - resume_parser 使用 capability_profile: long_context
    - long_context profile: min_context_window=500000, max_cost_per_1m_input=10.0,
      context_window 权重 0.50
    - 满足约束的模型: gpt-5.5, gemini-2.5-flash, gemini-2.5-pro, gemini-3.1-pro
      (gpt-5.6-sol 被 max_cost 排除: $15 > $10)
    - gemini-2.5-pro 拥有最大 context_window (2,097,152)，
      在 context_window 权重 50% 的评分体系下得分最高
    """

    def test_plan_store_assigns_gemini_2_5_pro(self, real_plan_store):
        """方案表中 resume_screening/resume_parser 应被分配 gemini-2.5-pro。"""
        model = real_plan_store.get_model("resume_screening", "resume_parser")
        assert model == "gemini-2.5-pro", (
            f"Expected 'gemini-2.5-pro' but got '{model}'"
        )

    @pytest.mark.asyncio
    async def test_transaction_router_routes_to_gemini_2_5_pro(
        self, real_plan_store, mock_pool
    ):
        """端到端: TransactionRouterCallback 收到 resume_parser 请求 → 路由到 gemini-2.5-pro。"""
        router = TransactionRouterCallback(
            plan_store=real_plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        data = {
            "messages": [{"role": "user", "content": "Parse this resume document."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "gemini-2.5-pro", (
            f"Expected model 'gemini-2.5-pro' but got '{data.get('model')}'"
        )
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"

    def test_long_context_profile_constraints_filter_correctly(self):
        """验证 long_context profile 的硬约束正确过滤模型。"""
        profiles_path = CONFIG_DIR / "capability_profiles.yaml"
        profile_manager = CapabilityProfileManager(config_path=profiles_path)
        profile = profile_manager.get_profile("long_context")
        models = _load_models_from_yaml()

        # 验证约束参数
        assert profile.min_context_window == 500000
        assert profile.max_cost_per_1m_input == 10.0

        # 过滤后的候选模型
        candidates = profile_manager.filter_by_constraints(models, profile)
        candidate_names = [m["name"] for m in candidates]

        # gpt-5.6-sol ($15 > $10) 应被排除
        assert "gpt-5.6-sol" not in candidate_names

        # context_window < 500000 的模型应被排除
        assert "local-7b" not in candidate_names       # 32,000
        assert "deepseek-v4-pro" not in candidate_names  # 128,000
        assert "claude-sonnet" not in candidate_names  # 200,000
        assert "codex-mini" not in candidate_names     # 200,000
        assert "gpt-5.2" not in candidate_names        # 400,000
        assert "gpt-5.4-mini" not in candidate_names   # 400,000

        # gemini-2.5-pro 应在候选列表中
        assert "gemini-2.5-pro" in candidate_names

    def test_gemini_2_5_pro_scores_highest_for_long_context(self):
        """验证 gemini-2.5-pro 在 long_context profile 下得分最高。"""
        profiles_path = CONFIG_DIR / "capability_profiles.yaml"
        profile_manager = CapabilityProfileManager(config_path=profiles_path)
        profile = profile_manager.get_profile("long_context")
        models = _load_models_from_yaml()

        # 过滤并评分
        candidates = profile_manager.filter_by_constraints(models, profile)
        scored = [
            (m["name"], profile_manager.score_model(m, profile))
            for m in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # gemini-2.5-pro 应排第一
        assert scored[0][0] == "gemini-2.5-pro", (
            f"Expected 'gemini-2.5-pro' to score highest but got: {scored}"
        )
