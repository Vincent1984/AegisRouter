"""V4-5 验证测试：两个模型区间重叠时，策略 lowest_cost 正确选择更便宜的模型

使用 design.md 2.3.7 节中定义的完整 5 模型路由表，
验证所有重叠场景下 lowest_cost 策略的正确性。

路由表（design.md 2.3.7）:
  local-7b:       score_range=[0.0, 0.20],  cost=0.0
  deepseek-v3:    score_range=[0.10, 0.50], cost=0.27
  gemini-1.5-pro: score_range=[0.35, 0.65], cost=1.25
  gpt-4o:         score_range=[0.50, 0.82], cost=2.50
  o1:             score_range=[0.75, 1.0],  cost=15.00

重叠验证表（design.md 2.3.5）:
  score=0.12 → local-7b + deepseek-v3     → lowest_cost 选 local-7b (cost=0)
  score=0.35 → deepseek-v3 + gemini-1.5-pro → lowest_cost 选 deepseek-v3 (cost=0.27)
  score=0.55 → gemini-1.5-pro + gpt-4o    → lowest_cost 选 gemini-1.5-pro (cost=1.25)
  score=0.80 → gpt-4o + o1                → lowest_cost 选 gpt-4o (cost=2.50)
"""

import pytest

from aegis_router.router.route_resolver import RouteResolver


# ---------------------------------------------------------------------------
# Fixture: 完整 5 模型路由表（精确复刻 design.md 2.3.7）
# ---------------------------------------------------------------------------

@pytest.fixture
def design_doc_tiers():
    """design.md 2.3.7 节定义的完整 5 模型路由表。"""
    return [
        {
            "name": "local-7b",
            "model": "ollama/qwen2-7b",
            "computed_score": 0.15,
            "score_range": (0.0, 0.20),
            "cost_per_1m_input": 0.0,
            "overridden": False,
        },
        {
            "name": "deepseek-v3",
            "model": "deepseek/deepseek-chat",
            "computed_score": 0.42,
            "score_range": (0.10, 0.50),
            "cost_per_1m_input": 0.27,
            "overridden": False,
        },
        {
            "name": "gemini-1.5-pro",
            "model": "gemini/gemini-1.5-pro",
            "computed_score": 0.55,
            "score_range": (0.35, 0.65),
            "cost_per_1m_input": 1.25,
            "overridden": False,
        },
        {
            "name": "gpt-4o",
            "model": "openai/gpt-4o",
            "computed_score": 0.72,
            "score_range": (0.50, 0.82),
            "cost_per_1m_input": 2.50,
            "overridden": False,
        },
        {
            "name": "o1",
            "model": "openai/o1",
            "computed_score": 0.91,
            "score_range": (0.75, 1.0),
            "cost_per_1m_input": 15.00,
            "overridden": False,
        },
    ]


# ---------------------------------------------------------------------------
# 重叠场景验证（design.md 2.3.5 路由决策表）
# ---------------------------------------------------------------------------

class TestOverlapLowestCostDesignDoc:
    """验证 design.md 中定义的所有重叠场景，lowest_cost 策略正确选择更便宜的模型。"""

    def test_score_0_12_selects_local_7b_over_deepseek(self, design_doc_tiers):
        """score=0.12 命中 local-7b(cost=0) + deepseek-v3(cost=0.27)，选 local-7b。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.12)

        assert result["model"] == "ollama/qwen2-7b"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2

    def test_score_0_35_selects_deepseek_over_gemini(self, design_doc_tiers):
        """score=0.35 命中 deepseek-v3(cost=0.27) + gemini-1.5-pro(cost=1.25)，选 deepseek-v3。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.35)

        assert result["model"] == "deepseek/deepseek-chat"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2

    def test_score_0_55_selects_gemini_over_gpt4o(self, design_doc_tiers):
        """score=0.55 命中 gemini-1.5-pro(cost=1.25) + gpt-4o(cost=2.50)，选 gemini-1.5-pro。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.55)

        assert result["model"] == "gemini/gemini-1.5-pro"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2

    def test_score_0_80_selects_gpt4o_over_o1(self, design_doc_tiers):
        """score=0.80 命中 gpt-4o(cost=2.50) + o1(cost=15.00)，选 gpt-4o。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.80)

        assert result["model"] == "openai/gpt-4o"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2


# ---------------------------------------------------------------------------
# 边界条件验证
# ---------------------------------------------------------------------------

class TestOverlapBoundaryConditions:
    """验证区间边界处的重叠行为。"""

    def test_exact_boundary_0_10_overlap_local_and_deepseek(self, design_doc_tiers):
        """score=0.10 正好在 deepseek-v3 区间下界，同时在 local-7b 区间内，
        应命中两个候选（inclusive boundaries），lowest_cost 选 local-7b。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.10)

        assert result["model"] == "ollama/qwen2-7b"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2

    def test_exact_boundary_0_20_overlap_local_and_deepseek(self, design_doc_tiers):
        """score=0.20 正好在 local-7b 区间上界，同时在 deepseek-v3 区间内，
        应命中两个候选（inclusive boundaries），lowest_cost 选 local-7b。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.20)

        assert result["model"] == "ollama/qwen2-7b"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2

    def test_exact_boundary_0_50_overlap_deepseek_gemini_gpt4o(self, design_doc_tiers):
        """score=0.50 同时在 deepseek-v3(上界) + gemini-1.5-pro(区间内) + gpt-4o(下界)，
        三个候选命中，lowest_cost 选 deepseek-v3(cost=0.27)。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.50)

        assert result["model"] == "deepseek/deepseek-chat"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 3

    def test_exact_boundary_0_75_overlap_gpt4o_and_o1(self, design_doc_tiers):
        """score=0.75 正好在 o1 区间下界，同时在 gpt-4o 区间内，
        应命中两个候选，lowest_cost 选 gpt-4o。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.75)

        assert result["model"] == "openai/gpt-4o"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2

    def test_no_overlap_score_0_95_single_match_o1(self, design_doc_tiers):
        """score=0.95 仅命中 o1（单候选），应返回 single_match。"""
        resolver = RouteResolver(design_doc_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.95)

        assert result["model"] == "openai/o1"
        assert result["reason"] == "single_match"
