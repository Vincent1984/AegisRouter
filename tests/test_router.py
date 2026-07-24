"""智能路由测试 — RouteResolver 区间匹配与重叠策略"""

import pytest

from aegis_router.router.route_resolver import RouteResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tiers():
    """模拟 build_routing_table() 输出的路由表（按 computed_score 升序）。"""
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
            "name": "deepseek-chat",
            "model": "deepseek/deepseek-chat",
            "computed_score": 0.45,
            "score_range": (0.20, 0.60),
            "cost_per_1m_input": 0.27,
            "overridden": False,
        },
        {
            "name": "gpt-4o",
            "model": "openai/gpt-4o",
            "computed_score": 0.75,
            "score_range": (0.55, 0.85),
            "cost_per_1m_input": 5.0,
            "overridden": False,
        },
        {
            "name": "o1",
            "model": "openai/o1",
            "computed_score": 0.92,
            "score_range": (0.80, 1.0),
            "cost_per_1m_input": 15.0,
            "overridden": False,
        },
    ]


# ---------------------------------------------------------------------------
# Single match
# ---------------------------------------------------------------------------

class TestSingleMatch:
    """score 仅命中一个 tier 的场景。"""

    def test_score_hits_only_local_model(self, sample_tiers):
        resolver = RouteResolver(sample_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.10)
        assert result["model"] == "ollama/qwen2-7b"
        assert result["reason"] == "single_match"
        assert result["candidates"] == ["ollama/qwen2-7b"]

    def test_score_hits_only_mid_model(self, sample_tiers):
        resolver = RouteResolver(sample_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.30)
        assert result["model"] == "deepseek/deepseek-chat"
        assert result["reason"] == "single_match"
        assert result["candidates"] == ["deepseek/deepseek-chat"]

    def test_score_hits_only_top_model(self, sample_tiers):
        resolver = RouteResolver(sample_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.95)
        assert result["model"] == "openai/o1"
        assert result["reason"] == "single_match"
        assert result["candidates"] == ["openai/o1"]


# ---------------------------------------------------------------------------
# No match — fallback
# ---------------------------------------------------------------------------

class TestNoMatch:
    """score 不命中任何 tier，使用兜底模型。"""

    def test_empty_tiers_triggers_fallback(self):
        resolver = RouteResolver([], strategy="lowest_cost", fallback_model="deepseek-v3")
        result = resolver.resolve(0.5)
        assert result["model"] == "deepseek-v3"
        assert result["reason"] == "no_match_fallback"
        assert result["candidates"] == []

    def test_score_outside_all_ranges(self, sample_tiers):
        """构造一个 gap：移除中间 tier 使 score=0.25 无命中。"""
        # 修改 tiers 使 0.21~0.29 无覆盖
        tiers = [
            {
                "name": "local-7b",
                "model": "ollama/qwen2-7b",
                "computed_score": 0.10,
                "score_range": (0.0, 0.15),
                "cost_per_1m_input": 0.0,
                "overridden": False,
            },
            {
                "name": "gpt-4o",
                "model": "openai/gpt-4o",
                "computed_score": 0.75,
                "score_range": (0.60, 0.90),
                "cost_per_1m_input": 5.0,
                "overridden": False,
            },
        ]
        resolver = RouteResolver(tiers, strategy="lowest_cost", fallback_model="fallback-model")
        result = resolver.resolve(0.25)
        assert result["model"] == "fallback-model"
        assert result["reason"] == "no_match_fallback"
        assert result["candidates"] == []

    def test_custom_fallback_model(self, sample_tiers):
        resolver = RouteResolver([], strategy="lowest_cost", fallback_model="my-fallback")
        result = resolver.resolve(0.5)
        assert result["model"] == "my-fallback"


# ---------------------------------------------------------------------------
# Multiple candidates — overlap strategies
# ---------------------------------------------------------------------------

class TestOverlapLowestCost:
    """lowest_cost 策略：选最便宜的候选。"""

    def test_overlap_selects_cheapest(self, sample_tiers):
        """score=0.20 命中 local-7b 和 deepseek-chat，选 cost=0 的。"""
        resolver = RouteResolver(sample_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.20)
        assert result["model"] == "ollama/qwen2-7b"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2

    def test_overlap_higher_range(self, sample_tiers):
        """score=0.55 命中 deepseek-chat 和 gpt-4o，选 deepseek（更便宜）。"""
        resolver = RouteResolver(sample_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.55)
        assert result["model"] == "deepseek/deepseek-chat"
        assert result["reason"] == "overlap_lowest_cost"
        assert result["candidates_count"] == 2


class TestOverlapHighestCapability:
    """highest_capability 策略：选能力分最高的候选。"""

    def test_overlap_selects_highest_score(self, sample_tiers):
        """score=0.20 命中 local-7b(0.15) 和 deepseek-chat(0.45)，选 deepseek。"""
        resolver = RouteResolver(sample_tiers, strategy="highest_capability")
        result = resolver.resolve(0.20)
        assert result["model"] == "deepseek/deepseek-chat"
        assert result["reason"] == "overlap_highest_capability"
        assert result["candidates_count"] == 2

    def test_three_way_overlap(self, sample_tiers):
        """score=0.80 命中 gpt-4o(0.75) 和 o1(0.92)，选 o1。"""
        resolver = RouteResolver(sample_tiers, strategy="highest_capability")
        result = resolver.resolve(0.80)
        assert result["model"] == "openai/o1"
        assert result["reason"] == "overlap_highest_capability"
        assert result["candidates_count"] == 2


class TestOverlapRoundRobin:
    """round_robin 策略：轮询分发。"""

    def test_rotates_through_candidates(self, sample_tiers):
        """score=0.20 命中 2 个候选，连续调用应轮询。"""
        resolver = RouteResolver(sample_tiers, strategy="round_robin")

        results = [resolver.resolve(0.20)["model"] for _ in range(4)]

        # 候选为 [local-7b, deepseek-chat]，应该交替出现
        assert results[0] == "ollama/qwen2-7b"
        assert results[1] == "deepseek/deepseek-chat"
        assert results[2] == "ollama/qwen2-7b"
        assert results[3] == "deepseek/deepseek-chat"

    def test_counter_wraps_around(self, sample_tiers):
        """验证 counter 正确取模。"""
        resolver = RouteResolver(sample_tiers, strategy="round_robin")
        # 调用多次确保不越界
        for _ in range(10):
            result = resolver.resolve(0.20)
            assert result["model"] in ("ollama/qwen2-7b", "deepseek/deepseek-chat")


class TestOverlapRandom:
    """random 策略：随机选择。"""

    def test_random_returns_valid_candidate(self, sample_tiers):
        """score=0.20 命中 2 个候选，随机选择应返回其中之一。"""
        resolver = RouteResolver(sample_tiers, strategy="random")
        valid_models = {"ollama/qwen2-7b", "deepseek/deepseek-chat"}

        for _ in range(20):
            result = resolver.resolve(0.20)
            assert result["model"] in valid_models
            assert result["reason"] == "overlap_random"
            assert result["candidates_count"] == 2

    def test_random_produces_variety(self, sample_tiers):
        """多次调用 random 应产生不同结果（概率测试）。"""
        resolver = RouteResolver(sample_tiers, strategy="random")
        models_seen = set()

        for _ in range(50):
            result = resolver.resolve(0.20)
            models_seen.add(result["model"])

        # 50 次调用中，两个候选都应该至少出现一次
        assert len(models_seen) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """边界条件测试。"""

    def test_score_exactly_on_lower_boundary(self, sample_tiers):
        """score=0.0 正好在 local-7b 的下界，应命中。"""
        resolver = RouteResolver(sample_tiers, strategy="lowest_cost")
        result = resolver.resolve(0.0)
        assert result["model"] == "ollama/qwen2-7b"

    def test_score_exactly_on_upper_boundary(self, sample_tiers):
        """score=1.0 正好在 o1 的上界，应命中。"""
        resolver = RouteResolver(sample_tiers, strategy="lowest_cost")
        result = resolver.resolve(1.0)
        assert result["model"] == "openai/o1"

    def test_score_on_shared_boundary(self, sample_tiers):
        """score=0.20 是 local-7b 的上界也是 deepseek-chat 的下界，
        应同时命中两个候选（inclusive boundaries）。"""
        resolver = RouteResolver(sample_tiers, strategy="highest_capability")
        result = resolver.resolve(0.20)
        assert result["candidates_count"] == 2

    def test_invalid_strategy_raises_error(self, sample_tiers):
        """无效策略应抛出 ValueError。"""
        with pytest.raises(ValueError, match="Invalid strategy"):
            RouteResolver(sample_tiers, strategy="invalid_strategy")

    def test_empty_tiers_with_any_score(self):
        """空路由表对任何分数都返回 fallback。"""
        resolver = RouteResolver([], strategy="random", fallback_model="fb")
        assert resolver.resolve(0.0)["reason"] == "no_match_fallback"
        assert resolver.resolve(0.5)["reason"] == "no_match_fallback"
        assert resolver.resolve(1.0)["reason"] == "no_match_fallback"

    def test_single_tier_always_single_match(self):
        """只有一个 tier 且 score 命中时，返回 single_match。"""
        tiers = [
            {
                "name": "only-model",
                "model": "test/model",
                "computed_score": 0.5,
                "score_range": (0.0, 1.0),
                "cost_per_1m_input": 1.0,
                "overridden": False,
            }
        ]
        resolver = RouteResolver(tiers, strategy="lowest_cost")
        result = resolver.resolve(0.5)
        assert result["model"] == "test/model"
        assert result["reason"] == "single_match"
        assert result["candidates"] == ["test/model"]
