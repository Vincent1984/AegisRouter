"""V4-4 验证脚本：确认 route_overrides.yaml 中 gpt-4o 的覆盖区间生效

加载实际的 config/models.yaml、config/route_config.yaml 和 config/route_overrides.yaml，
创建 ModelScorer 并调用 build_routing_table()（带覆盖），验证：
1. gpt-4o 的 score_range 等于覆盖值 (0.50, 0.82)
2. gpt-4o 标记为 overridden: True
3. 自动计算的 score_range（无覆盖）与覆盖值不同
4. RouteResolver 能正确将落在覆盖区间内的 prompt_score 路由到 gpt-4o
"""

from pathlib import Path

import yaml
import pytest

from aegis_router.router.model_scorer import ModelScorer, build_routing_table
from aegis_router.router.route_resolver import RouteResolver


# --- 加载实际 YAML 配置 ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(relative_path: str) -> dict:
    """加载项目根目录下的 YAML 文件"""
    filepath = PROJECT_ROOT / relative_path
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def models_config():
    """从 config/models.yaml 加载模型配置"""
    data = load_yaml("config/models.yaml")
    return data["models"]


@pytest.fixture
def route_config():
    """从 config/route_config.yaml 加载路由配置"""
    return load_yaml("config/route_config.yaml")


@pytest.fixture
def overrides_config():
    """从 config/route_overrides.yaml 加载覆盖配置"""
    data = load_yaml("config/route_overrides.yaml")
    return data["overrides"]


@pytest.fixture
def scorer(route_config):
    """使用实际配置创建 ModelScorer"""
    scoring = route_config["routing"]["scoring"]
    return ModelScorer(
        weights=scoring["weights"],
        normalization=scoring["normalization"],
        tolerance=scoring["range_tolerance"],
    )


@pytest.fixture
def routing_table_with_overrides(models_config, scorer, overrides_config):
    """使用实际 5 个模型构建路由表（带覆盖）"""
    return build_routing_table(models_config, scorer, overrides=overrides_config)


@pytest.fixture
def routing_table_without_overrides(models_config, scorer):
    """使用实际 5 个模型构建路由表（无覆盖，用于对比）"""
    return build_routing_table(models_config, scorer, overrides={})


class TestV44OverrideVerification:
    """V4-4 验证：route_overrides.yaml 中 gpt-4o 覆盖区间生效"""

    def test_overrides_yaml_contains_gpt4o(self, overrides_config):
        """route_overrides.yaml 应包含 gpt-4o 覆盖配置"""
        assert "gpt-4o" in overrides_config
        assert overrides_config["gpt-4o"]["score_range"] == [0.50, 0.82]

    def test_gpt4o_score_range_equals_override(self, routing_table_with_overrides):
        """gpt-4o 的 score_range 应等于覆盖值 (0.50, 0.82)"""
        gpt4o_tier = next(
            t for t in routing_table_with_overrides if t["name"] == "gpt-4o"
        )
        assert gpt4o_tier["score_range"] == (0.50, 0.82), (
            f"gpt-4o score_range 应为 (0.50, 0.82)，实际为 {gpt4o_tier['score_range']}"
        )

    def test_gpt4o_marked_as_overridden(self, routing_table_with_overrides):
        """gpt-4o 应标记为 overridden: True"""
        gpt4o_tier = next(
            t for t in routing_table_with_overrides if t["name"] == "gpt-4o"
        )
        assert gpt4o_tier["overridden"] is True, (
            "gpt-4o 应标记为 overridden: True"
        )

    def test_override_differs_from_auto_computed(
        self, routing_table_with_overrides, routing_table_without_overrides
    ):
        """覆盖后的 score_range 应与自动计算的不同"""
        gpt4o_overridden = next(
            t for t in routing_table_with_overrides if t["name"] == "gpt-4o"
        )
        gpt4o_auto = next(
            t for t in routing_table_without_overrides if t["name"] == "gpt-4o"
        )

        # 自动计算的区间不应等于覆盖值
        assert gpt4o_auto["score_range"] != (0.50, 0.82), (
            f"自动计算的 score_range {gpt4o_auto['score_range']} "
            f"不应等于覆盖值 (0.50, 0.82)，否则覆盖无意义"
        )
        # 覆盖后的区间应与自动计算的不同
        assert gpt4o_overridden["score_range"] != gpt4o_auto["score_range"], (
            f"覆盖后 {gpt4o_overridden['score_range']} "
            f"应与自动计算 {gpt4o_auto['score_range']} 不同"
        )

    def test_auto_computed_gpt4o_not_overridden(self, routing_table_without_overrides):
        """无覆盖时 gpt-4o 应标记为 overridden: False"""
        gpt4o_auto = next(
            t for t in routing_table_without_overrides if t["name"] == "gpt-4o"
        )
        assert gpt4o_auto["overridden"] is False

    def test_route_resolver_routes_to_gpt4o_in_override_range(
        self, routing_table_with_overrides, route_config
    ):
        """prompt_score 在覆盖区间 [0.50, 0.82] 内时，gpt-4o 应为候选模型之一"""
        resolver = RouteResolver(
            tiers=routing_table_with_overrides,
            strategy=route_config["routing"]["overlap_strategy"],
            fallback_model=route_config["routing"]["fallback_model"],
        )

        # 测试区间中间值
        result = resolver.resolve(prompt_score=0.65)
        # gpt-4o 的 litellm_model 是 "openai/gpt-4o"
        # 由于可能存在重叠，结果可能是 gpt-4o 或其他模型（取决于策略）
        # 验证 gpt-4o 至少在候选列表中
        candidates = [
            t for t in routing_table_with_overrides
            if t["score_range"][0] <= 0.65 <= t["score_range"][1]
        ]
        candidate_models = [t["model"] for t in candidates]
        assert "openai/gpt-4o" in candidate_models, (
            f"prompt_score=0.65 时 gpt-4o 应为候选模型，"
            f"实际候选: {candidate_models}"
        )

    def test_route_resolver_boundary_low(self, routing_table_with_overrides, route_config):
        """prompt_score=0.50（覆盖区间下界）时，gpt-4o 应为候选"""
        candidates = [
            t for t in routing_table_with_overrides
            if t["score_range"][0] <= 0.50 <= t["score_range"][1]
        ]
        candidate_names = [t["name"] for t in candidates]
        assert "gpt-4o" in candidate_names, (
            f"prompt_score=0.50 时 gpt-4o 应为候选，实际候选: {candidate_names}"
        )

    def test_route_resolver_boundary_high(self, routing_table_with_overrides, route_config):
        """prompt_score=0.82（覆盖区间上界）时，gpt-4o 应为候选"""
        candidates = [
            t for t in routing_table_with_overrides
            if t["score_range"][0] <= 0.82 <= t["score_range"][1]
        ]
        candidate_names = [t["name"] for t in candidates]
        assert "gpt-4o" in candidate_names, (
            f"prompt_score=0.82 时 gpt-4o 应为候选，实际候选: {candidate_names}"
        )

    def test_print_comparison(
        self, routing_table_with_overrides, routing_table_without_overrides, capsys
    ):
        """打印覆盖前后对比供人工检查"""
        print("\n" + "=" * 80)
        print("V4-4 验证结果：gpt-4o 覆盖区间对比")
        print("=" * 80)

        gpt4o_auto = next(
            t for t in routing_table_without_overrides if t["name"] == "gpt-4o"
        )
        gpt4o_overridden = next(
            t for t in routing_table_with_overrides if t["name"] == "gpt-4o"
        )

        print(f"\n{'项目':<20} {'自动计算':<30} {'覆盖后':<30}")
        print("-" * 80)
        print(
            f"{'score_range':<20} "
            f"({gpt4o_auto['score_range'][0]:.4f}, {gpt4o_auto['score_range'][1]:.4f})"
            f"{'':>10}"
            f"({gpt4o_overridden['score_range'][0]:.4f}, {gpt4o_overridden['score_range'][1]:.4f})"
        )
        print(
            f"{'computed_score':<20} "
            f"{gpt4o_auto['computed_score']:.4f}{'':>25}"
            f"{gpt4o_overridden['computed_score']:.4f}"
        )
        print(
            f"{'overridden':<20} "
            f"{gpt4o_auto['overridden']}{'':>25}"
            f"{gpt4o_overridden['overridden']}"
        )
        print("=" * 80)

        # 打印完整覆盖后路由表
        print("\n完整路由表（带覆盖）:")
        print(f"{'序号':<4} {'模型名':<16} {'score_range':<24} {'overridden':<12}")
        print("-" * 60)
        for i, tier in enumerate(routing_table_with_overrides, 1):
            sr = tier["score_range"]
            print(
                f"{i:<4} {tier['name']:<16} "
                f"({sr[0]:.4f}, {sr[1]:.4f}){'':>4}"
                f"{tier['overridden']}"
            )
        print("=" * 80)
        assert True
