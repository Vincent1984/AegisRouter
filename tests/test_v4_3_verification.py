"""V4-3 验证脚本：确认 5 个模型的 computed_score + score_range 从低到高排列正确

加载实际的 config/models.yaml 和 config/route_config.yaml，
创建 ModelScorer 并调用 build_routing_table()，验证：
1. 所有 5 个模型都有 computed_score
2. 分数严格升序排列
3. 每个模型有有效的 score_range 元组
"""

import os
from pathlib import Path

import yaml
import pytest

from aegis_router.router.model_scorer import ModelScorer, build_routing_table


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
def scorer(route_config):
    """使用实际配置创建 ModelScorer"""
    scoring = route_config["routing"]["scoring"]
    return ModelScorer(
        weights=scoring["weights"],
        normalization=scoring["normalization"],
        tolerance=route_config["routing"]["scoring"]["range_tolerance"],
    )


@pytest.fixture
def routing_table(models_config, scorer):
    """使用实际 5 个模型构建路由表（无覆盖）"""
    return build_routing_table(models_config, scorer, overrides={})


class TestV43Verification:
    """V4-3 验证：5 个模型配置 → 自动计算分数并升序排列"""

    def test_all_five_models_have_computed_score(self, routing_table):
        """所有 5 个模型都应有 computed_score"""
        assert len(routing_table) == 5
        for tier in routing_table:
            assert "computed_score" in tier
            assert isinstance(tier["computed_score"], float)
            assert 0.0 <= tier["computed_score"] <= 1.0

    def test_scores_strictly_ascending(self, routing_table):
        """分数应严格升序排列（从低到高）"""
        scores = [tier["computed_score"] for tier in routing_table]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"分数未严格升序: index {i} ({routing_table[i]['name']}={scores[i]}) "
                f">= index {i+1} ({routing_table[i+1]['name']}={scores[i+1]})"
            )

    def test_each_model_has_valid_score_range(self, routing_table):
        """每个模型应有有效的 score_range 元组"""
        for tier in routing_table:
            score_range = tier["score_range"]
            assert isinstance(score_range, tuple), f"{tier['name']}: score_range 应为 tuple"
            assert len(score_range) == 2, f"{tier['name']}: score_range 应有 2 个元素"
            lower, upper = score_range
            assert 0.0 <= lower <= upper <= 1.0, (
                f"{tier['name']}: score_range ({lower}, {upper}) 无效"
            )
            # score_range 应包含 computed_score
            assert lower <= tier["computed_score"] <= upper, (
                f"{tier['name']}: computed_score {tier['computed_score']} "
                f"不在 score_range ({lower}, {upper}) 内"
            )

    def test_expected_model_order(self, routing_table):
        """验证模型排列顺序符合预期（local-7b 最低）"""
        names = [tier["name"] for tier in routing_table]
        # local-7b 应排第一（分数最低）
        assert names[0] == "local-7b", f"第一个模型应是 local-7b，实际是 {names[0]}"

    def test_print_routing_table(self, routing_table, capsys):
        """打印路由表供人工检查"""
        print("\n" + "=" * 70)
        print("V4-3 验证结果：5 个模型路由表（按 computed_score 升序）")
        print("=" * 70)
        print(f"{'序号':<4} {'模型名':<16} {'computed_score':<16} {'score_range':<24}")
        print("-" * 70)
        for i, tier in enumerate(routing_table, 1):
            sr = tier["score_range"]
            print(
                f"{i:<4} {tier['name']:<16} {tier['computed_score']:<16.4f} "
                f"({sr[0]:.4f}, {sr[1]:.4f})"
            )
        print("=" * 70)
        # 强制 capsys 捕获输出（pytest -s 时可见）
        assert True
