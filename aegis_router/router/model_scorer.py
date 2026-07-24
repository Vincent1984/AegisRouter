"""模型能力评分引擎

基于加权归一化算法计算模型能力分数，并自动生成路由区间 (score_range)。
设计参考: design.md 2.3.4 节
"""

from __future__ import annotations


class ModelScorer:
    """模型能力评分引擎 — 算法可替换"""

    def __init__(self, weights: dict, normalization: dict, tolerance: float) -> None:
        """初始化评分引擎。

        Args:
            weights: 各维度权重，例如 {"benchmark_mmlu": 0.25, ...}
            normalization: 各维度归一化范围，例如 {"benchmark_mmlu": [50, 95], ...}
            tolerance: 路由区间容差值
        """
        self.weights = weights
        self.normalization = normalization
        self.tolerance = tolerance

    def normalize(self, value: float | None, min_val: float, max_val: float) -> float:
        """归一化到 [0, 1]，超出范围则 clamp。

        Args:
            value: 原始参数值，None 表示未知
            min_val: 归一化下界
            max_val: 归一化上界

        Returns:
            归一化后的值，范围 [0, 1]
        """
        if value is None:
            return 0.5  # 未知参数取中位值
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def compute_score(self, params: dict) -> float:
        """加权归一化计算模型能力分数。

        Args:
            params: 模型参数字典，包含 benchmark_mmlu, benchmark_humaneval 等

        Returns:
            模型能力分数，范围 [0, 1]
        """
        score = 0.0

        # 各 benchmark 维度
        for key in ["benchmark_mmlu", "benchmark_humaneval", "benchmark_math"]:
            if key in self.weights:
                norm_range = self.normalization[key]
                score += self.weights[key] * self.normalize(
                    params.get(key, None), norm_range[0], norm_range[1]
                )

        # 上下文长度
        if "context_window" in self.weights:
            norm_range = self.normalization["context_window"]
            score += self.weights["context_window"] * self.normalize(
                params.get("context_window", 4096), norm_range[0], norm_range[1]
            )

        # 性价比 (成本越低分越高)
        if "cost_efficiency" in self.weights:
            norm_range = self.normalization["cost_per_1m_input"]
            cost_norm = self.normalize(
                params.get("cost_per_1m_input", 0), norm_range[0], norm_range[1]
            )
            score += self.weights["cost_efficiency"] * (1.0 - cost_norm)

        return max(0.0, min(1.0, score))

    def compute_score_range(self, params: dict) -> tuple[float, float]:
        """计算能力分数并生成路由区间。

        Args:
            params: 模型参数字典

        Returns:
            路由区间元组 (lower, upper)，clamped 到 [0, 1]
        """
        score = self.compute_score(params)
        return (
            max(0.0, score - self.tolerance),
            min(1.0, score + self.tolerance),
        )


def build_routing_table(
    models_config: list, scorer: ModelScorer, overrides: dict
) -> list:
    """根据模型参数自动生成路由表。

    Args:
        models_config: 模型配置列表，每项包含 name, litellm_model, params
        scorer: ModelScorer 实例
        overrides: 人工覆盖字典，key 为模型名，value 含 score_range

    Returns:
        按 computed_score 升序排列的路由层级列表
    """
    tiers: list[dict] = []
    for model in models_config:
        name = model["name"]

        # 检查是否有人工覆盖
        if name in overrides:
            score_range = tuple(overrides[name]["score_range"])
            computed_score = scorer.compute_score(model["params"])
        else:
            computed_score = scorer.compute_score(model["params"])
            score_range = scorer.compute_score_range(model["params"])

        tiers.append(
            {
                "name": name,
                "model": model["litellm_model"],
                "computed_score": computed_score,
                "score_range": score_range,
                "cost_per_1m_input": model["params"].get("cost_per_1m_input", 0),
                "overridden": name in overrides,
            }
        )

    # 按 computed_score 排序
    tiers.sort(key=lambda t: t["computed_score"])
    return tiers
