"""能力 Profile 管理器

加载和管理 CapabilityProfile，根据 Profile 权重为模型打分，
基于硬约束过滤不满足条件的模型。

设计参考: design.md CapabilityProfileManager 节
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CapabilityProfile:
    """能力 Profile 定义

    Attributes:
        name: Profile 名称标识
        description: Profile 描述
        scoring_weights: 各维度评分权重
        min_score_threshold: 最低能力门槛 (低于此分数的模型被过滤)
        max_cost_per_1m_input: 成本硬约束上限 ($/1M input tokens)
        min_context_window: 最低上下文长度约束
        prefer_models: 偏好模型列表 (候选中命中的模型优先选中)
    """

    name: str
    description: str
    scoring_weights: dict[str, float]
    min_score_threshold: float = 0.0
    max_cost_per_1m_input: float = 60.0
    min_context_window: Optional[int] = None
    prefer_models: list[str] = field(default_factory=list)


# === 内置 6 种默认 Profile ===

DEFAULT_PROFILES: dict[str, CapabilityProfile] = {
    "lightweight": CapabilityProfile(
        name="lightweight",
        description="低延迟低成本，简单分类/意图识别",
        scoring_weights={
            "benchmark_mmlu": 0.10,
            "benchmark_humaneval": 0.05,
            "benchmark_math": 0.05,
            "context_window": 0.05,
            "cost_efficiency": 0.75,
        },
        min_score_threshold=0.0,
        max_cost_per_1m_input=0.5,
        prefer_models=["local-7b"],
    ),
    "medium": CapabilityProfile(
        name="medium",
        description="平衡质量和成本",
        scoring_weights={
            "benchmark_mmlu": 0.25,
            "benchmark_humaneval": 0.15,
            "benchmark_math": 0.15,
            "context_window": 0.10,
            "cost_efficiency": 0.35,
        },
        min_score_threshold=0.30,
        max_cost_per_1m_input=3.0,
    ),
    "strong_reasoning": CapabilityProfile(
        name="strong_reasoning",
        description="强推理，复杂逻辑/数学/分析",
        scoring_weights={
            "benchmark_mmlu": 0.15,
            "benchmark_humaneval": 0.30,
            "benchmark_math": 0.35,
            "context_window": 0.05,
            "cost_efficiency": 0.15,
        },
        min_score_threshold=0.60,
        max_cost_per_1m_input=20.0,
    ),
    "code_specialist": CapabilityProfile(
        name="code_specialist",
        description="代码专精",
        scoring_weights={
            "benchmark_mmlu": 0.10,
            "benchmark_humaneval": 0.50,
            "benchmark_math": 0.15,
            "context_window": 0.10,
            "cost_efficiency": 0.15,
        },
        min_score_threshold=0.50,
        max_cost_per_1m_input=10.0,
        prefer_models=["codex-mini", "gpt-5.5"],
    ),
    "long_context": CapabilityProfile(
        name="long_context",
        description="超长上下文",
        scoring_weights={
            "benchmark_mmlu": 0.15,
            "benchmark_humaneval": 0.10,
            "benchmark_math": 0.10,
            "context_window": 0.50,
            "cost_efficiency": 0.15,
        },
        min_score_threshold=0.35,
        min_context_window=500000,
        max_cost_per_1m_input=10.0,
    ),
    "heavy": CapabilityProfile(
        name="heavy",
        description="最强模型，复杂推理",
        scoring_weights={
            "benchmark_mmlu": 0.30,
            "benchmark_humaneval": 0.25,
            "benchmark_math": 0.30,
            "context_window": 0.10,
            "cost_efficiency": 0.05,
        },
        min_score_threshold=0.75,
        max_cost_per_1m_input=60.0,
    ),
}


# === 默认归一化范围 ===

DEFAULT_NORMALIZATION: dict[str, list[float]] = {
    "benchmark_mmlu": [50, 95],
    "benchmark_humaneval": [30, 95],
    "benchmark_math": [20, 95],
    "context_window": [4096, 2000000],
    "cost_per_1m_input": [0, 20],
}


class CapabilityProfileManager:
    """Profile 加载、评分、约束过滤

    负责:
    1. 从 YAML 文件加载 Profile 配置，文件不存在时使用内置默认值
    2. 根据 Profile 权重为模型打分 (归一化到 [0, 1])
    3. 根据硬约束过滤不满足条件的模型
    """

    def __init__(
        self,
        config_path: str | Path = "config/capability_profiles.yaml",
        normalization: Optional[dict[str, list[float]]] = None,
    ) -> None:
        """初始化 Profile 管理器。

        Args:
            config_path: Profile 配置文件路径
            normalization: 各维度归一化范围，None 时使用默认值
        """
        self.config_path = Path(config_path)
        self.normalization = normalization or DEFAULT_NORMALIZATION
        self.profiles: dict[str, CapabilityProfile] = self._load_profiles()

    def _load_profiles(self) -> dict[str, CapabilityProfile]:
        """加载 Profile 配置。

        优先从 YAML 文件加载，文件不存在或为空时使用内置默认值。

        Returns:
            Profile 名称到 CapabilityProfile 实例的映射
        """
        if not self.config_path.exists():
            logger.info(
                "Profile 配置文件 '%s' 不存在，使用内置默认 Profile",
                self.config_path,
            )
            return dict(DEFAULT_PROFILES)

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(
                "加载 Profile 配置文件 '%s' 失败: %s，使用内置默认 Profile",
                self.config_path,
                e,
            )
            return dict(DEFAULT_PROFILES)

        if not data or not data.get("profiles"):
            logger.info(
                "Profile 配置文件 '%s' 为空，使用内置默认 Profile",
                self.config_path,
            )
            return dict(DEFAULT_PROFILES)

        profiles: dict[str, CapabilityProfile] = {}
        for name, cfg in data["profiles"].items():
            profiles[name] = CapabilityProfile(
                name=name,
                description=cfg.get("description", ""),
                scoring_weights=cfg.get("scoring_weights", {}),
                min_score_threshold=cfg.get("min_score_threshold", 0.0),
                max_cost_per_1m_input=cfg.get("max_cost_per_1m_input", 60.0),
                min_context_window=cfg.get("min_context_window"),
                prefer_models=cfg.get("prefer_models", []),
            )

        logger.info("从 '%s' 加载了 %d 个 Profile", self.config_path, len(profiles))
        return profiles

    def get_profile(self, name: str) -> CapabilityProfile:
        """获取指定名称的 Profile。

        如果名称不存在，记录警告并降级返回 'medium' Profile。

        Args:
            name: Profile 名称

        Returns:
            对应的 CapabilityProfile 实例
        """
        if name not in self.profiles:
            logger.warning("Profile '%s' not found, fallback to 'medium'", name)
            return self.profiles["medium"]
        return self.profiles[name]

    def score_model(self, model: dict[str, Any], profile: CapabilityProfile) -> float:
        """用 Profile 权重为模型打分。

        Args:
            model: 模型字典，包含 'params' 键
            profile: 能力 Profile

        Returns:
            模型得分，范围 [0, 1]
        """
        params = model.get("params", {})
        score = 0.0

        for dim, weight in profile.scoring_weights.items():
            if dim == "cost_efficiency":
                cost = params.get("cost_per_1m_input", 0)
                norm_range = self.normalization.get("cost_per_1m_input", [0, 20])
                score += weight * (1.0 - self._normalize(cost, norm_range[0], norm_range[1]))
            elif dim == "context_window":
                val = params.get("context_window", 4096)
                norm_range = self.normalization.get("context_window", [4096, 2000000])
                score += weight * self._normalize(val, norm_range[0], norm_range[1])
            else:
                # benchmark 维度
                val = params.get(dim)
                norm_range = self.normalization.get(dim, [0, 100])
                score += weight * self._normalize(val, norm_range[0], norm_range[1])

        return max(0.0, min(1.0, score))

    def filter_by_constraints(
        self, models: list[dict[str, Any]], profile: CapabilityProfile
    ) -> list[dict[str, Any]]:
        """根据 Profile 硬约束过滤模型。

        过滤条件:
        - 模型得分 >= min_score_threshold
        - 模型成本 <= max_cost_per_1m_input
        - 模型上下文长度 >= min_context_window (如果有设置)

        Args:
            models: 模型字典列表
            profile: 能力 Profile

        Returns:
            满足所有约束的模型列表
        """
        return [m for m in models if self._meets_constraints(m, profile)]

    def _meets_constraints(self, model: dict[str, Any], profile: CapabilityProfile) -> bool:
        """检查模型是否满足 Profile 硬约束。

        Args:
            model: 模型字典
            profile: 能力 Profile

        Returns:
            是否满足所有约束
        """
        params = model.get("params", {})

        # 可用性约束：标记为 available: false 的模型直接排除
        if params.get("available") is False:
            return False

        # 成本约束
        if params.get("cost_per_1m_input", 0) > profile.max_cost_per_1m_input:
            return False

        # 上下文长度约束
        if profile.min_context_window is not None:
            if params.get("context_window", 0) < profile.min_context_window:
                return False

        # 最低分数门槛
        if self.score_model(model, profile) < profile.min_score_threshold:
            return False

        return True

    def select_best_model(
        self, models: list[dict[str, Any]], profile: CapabilityProfile
    ) -> str | None:
        """从候选模型中选出最佳模型。

        选择逻辑:
        1. 用硬约束过滤候选模型
        2. 对通过约束的模型打分并排序
        3. 如果 profile.prefer_models 非空，优先选择列表中第一个出现在候选中的模型
        4. 否则选择得分最高的模型

        Args:
            models: 模型字典列表
            profile: 能力 Profile

        Returns:
            被选中模型的名称，如果没有候选模型则返回 None
        """
        # Step 1: 硬约束过滤
        candidates = self.filter_by_constraints(models, profile)
        if not candidates:
            return None

        # Step 2: 打分并排序 (得分降序)
        scored = [
            (m, self.score_model(m, profile)) for m in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 3: 偏好选择
        if profile.prefer_models:
            for m, _score in scored:
                if m["name"] in profile.prefer_models:
                    return m["name"]

        # Step 4: 返回最高分模型
        return scored[0][0]["name"]

    @staticmethod
    def _normalize(value: float | None, min_val: float, max_val: float) -> float:
        """归一化到 [0, 1]，超出范围则 clamp。

        Args:
            value: 原始参数值，None 表示未知
            min_val: 归一化下界
            max_val: 归一化上界

        Returns:
            归一化后的值，范围 [0, 1]
        """
        if value is None:
            return 0.5
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
