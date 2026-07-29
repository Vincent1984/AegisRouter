"""Agent-WorkBuddy 方案生成器

纯函数组件：输入配置（Agent 定义列表 + 模型池 + Profile 管理器）→ 输出方案表。
在系统启动时或配置变更时调用，为每个 Agent 预计算模型分配。

与 TemplatePlanGenerator 的区别：
- 路由 key 从 (template, agent) 简化为 agent（单维度）
- 输入为扁平 Agent 列表，无模板分组
- 需要处理重复 Agent 名称（最后定义胜出）

设计参考: design.md AgentPlanGenerator 节
需求参考: FR-2.1 ~ FR-2.7, FR-8.1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from aegis_router.router.agent_plan_store import AgentPlanStore
from aegis_router.router.capability_profiles import CapabilityProfileManager
from aegis_router.observability.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class AgentWorkbuddyDef:
    """agent_workbuddy.yaml 中的 Agent 定义

    Attributes:
        name: Agent 唯一标识（用作路由查表 key）
        capability_profile: Profile 名称（用于选模型打分）
        override_model: 管理员直接指定模型（最高优先级，跳过 Profile 评分）
        description: Agent 描述（文档用，不参与路由逻辑）
    """

    name: str
    capability_profile: str
    override_model: Optional[str] = None
    description: Optional[str] = None


def load_agent_workbuddy_config(
    config_path: str | Path = "config/agent_workbuddy.yaml",
) -> list[AgentWorkbuddyDef]:
    """从 YAML 文件加载 Agent-WorkBuddy 配置。

    加载逻辑:
    - 文件不存在: 返回空列表 + 日志警告（系统正常启动，所有请求走 fallback）
    - 文件为空或无 agents 字段: 返回空列表
    - YAML 语法错误: 抛出 ValueError（启动失败，明确错误信息）
    - Agent 条目缺少必填字段 (name): 跳过该条目并记录警告

    与 load_templates() 的区别：
    - YAML 语法错误时抛出异常（而非返回空 dict），符合 FR-4.3 要求
    - 返回扁平的 AgentWorkbuddyDef 列表（无模板分组）
    - 缺少 capability_profile 时默认为 'medium'（而非跳过）

    Args:
        config_path: agent_workbuddy.yaml 文件路径

    Returns:
        AgentWorkbuddyDef 实例列表

    Raises:
        ValueError: 当 YAML 语法错误时抛出，包含明确错误信息
    """
    path = Path(config_path)

    if not path.exists():
        logger.warning(
            "Agent-WorkBuddy 配置文件 '%s' 不存在，方案表为空，所有请求使用 fallback",
            path,
        )
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Agent-WorkBuddy 配置文件 '{path}' YAML 语法错误: {e}"
        ) from e

    if not data:
        logger.info("Agent-WorkBuddy 配置文件 '%s' 为空，方案表为空", path)
        return []

    if not isinstance(data, dict):
        logger.warning(
            "Agent-WorkBuddy 配置文件 '%s' 顶层结构不是字典，方案表为空",
            path,
        )
        return []

    agents_data = data.get("agents")
    if not agents_data or not isinstance(agents_data, list):
        logger.info(
            "Agent-WorkBuddy 配置文件 '%s' 中无有效 'agents' 列表",
            path,
        )
        return []

    agents: list[AgentWorkbuddyDef] = []

    for i, entry in enumerate(agents_data):
        if not isinstance(entry, dict):
            logger.warning(
                "Agent-WorkBuddy 配置中第 %d 个条目无效（非字典），跳过",
                i + 1,
            )
            continue

        name = entry.get("name")
        if not name or not isinstance(name, str):
            logger.warning(
                "Agent-WorkBuddy 配置中第 %d 个条目缺少 'name' 字段，跳过",
                i + 1,
            )
            continue

        capability_profile = entry.get("capability_profile", "medium")
        if not isinstance(capability_profile, str):
            capability_profile = "medium"

        override_model = entry.get("override_model")
        description = entry.get("description")

        agents.append(
            AgentWorkbuddyDef(
                name=name,
                capability_profile=capability_profile,
                override_model=override_model,
                description=description,
            )
        )

    logger.info(
        "从 '%s' 加载了 %d 个 Agent 定义",
        path,
        len(agents),
    )

    return agents


class AgentPlanGenerator:
    """纯函数：输入 Agent 列表配置 → 输出 AgentPlanStore。

    确定性：相同配置永远生成相同方案。
    无状态：不持有外部依赖，所有依赖通过构造函数注入。

    复用 CapabilityProfileManager 的评分和约束过滤逻辑。

    Attributes:
        profile_manager: 能力 Profile 管理器实例
        models: 模型池（字典列表，来自 models.yaml）
        fallback_model: 无候选时的降级模型名称
        trigger_reason: 方案生成触发原因
    """

    def __init__(
        self,
        profile_manager: CapabilityProfileManager,
        models: list[dict[str, Any]],
        fallback_model: str,
        trigger_reason: str = "startup",
    ) -> None:
        """初始化方案生成器。

        Args:
            profile_manager: CapabilityProfileManager 实例，负责 Profile 加载和评分
            models: 模型字典列表，每项包含 name, params 等字段
            fallback_model: 无候选模型时使用的降级模型名称
            trigger_reason: 方案生成触发原因（如 "startup", "agent_workbuddy.yaml" 等）
        """
        self.profile_manager = profile_manager
        self.models = models
        self.fallback_model = fallback_model
        self.trigger_reason = trigger_reason
        self._audit = AuditLogger()
        self._model_names: set[str] = {m["name"] for m in models}

    def generate_all(self, agents: list[AgentWorkbuddyDef]) -> AgentPlanStore:
        """为所有 Agent 生成模型分配方案。

        对每个 Agent:
        1. 有 override_model 且在 models 列表中 → 直接使用
        2. 有 override_model 但不在 models 列表中 → 警告，仍然使用
        3. 否则: 加载 Profile → 对所有模型打分 → 过滤约束 → 选最优
        4. 若无候选模型 → 使用 fallback_model 并记录 NO_CANDIDATE 警告

        处理重复 Agent 名称：后定义的覆盖前面的，记录 DUPLICATE_AGENT 警告。

        Args:
            agents: AgentWorkbuddyDef 实例列表

        Returns:
            填充完成的 AgentPlanStore 实例
        """
        store = AgentPlanStore()
        seen_names: set[str] = set()
        warnings: list[str] = []

        for agent_def in agents:
            # 检测重复 Agent 名称
            if agent_def.name in seen_names:
                warn_msg = (
                    f"DUPLICATE_AGENT: agent='{agent_def.name}' — "
                    f"重复定义，后定义覆盖前面的"
                )
                logger.warning(warn_msg)
                warnings.append(warn_msg)
            seen_names.add(agent_def.name)

            model, reason = self._select_model(agent_def, warnings)
            store.set_model(agent_def.name, model)

        # Audit: log plan generation event (FR-8.1)
        self._audit.log_plan_generation_event(
            trigger_reason=self.trigger_reason,
            template_name="agent_workbuddy",
            assignments=store.get_all_plans(),
            total_agents=len(seen_names),
        )

        # 生成完成后输出完整方案日志
        self._log_generated_plans(store, agents)

        return store

    def _select_model(
        self, agent_def: AgentWorkbuddyDef, warnings: list[str]
    ) -> tuple[str, str]:
        """为单个 Agent 选择模型。

        选择优先级:
        1. override_model（管理员直接指定，最高优先级）
        2. Profile 自动选择（打分 + 约束过滤 + 偏好加权）
        3. fallback_model（无候选时降级）

        Args:
            agent_def: Agent 定义
            warnings: 警告列表（可追加新警告）

        Returns:
            (模型名称, 选择原因) 元组
        """
        # 优先级 1: override_model
        if agent_def.override_model:
            # 校验 override_model 是否在 models 列表中
            if agent_def.override_model not in self._model_names:
                warn_msg = (
                    f"OVERRIDE_MODEL_NOT_FOUND: agent='{agent_def.name}', "
                    f"override_model='{agent_def.override_model}' "
                    f"不在 models 列表中，仍然使用"
                )
                logger.warning(warn_msg)
                warnings.append(warn_msg)
            return agent_def.override_model, "override"

        # 优先级 2: Profile 驱动选模型
        profile = self.profile_manager.get_profile(agent_def.capability_profile)

        # 检测 Profile 是否降级（get_profile 内部已 warning，此处追踪）
        if agent_def.capability_profile not in self.profile_manager.profiles:
            warn_msg = (
                f"PROFILE_NOT_FOUND: agent='{agent_def.name}', "
                f"profile='{agent_def.capability_profile}' 不存在，"
                f"降级为 'medium' Profile"
            )
            logger.warning(warn_msg)
            warnings.append(warn_msg)

        best_model = self.profile_manager.select_best_model(self.models, profile)

        if best_model is None:
            # 优先级 3: 无候选，使用 fallback
            warn_msg = (
                f"NO_CANDIDATE: agent='{agent_def.name}', "
                f"profile='{agent_def.capability_profile}' — "
                f"无模型满足硬约束，使用 fallback 模型 '{self.fallback_model}'"
            )
            logger.warning(warn_msg)
            warnings.append(warn_msg)
            return self.fallback_model, "fallback"

        return best_model, "scored"

    def _log_generated_plans(
        self, store: AgentPlanStore, agents: list[AgentWorkbuddyDef]
    ) -> None:
        """输出完整方案生成日志（汇总表）。

        日志格式参考 design.md 方案生成示例节，包含:
        - 每个 Agent 的模型分配结果
        - Profile 名称和得分
        - override 标记

        Args:
            store: 已填充的方案表
            agents: Agent 定义列表
        """
        # 为确定性输出，按最终 store 中的 agent 顺序（即去重后最后出现的顺序）
        # 收集唯一 agent（保留最后出现的定义）
        unique_agents: dict[str, AgentWorkbuddyDef] = {}
        for agent_def in agents:
            unique_agents[agent_def.name] = agent_def

        lines: list[str] = ["Agent-WorkBuddy Router: 方案生成完成"]
        lines.append("")
        lines.append(
            f"  {'Agent':<20} → {'Model':<20} (Profile)"
        )
        lines.append(f"  {'─' * 55}")

        for agent_name, agent_def in unique_agents.items():
            model = store.get_model(agent_name)
            if agent_def.override_model:
                lines.append(
                    f"  {agent_name:<20} → {model:<20} "
                    f"(override, 管理员指定)"
                )
            else:
                profile = self.profile_manager.get_profile(
                    agent_def.capability_profile
                )
                score = self._get_model_score(model, profile)
                preferred = ""
                if profile.prefer_models and model in profile.prefer_models:
                    preferred = ", preferred"
                lines.append(
                    f"  {agent_name:<20} → {model:<20} "
                    f"(profile={agent_def.capability_profile}, "
                    f"score={score:.2f}{preferred})"
                )

        lines.append("")
        lines.append(
            f"  Total: {len(store)} agents, Fallback: {self.fallback_model}"
        )

        logger.info("\n".join(lines))

    def _get_model_score(self, model_name: str | None, profile: Any) -> float:
        """获取指定模型在指定 Profile 下的得分。

        Args:
            model_name: 模型名称
            profile: CapabilityProfile 实例

        Returns:
            模型得分，未找到模型时返回 0.0
        """
        if model_name is None:
            return 0.0
        for m in self.models:
            if m["name"] == model_name:
                return self.profile_manager.score_model(m, profile)
        return 0.0
