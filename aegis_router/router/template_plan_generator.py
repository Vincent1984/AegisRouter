"""模板方案生成器

纯函数组件：输入配置（模板定义 + 模型池 + Profile 管理器）→ 输出方案表。
在系统启动时或配置变更时调用，为每个模板中的每个 Agent 预计算模型分配。

设计参考: design.md TemplatePlanGenerator 节
需求参考: FR-4.1 ~ FR-4.7, FR-8.1
"""

from __future__ import annotations

import logging
from typing import Any

from aegis_router.router.capability_profiles import CapabilityProfileManager
from aegis_router.router.routing_plan_store import RoutingPlanStore
from aegis_router.router.template_models import AgentDef, TemplateDef
from aegis_router.observability.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class TemplatePlanGenerator:
    """纯函数：输入配置 → 输出方案表。

    确定性：相同配置永远生成相同方案。
    无状态：不持有外部依赖，所有依赖通过构造函数注入。

    Attributes:
        profile_manager: 能力 Profile 管理器实例
        models: 模型池（字典列表，来自 models.yaml）
        fallback_model: 无候选时的降级模型名称
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
            trigger_reason: 方案生成触发原因（如 "startup", "models.yaml" 等）
        """
        self.profile_manager = profile_manager
        self.models = models
        self.fallback_model = fallback_model
        self.trigger_reason = trigger_reason
        self._audit = AuditLogger()

    def generate_all(self, templates: dict[str, TemplateDef]) -> RoutingPlanStore:
        """为所有模板中的所有 Agent 生成模型分配方案。

        对每个模板的每个 Agent:
        1. 有 override_model → 直接使用
        2. 否则: 加载 Profile → 对所有模型打分 → 过滤约束 → 选最优
        3. 若无候选模型 → 使用 fallback_model 并记录 NO_CANDIDATE 警告

        Args:
            templates: 模板名称到 TemplateDef 实例的映射

        Returns:
            填充完成的 RoutingPlanStore 实例
        """
        store = RoutingPlanStore()

        for tpl_name, tpl_def in templates.items():
            for agent_def in tpl_def.agents:
                model, reason = self._select_model(agent_def)
                store.set_model(tpl_name, agent_def.name, model)

            # Audit: log plan generation event per template (FR-8.1)
            template_plan = store.get_template_plan(tpl_name)
            self._audit.log_plan_generation_event(
                trigger_reason=self.trigger_reason,
                template_name=tpl_name,
                assignments=template_plan,
                total_agents=len(tpl_def.agents),
            )

        # 生成完成后输出完整方案日志
        self._log_generated_plans(store, templates)

        return store

    def _select_model(self, agent_def: AgentDef) -> tuple[str, str]:
        """为单个 Agent 选择模型。

        选择优先级:
        1. override_model（管理员直接指定，最高优先级）
        2. Profile 自动选择（打分 + 约束过滤 + 偏好加权）
        3. fallback_model（无候选时降级）

        Args:
            agent_def: Agent 定义

        Returns:
            (模型名称, 选择原因) 元组
        """
        # 覆盖优先
        if agent_def.override_model:
            return agent_def.override_model, "override"

        # Profile 驱动选模型
        profile = self.profile_manager.get_profile(agent_def.capability_profile)
        best_model = self.profile_manager.select_best_model(self.models, profile)

        if best_model is None:
            logger.warning(
                "NO_CANDIDATE: agent='%s', profile='%s' — "
                "无模型满足硬约束，使用 fallback 模型 '%s'",
                agent_def.name,
                agent_def.capability_profile,
                self.fallback_model,
            )
            return self.fallback_model, "fallback"

        return best_model, "scored"

    def _log_generated_plans(
        self, store: RoutingPlanStore, templates: dict[str, TemplateDef]
    ) -> None:
        """输出完整方案生成日志。

        日志格式参考 design.md 方案生成示例节，包含:
        - 每个模板的所有 Agent 分配结果
        - Profile 名称和得分
        - override 标记

        Args:
            store: 已填充的方案表
            templates: 模板定义映射
        """
        lines: list[str] = ["Transaction Router: 方案生成完成"]

        for tpl_name, tpl_def in templates.items():
            lines.append(f"\n模板: {tpl_name}")
            for agent_def in tpl_def.agents:
                model = store.get_model(tpl_name, agent_def.name)
                if agent_def.override_model:
                    lines.append(
                        f"  {agent_def.name:<20} → {model:<20} "
                        f"(override, 管理员指定)"
                    )
                else:
                    profile = self.profile_manager.get_profile(
                        agent_def.capability_profile
                    )
                    # 计算得分用于日志展示
                    score = self._get_model_score(model, profile)
                    preferred = ""
                    if profile.prefer_models and model in profile.prefer_models:
                        preferred = ", preferred"
                    lines.append(
                        f"  {agent_def.name:<20} → {model:<20} "
                        f"(profile={agent_def.capability_profile}, "
                        f"score={score:.2f}{preferred})"
                    )

        logger.info("\n".join(lines))

    def _get_model_score(self, model_name: str, profile: Any) -> float:
        """获取指定模型在指定 Profile 下的得分。

        Args:
            model_name: 模型名称
            profile: CapabilityProfile 实例

        Returns:
            模型得分，未找到模型时返回 0.0
        """
        for m in self.models:
            if m["name"] == model_name:
                return self.profile_manager.score_model(m, profile)
        return 0.0
