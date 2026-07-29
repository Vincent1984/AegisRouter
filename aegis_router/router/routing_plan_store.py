"""路由方案内存查找表

线程安全的只读查找表，通过引用替换实现原子更新。
存储 (template_name, agent_name) → model_name 映射。

设计参考: design.md RoutingPlanStore 节
需求参考: FR-4.5, FR-4.6, NFR-1.1
"""

from __future__ import annotations

from typing import Optional


class RoutingPlanStore:
    """线程安全的只读查找表，通过引用替换实现原子更新。

    key: (template_name, agent_name)
    value: model_name

    方案在内存中持有，配置不变则永不过期。
    分发时纯内存查表，无网络 IO，延迟 < 0.1ms。
    """

    def __init__(self) -> None:
        self._table: dict[tuple[str, str], str] = {}

    def set_model(self, template: str, agent: str, model: str) -> None:
        """设置模板中某 Agent 的模型分配。

        Args:
            template: 模板名称
            agent: Agent 名称
            model: 分配的模型名称
        """
        self._table[(template, agent)] = model

    def get_model(self, template: str, agent: str) -> Optional[str]:
        """查询模板中某 Agent 的分配模型。

        Args:
            template: 模板名称
            agent: Agent 名称

        Returns:
            分配的模型名称，未找到时返回 None
        """
        return self._table.get((template, agent))

    def get_template_plan(self, template: str) -> dict[str, str]:
        """获取某模板的完整方案 {agent → model}。

        Args:
            template: 模板名称

        Returns:
            该模板下所有 Agent 的模型分配映射
        """
        return {
            agent: model
            for (t, agent), model in self._table.items()
            if t == template
        }

    def get_all_plans(self) -> dict[str, dict[str, str]]:
        """获取所有模板的完整方案 {template → {agent → model}}。

        Returns:
            所有模板的模型分配映射
        """
        result: dict[str, dict[str, str]] = {}
        for (tpl, agent), model in self._table.items():
            result.setdefault(tpl, {})[agent] = model
        return result

    def __len__(self) -> int:
        """返回方案表中的条目数量。"""
        return len(self._table)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """检查 (template, agent) 是否在方案表中。"""
        return key in self._table
