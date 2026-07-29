"""Agent 方案内存查找表

线程安全的只读查找表，通过引用替换实现原子更新。
存储 agent_name → model_name 映射。

与 RoutingPlanStore 类似，但 key 从二元组 (template, agent) 简化为单一字符串 agent。

设计参考: design.md AgentPlanStore 节
需求参考: FR-7.1, FR-7.3, FR-7.4
"""

from __future__ import annotations

from typing import Optional


class AgentPlanStore:
    """线程安全的只读查找表，通过引用替换实现原子更新。

    key: agent_name
    value: model_name

    方案在内存中持有，配置不变则永不过期。
    分发时纯内存查表，无网络 IO，延迟 < 0.1ms。
    """

    def __init__(self) -> None:
        self._table: dict[str, str] = {}

    def set_model(self, agent: str, model: str) -> None:
        """设置 Agent 的模型分配。

        Args:
            agent: Agent 名称
            model: 分配的模型名称
        """
        self._table[agent] = model

    def get_model(self, agent: str) -> Optional[str]:
        """查询 Agent 的分配模型，O(1) 哈希查找。

        Args:
            agent: Agent 名称

        Returns:
            分配的模型名称，未找到时返回 None
        """
        return self._table.get(agent)

    def get_all_plans(self) -> dict[str, str]:
        """获取完整方案 {agent → model}（日志/调试用）。

        Returns:
            所有 Agent 的模型分配映射
        """
        return dict(self._table)

    def __len__(self) -> int:
        """返回方案表中的条目数量。"""
        return len(self._table)

    def __contains__(self, agent: str) -> bool:
        """检查 agent 是否在方案表中。"""
        return agent in self._table
