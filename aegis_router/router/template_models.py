"""业务流程模板数据模型与 YAML 加载器

加载和管理模板定义（TemplateDef），每个模板包含若干 Agent 定义（AgentDef），
每个 Agent 声明其所需的能力 Profile 和可选的模型覆盖。

设计参考: design.md 模板数据模型节
需求参考: FR-2.1 ~ FR-2.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AgentDef:
    """模板中的 Agent 定义

    Attributes:
        name: Agent 标识
        capability_profile: Profile 名称（用于选模型打分）
        override_model: 管理员直接指定模型（最高优先级，跳过 Profile 评分）
    """

    name: str
    capability_profile: str
    override_model: Optional[str] = None


@dataclass
class TemplateDef:
    """业务流程模板

    Attributes:
        name: 模板名称标识
        description: 模板描述
        agents: 模板中的 Agent 列表
    """

    name: str
    description: str
    agents: list[AgentDef]


def load_templates(
    config_path: str | Path = "config/transaction_templates.yaml",
) -> dict[str, TemplateDef]:
    """从 YAML 文件加载业务流程模板。

    加载逻辑:
    - 文件不存在: 返回空 dict，系统正常启动（所有请求走 fallback）
    - 文件为空: 返回空 dict
    - 文件语法错误: 记录错误日志，返回空 dict（保持上一版配置）
    - Agent 缺少必填字段: 跳过该 Agent 并记录警告

    Args:
        config_path: 模板配置文件路径

    Returns:
        模板名称到 TemplateDef 实例的映射
    """
    path = Path(config_path)

    if not path.exists():
        logger.info(
            "模板配置文件 '%s' 不存在，系统将无预计算方案，所有请求使用 fallback",
            path,
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(
            "模板配置文件 '%s' 解析失败: %s，保持上一版配置",
            path,
            e,
        )
        return {}
    except Exception as e:
        logger.error(
            "读取模板配置文件 '%s' 失败: %s",
            path,
            e,
        )
        return {}

    if not data:
        logger.info("模板配置文件 '%s' 为空，系统将无预计算方案", path)
        return {}

    templates_data = data.get("templates")
    if not templates_data or not isinstance(templates_data, dict):
        logger.info(
            "模板配置文件 '%s' 中无有效 'templates' 字段",
            path,
        )
        return {}

    templates: dict[str, TemplateDef] = {}

    for template_name, template_cfg in templates_data.items():
        if not template_name or not isinstance(template_name, str):
            logger.warning("跳过无效模板名称: %r", template_name)
            continue

        if not isinstance(template_cfg, dict):
            logger.warning("模板 '%s' 配置无效（非字典），跳过", template_name)
            continue

        description = template_cfg.get("description", "")
        agents_data = template_cfg.get("agents", [])

        if not isinstance(agents_data, list):
            logger.warning(
                "模板 '%s' 的 agents 字段不是列表，跳过",
                template_name,
            )
            continue

        agents: list[AgentDef] = []
        for i, agent_cfg in enumerate(agents_data):
            if not isinstance(agent_cfg, dict):
                logger.warning(
                    "模板 '%s' 中第 %d 个 Agent 配置无效（非字典），跳过",
                    template_name,
                    i + 1,
                )
                continue

            agent_name = agent_cfg.get("name", "")
            capability_profile = agent_cfg.get("capability_profile", "")

            # 验证必填字段
            if not agent_name:
                logger.warning(
                    "模板 '%s' 中第 %d 个 Agent 缺少 'name' 字段，跳过",
                    template_name,
                    i + 1,
                )
                continue

            if not capability_profile:
                logger.warning(
                    "模板 '%s' 中 Agent '%s' 缺少 'capability_profile' 字段，跳过",
                    template_name,
                    agent_name,
                )
                continue

            override_model = agent_cfg.get("override_model")

            agents.append(
                AgentDef(
                    name=agent_name,
                    capability_profile=capability_profile,
                    override_model=override_model,
                )
            )

        templates[template_name] = TemplateDef(
            name=template_name,
            description=description,
            agents=agents,
        )

    logger.info(
        "从 '%s' 加载了 %d 个模板（共 %d 个 Agent）",
        path,
        len(templates),
        sum(len(t.agents) for t in templates.values()),
    )
    return templates
