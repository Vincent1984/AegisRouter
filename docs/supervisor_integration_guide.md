# Supervisor 集成指南

> 本文档详细说明如何在 Supervisor 编排代码中注入 `metadata.transaction`，
> 以及各主流 Agent 框架（LangChain、LangGraph、自定义框架）的集成示例。

---

## 目录

- [概述](#概述)
- [核心原理](#核心原理)
- [metadata 注入格式](#metadata-注入格式)
- [LangChain 集成](#langchain-集成)
- [LangGraph 集成](#langgraph-集成)
- [自定义框架集成](#自定义框架集成)
- [多模板多 Agent 流水线示例](#多模板多-agent-流水线示例)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 概述

AegisRouter 事务级路由的核心设计理念：

- **Agent 代码零修改** — Agent 不知道也不需要知道路由逻辑
- **Supervisor 注入路由上下文** — 编排层通过 `metadata.transaction` 告知 AegisRouter 当前请求属于哪个模板的哪个 Agent
- **纯查表分发** — AegisRouter 读取 metadata → 查内存方案表 → 直接路由到预计算模型


```
┌────────────────┐     metadata.transaction      ┌───────────────┐
│   Supervisor   │  ───────────────────────────►  │  AegisRouter  │
│  (编排层)       │   template + agent            │  (路由网关)    │
└────────────────┘                                └───────┬───────┘
        │                                                  │
        │ 调度各 Agent                                      │ 查表分发
        ▼                                                  ▼
┌────────────────┐                                ┌───────────────┐
│    Agent       │                                │   目标 LLM    │
│ (零修改调LLM)   │                                │               │
└────────────────┘                                └───────────────┘
```

---

## 核心原理

1. 管理员在 `config/transaction_templates.yaml` 中定义业务流程模板
2. AegisRouter 启动时为每个 `(template, agent)` 预计算最优模型
3. Supervisor 编排多个 Agent 时，将 `template` 和 `agent` 注入请求 metadata
4. AegisRouter 读取 metadata → 查表 → 覆盖请求中的 `model` 字段 → 转发到目标 LLM
5. Agent 代码中 `model` 字段写什么都无所谓（会被覆盖），通常写 `"placeholder"`

---

## metadata 注入格式

### 请求格式

Supervisor 需要在请求体中注入以下结构：

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "metadata": {
    "transaction": {
      "template": "<模板名称>",
      "agent": "<Agent 标识>"
    }
  }
}
```


| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `template` | string | 是 | 模板名称，对应 `transaction_templates.yaml` 中的 key |
| `agent` | string | 是 | Agent 标识，对应模板中 agents 列表的 `name` 字段 |

### 响应格式

AegisRouter 在响应中附加 `aegis_metadata` 字段，Supervisor 可据此追踪路由结果：

```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}}],
  "aegis_metadata": {
    "template": "resume_screening",
    "agent": "resume_parser",
    "assigned_model": "gemini-2.5-pro",
    "routing_plugin": "transaction",
    "warnings": []
  }
}
```

### 降级行为

| 场景 | 行为 |
|------|------|
| 无 `metadata.transaction` | 使用 fallback 模型 |
| 模板不存在 | HTTP 400 错误 |
| Agent 不在模板中 | fallback 模型 + `UNKNOWN_AGENT` 警告 |

---

## LangChain 集成

### 基本用法 — ChatOpenAI + extra_body

LangChain 的 `ChatOpenAI` 支持 `model_kwargs` 传递额外参数到请求体：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# 创建指向 AegisRouter 的 LLM 客户端
def create_agent_llm(template: str, agent: str) -> ChatOpenAI:
    """为指定的 template + agent 创建 LangChain LLM 实例。

    Args:
        template: 模板名称（如 "resume_screening"）
        agent: Agent 标识（如 "resume_parser"）

    Returns:
        配置好 transaction metadata 的 ChatOpenAI 实例
    """
    return ChatOpenAI(
        model="placeholder",  # 会被 AegisRouter 覆盖
        openai_api_key="sk-your-master-key",
        openai_api_base="http://localhost:8000/v1",
        model_kwargs={
            "extra_body": {
                "metadata": {
                    "transaction": {
                        "template": template,
                        "agent": agent,
                    }
                }
            }
        },
    )


# ═══ Supervisor 编排代码 ═══

class ResumeScreeningSupervisor:
    """简历筛选 Supervisor — 编排 4 个 Agent"""

    def __init__(self):
        self.intent_llm = create_agent_llm("resume_screening", "intent_classifier")
        self.parser_llm = create_agent_llm("resume_screening", "resume_parser")
        self.matcher_llm = create_agent_llm("resume_screening", "skill_matcher")
        self.checker_llm = create_agent_llm("resume_screening", "compliance_checker")

    def run(self, resume_text: str) -> str:
        # Step 1: 意图分类 → 路由到 lightweight 模型
        intent = self.intent_llm.invoke(
            [HumanMessage(content=f"判断以下文本是否为简历: {resume_text[:200]}")]
        )

        # Step 2: 简历解析 → 路由到 long_context 模型
        parsed = self.parser_llm.invoke(
            [HumanMessage(content=f"解析以下简历的结构化信息:\n{resume_text}")]
        )

        # Step 3: 技能匹配 → 路由到 strong_reasoning 模型
        matched = self.matcher_llm.invoke(
            [HumanMessage(content=f"从以下信息中提取并匹配技能:\n{parsed.content}")]
        )

        # Step 4: 合规检查 → 路由到 medium 模型
        result = self.checker_llm.invoke(
            [HumanMessage(content=f"检查以下结果的合规性:\n{matched.content}")]
        )

        return result.content
```


### 使用 LangChain Chain 组合

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建带 metadata 的 Chain
def create_agent_chain(template: str, agent: str, system_prompt: str):
    llm = ChatOpenAI(
        model="placeholder",
        openai_api_key="sk-your-master-key",
        openai_api_base="http://localhost:8000/v1",
        model_kwargs={
            "extra_body": {
                "metadata": {
                    "transaction": {"template": template, "agent": agent}
                }
            }
        },
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    return prompt | llm | StrOutputParser()


# Supervisor 编排多个 Chain
intent_chain = create_agent_chain(
    "resume_screening", "intent_classifier",
    "你是一个意图分类器，判断输入是否为简历。"
)

parser_chain = create_agent_chain(
    "resume_screening", "resume_parser",
    "你是一个简历解析器，提取结构化信息。"
)

# 串联执行
intent_result = intent_chain.invoke({"input": resume_text})
parsed_result = parser_chain.invoke({"input": resume_text})
```

---

## LangGraph 集成

### 基本用法 — 状态机 + metadata 注入

LangGraph 通过状态机（StateGraph）编排多个 Agent 节点。
Supervisor 在定义节点时，为每个节点的 LLM 调用注入对应的 `metadata.transaction`。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ═══ 状态定义 ═══

class PipelineState(TypedDict):
    input_text: str
    intent: str
    parsed_data: str
    skill_match: str
    final_result: str


# ═══ 创建各节点的 LLM ═══

def make_llm(template: str, agent: str) -> ChatOpenAI:
    return ChatOpenAI(
        model="placeholder",
        openai_api_key="sk-your-master-key",
        openai_api_base="http://localhost:8000/v1",
        model_kwargs={
            "extra_body": {
                "metadata": {
                    "transaction": {"template": template, "agent": agent}
                }
            }
        },
    )



# ═══ 节点函数 ═══

intent_llm = make_llm("resume_screening", "intent_classifier")
parser_llm = make_llm("resume_screening", "resume_parser")
matcher_llm = make_llm("resume_screening", "skill_matcher")
checker_llm = make_llm("resume_screening", "compliance_checker")


def classify_intent(state: PipelineState) -> PipelineState:
    """意图分类节点 → AegisRouter 路由到 lightweight 模型"""
    resp = intent_llm.invoke(
        [HumanMessage(content=f"判断以下文本是否为简历:\n{state['input_text'][:200]}")]
    )
    return {"intent": resp.content}


def parse_resume(state: PipelineState) -> PipelineState:
    """简历解析节点 → AegisRouter 路由到 long_context 模型"""
    resp = parser_llm.invoke(
        [HumanMessage(content=f"解析简历:\n{state['input_text']}")]
    )
    return {"parsed_data": resp.content}


def match_skills(state: PipelineState) -> PipelineState:
    """技能匹配节点 → AegisRouter 路由到 strong_reasoning 模型"""
    resp = matcher_llm.invoke(
        [HumanMessage(content=f"匹配技能:\n{state['parsed_data']}")]
    )
    return {"skill_match": resp.content}


def check_compliance(state: PipelineState) -> PipelineState:
    """合规检查节点 → AegisRouter 路由到 medium 模型"""
    resp = checker_llm.invoke(
        [HumanMessage(content=f"合规检查:\n{state['skill_match']}")]
    )
    return {"final_result": resp.content}


# ═══ 构建 LangGraph ═══

def build_resume_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # 添加节点
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("parse_resume", parse_resume)
    graph.add_node("match_skills", match_skills)
    graph.add_node("check_compliance", check_compliance)

    # 添加边（顺序执行）
    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "parse_resume")
    graph.add_edge("parse_resume", "match_skills")
    graph.add_edge("match_skills", "check_compliance")
    graph.add_edge("check_compliance", END)

    return graph.compile()


# ═══ 执行 ═══

app = build_resume_graph()
result = app.invoke({"input_text": "张三，5年Python开发经验..."})
print(result["final_result"])
```


### 带条件分支的 LangGraph

```python
from langgraph.graph import StateGraph, END


class CodeReviewState(TypedDict):
    code: str
    analysis: str
    issues: str
    has_issues: bool
    fix_suggestion: str


analyzer_llm = make_llm("code_review", "code_analyzer")
detector_llm = make_llm("code_review", "issue_detector")
fixer_llm = make_llm("code_review", "fix_suggester")


def analyze_code(state: CodeReviewState) -> CodeReviewState:
    resp = analyzer_llm.invoke(
        [HumanMessage(content=f"分析以下代码结构:\n{state['code']}")]
    )
    return {"analysis": resp.content}


def detect_issues(state: CodeReviewState) -> CodeReviewState:
    resp = detector_llm.invoke(
        [HumanMessage(content=f"检测问题:\n{state['analysis']}")]
    )
    has_issues = "无问题" not in resp.content
    return {"issues": resp.content, "has_issues": has_issues}


def suggest_fix(state: CodeReviewState) -> CodeReviewState:
    resp = fixer_llm.invoke(
        [HumanMessage(content=f"为以下问题提供修复建议:\n{state['issues']}")]
    )
    return {"fix_suggestion": resp.content}


def route_after_detection(state: CodeReviewState) -> str:
    """条件路由：有问题则生成修复建议，无问题直接结束"""
    return "suggest_fix" if state.get("has_issues") else END


# 构建带条件分支的图
graph = StateGraph(CodeReviewState)
graph.add_node("analyze_code", analyze_code)
graph.add_node("detect_issues", detect_issues)
graph.add_node("suggest_fix", suggest_fix)

graph.set_entry_point("analyze_code")
graph.add_edge("analyze_code", "detect_issues")
graph.add_conditional_edges("detect_issues", route_after_detection)
graph.add_edge("suggest_fix", END)

app = graph.compile()
```

---

## 自定义框架集成

### 使用 httpx（同步）

```python
import httpx

AEGIS_BASE_URL = "http://localhost:8000/v1/chat/completions"
AEGIS_API_KEY = "sk-your-master-key"


def call_agent(
    template: str,
    agent: str,
    messages: list[dict],
    stream: bool = False,
) -> dict:
    """通用 Agent 调用函数 — 注入 transaction metadata。

    Args:
        template: 模板名称
        agent: Agent 标识
        messages: OpenAI 格式的消息列表
        stream: 是否启用流式响应

    Returns:
        AegisRouter 响应（含 aegis_metadata）
    """
    payload = {
        "model": "placeholder",
        "messages": messages,
        "stream": stream,
        "metadata": {
            "transaction": {
                "template": template,
                "agent": agent,
            }
        },
    }

    resp = httpx.post(
        AEGIS_BASE_URL,
        headers={"Authorization": f"Bearer {AEGIS_API_KEY}"},
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()



# Supervisor 编排示例
def run_code_review(code: str) -> str:
    # Step 1: 代码分析
    r1 = call_agent("code_review", "code_analyzer", [
        {"role": "user", "content": f"分析以下代码:\n{code}"}
    ])
    analysis = r1["choices"][0]["message"]["content"]
    print(f"[code_analyzer] 路由到: {r1['aegis_metadata']['assigned_model']}")

    # Step 2: 问题检测
    r2 = call_agent("code_review", "issue_detector", [
        {"role": "user", "content": f"检测以下分析中的问题:\n{analysis}"}
    ])
    issues = r2["choices"][0]["message"]["content"]
    print(f"[issue_detector] 路由到: {r2['aegis_metadata']['assigned_model']}")

    # Step 3: 修复建议
    r3 = call_agent("code_review", "fix_suggester", [
        {"role": "user", "content": f"为以下问题提供修复代码:\n{issues}"}
    ])
    print(f"[fix_suggester] 路由到: {r3['aegis_metadata']['assigned_model']}")

    return r3["choices"][0]["message"]["content"]
```

### 使用 aiohttp（异步）

```python
import aiohttp
import asyncio

AEGIS_BASE_URL = "http://localhost:8000/v1/chat/completions"
AEGIS_API_KEY = "sk-your-master-key"


async def call_agent_async(
    session: aiohttp.ClientSession,
    template: str,
    agent: str,
    messages: list[dict],
) -> dict:
    """异步 Agent 调用 — 支持并发多 Agent 场景。"""
    payload = {
        "model": "placeholder",
        "messages": messages,
        "metadata": {
            "transaction": {
                "template": template,
                "agent": agent,
            }
        },
    }

    async with session.post(
        AEGIS_BASE_URL,
        headers={"Authorization": f"Bearer {AEGIS_API_KEY}"},
        json=payload,
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def run_supplier_evaluation(data: str):
    """并发执行供应商评估流程中的独立 Agent"""
    async with aiohttp.ClientSession() as session:
        # Step 1: 数据采集（独立 Agent，可并行）
        r1 = await call_agent_async(
            session, "supplier_evaluation", "data_collector",
            [{"role": "user", "content": f"采集供应商数据:\n{data}"}]
        )
        collected = r1["choices"][0]["message"]["content"]

        # Step 2: 绩效评分 和 合规检查 可并发执行
        r2, r3 = await asyncio.gather(
            call_agent_async(
                session, "supplier_evaluation", "performance_scorer",
                [{"role": "user", "content": f"评估绩效:\n{collected}"}]
            ),
            call_agent_async(
                session, "supplier_evaluation", "compliance_checker",
                [{"role": "user", "content": f"合规检查:\n{collected}"}]
            ),
        )

        # Step 3: 综合分级
        combined = f"绩效: {r2['choices'][0]['message']['content']}\n"
        combined += f"合规: {r3['choices'][0]['message']['content']}"

        r4 = await call_agent_async(
            session, "supplier_evaluation", "tier_determiner",
            [{"role": "user", "content": f"综合判定供应商等级:\n{combined}"}]
        )

        return r4["choices"][0]["message"]["content"]


# 运行
result = asyncio.run(run_supplier_evaluation("供应商A的数据..."))
```


### 使用 OpenAI SDK（推荐）

```python
from openai import OpenAI, AsyncOpenAI


# 同步客户端
client = OpenAI(
    api_key="sk-your-master-key",
    base_url="http://localhost:8000/v1",
)


def call_with_transaction(template: str, agent: str, content: str) -> str:
    """使用 OpenAI SDK 的 extra_body 注入 metadata"""
    response = client.chat.completions.create(
        model="placeholder",
        messages=[{"role": "user", "content": content}],
        extra_body={
            "metadata": {
                "transaction": {
                    "template": template,
                    "agent": agent,
                }
            }
        },
    )
    return response.choices[0].message.content


# 异步客户端
async_client = AsyncOpenAI(
    api_key="sk-your-master-key",
    base_url="http://localhost:8000/v1",
)


async def call_with_transaction_async(template: str, agent: str, content: str) -> str:
    """异步版本"""
    response = await async_client.chat.completions.create(
        model="placeholder",
        messages=[{"role": "user", "content": content}],
        extra_body={
            "metadata": {
                "transaction": {
                    "template": template,
                    "agent": agent,
                }
            }
        },
    )
    return response.choices[0].message.content
```

---

## 多模板多 Agent 流水线示例

以下示例展示一个复杂场景：Supervisor 根据用户意图，动态选择不同模板执行。

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-master-key",
    base_url="http://localhost:8000/v1",
)


class MultiTemplateSupervisor:
    """多模板 Supervisor — 根据意图动态选择流程模板。

    支持的模板:
      - resume_screening: 简历筛选
      - code_review: 代码审查
      - supplier_evaluation: 供应商评估
    """

    # 模板 → Agent 列表的映射（按执行顺序）
    TEMPLATE_AGENTS = {
        "resume_screening": [
            "intent_classifier",
            "resume_parser",
            "skill_matcher",
            "compliance_checker",
        ],
        "code_review": [
            "code_analyzer",
            "issue_detector",
            "fix_suggester",
        ],
        "supplier_evaluation": [
            "data_collector",
            "performance_scorer",
            "compliance_checker",
            "tier_determiner",
        ],
    }


    def _call_agent(self, template: str, agent: str, content: str) -> dict:
        """调用单个 Agent，注入 transaction metadata。"""
        response = client.chat.completions.create(
            model="placeholder",
            messages=[{"role": "user", "content": content}],
            extra_body={
                "metadata": {
                    "transaction": {
                        "template": template,
                        "agent": agent,
                    }
                }
            },
        )

        # 从 response 中提取 aegis_metadata（如果可用）
        result = {
            "content": response.choices[0].message.content,
            "model": getattr(response, "model", "unknown"),
        }

        # 解析原始响应获取 aegis_metadata
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        if "aegis_metadata" in raw:
            result["aegis_metadata"] = raw["aegis_metadata"]

        return result

    def run_pipeline(self, template: str, input_text: str) -> list[dict]:
        """按顺序执行指定模板的所有 Agent。

        Args:
            template: 模板名称
            input_text: 初始输入文本

        Returns:
            每个 Agent 的执行结果列表
        """
        if template not in self.TEMPLATE_AGENTS:
            raise ValueError(
                f"未知模板 '{template}'，可用: {list(self.TEMPLATE_AGENTS.keys())}"
            )

        agents = self.TEMPLATE_AGENTS[template]
        results = []
        current_input = input_text

        for agent_name in agents:
            print(f"  → 执行 Agent: {agent_name} (template={template})")

            result = self._call_agent(template, agent_name, current_input)
            results.append({"agent": agent_name, **result})

            # 下一个 Agent 使用上一个 Agent 的输出作为输入
            current_input = result["content"]

            print(f"    ✓ 完成 (model={result.get('model', 'N/A')})")

        return results

    def route_and_execute(self, user_input: str) -> list[dict]:
        """根据用户输入自动判断模板并执行。

        先用一个轻量 Agent 判断意图，然后路由到对应模板。
        """
        # 意图判断（使用 resume_screening 的 intent_classifier）
        intent_result = self._call_agent(
            "resume_screening", "intent_classifier",
            f"判断以下输入属于哪种场景(resume/code/supplier): {user_input[:200]}"
        )

        intent = intent_result["content"].strip().lower()

        # 映射到模板
        template_map = {
            "resume": "resume_screening",
            "code": "code_review",
            "supplier": "supplier_evaluation",
        }

        template = template_map.get(intent, "resume_screening")
        print(f"[Supervisor] 意图={intent} → 模板={template}")

        return self.run_pipeline(template, user_input)


# ═══ 使用示例 ═══

supervisor = MultiTemplateSupervisor()

# 直接指定模板执行
results = supervisor.run_pipeline("code_review", "def foo():\n  return None")

# 自动路由执行
results = supervisor.route_and_execute("请审查这段Python代码...")
```

---

## 最佳实践

### 1. metadata 封装为工厂函数

避免在每个调用点重复构造 metadata 结构：

```python
def make_transaction_metadata(template: str, agent: str) -> dict:
    """标准化 transaction metadata 构造。"""
    return {
        "metadata": {
            "transaction": {
                "template": template,
                "agent": agent,
            }
        }
    }


# 使用
response = client.chat.completions.create(
    model="placeholder",
    messages=[...],
    extra_body=make_transaction_metadata("code_review", "code_analyzer"),
)
```


### 2. 响应中的 aegis_metadata 追踪

记录每个 Agent 实际被路由到的模型，用于调试和审计：

```python
def call_and_log(template: str, agent: str, content: str) -> str:
    """调用 Agent 并记录路由信息。"""
    import httpx
    import logging

    logger = logging.getLogger("supervisor")

    resp = httpx.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer sk-your-master-key"},
        json={
            "model": "placeholder",
            "messages": [{"role": "user", "content": content}],
            "metadata": {"transaction": {"template": template, "agent": agent}},
        },
    )
    data = resp.json()

    # 记录路由结果
    meta = data.get("aegis_metadata", {})
    logger.info(
        "Agent 调用完成: template=%s, agent=%s, model=%s, warnings=%s",
        meta.get("template"),
        meta.get("agent"),
        meta.get("assigned_model"),
        meta.get("warnings", []),
    )

    # 检查警告
    warnings = meta.get("warnings", [])
    if "UNKNOWN_AGENT" in warnings:
        logger.warning("Agent '%s' 不在模板 '%s' 中，已降级到 fallback 模型", agent, template)

    return data["choices"][0]["message"]["content"]
```

### 3. 错误处理

```python
import httpx


class AegisRouterError(Exception):
    """AegisRouter 返回的错误。"""
    pass


class TemplateNotFoundError(AegisRouterError):
    """模板不存在 (HTTP 400)。"""
    pass


def call_agent_safe(template: str, agent: str, content: str) -> str:
    """带完善错误处理的 Agent 调用。"""
    try:
        resp = httpx.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Authorization": "Bearer sk-your-master-key"},
            json={
                "model": "placeholder",
                "messages": [{"role": "user", "content": content}],
                "metadata": {"transaction": {"template": template, "agent": agent}},
            },
            timeout=60.0,
        )

        if resp.status_code == 400:
            error_body = resp.json()
            raise TemplateNotFoundError(
                f"模板 '{template}' 不存在: {error_body.get('error', '')}"
            )

        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    except httpx.TimeoutException:
        raise AegisRouterError(f"Agent '{agent}' 调用超时")
    except httpx.ConnectError:
        raise AegisRouterError("无法连接 AegisRouter (http://localhost:8000)")
```

### 4. Agent 代码保持零修改

Agent 内部代码不应感知 `metadata.transaction`。metadata 应由 Supervisor 层统一注入：

```python
# ✅ 正确做法：Agent 接收 metadata 参数，但不理解其含义
class GenericAgent:
    def __init__(self, client, extra_body: dict | None = None):
        self.client = client
        self.extra_body = extra_body or {}

    def run(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="placeholder",
            messages=[{"role": "user", "content": prompt}],
            extra_body=self.extra_body,  # Supervisor 传入，Agent 不关心内容
        )
        return response.choices[0].message.content


# Supervisor 创建 Agent 时注入 metadata
parser_agent = GenericAgent(
    client=client,
    extra_body=make_transaction_metadata("resume_screening", "resume_parser"),
)

# Agent 执行时完全不感知路由逻辑
result = parser_agent.run("解析这份简历...")
```


### 5. 流式响应处理

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-master-key",
    base_url="http://localhost:8000/v1",
)


def call_agent_stream(template: str, agent: str, content: str):
    """流式调用 Agent — 适用于需要实时展示输出的场景。"""
    stream = client.chat.completions.create(
        model="placeholder",
        messages=[{"role": "user", "content": content}],
        stream=True,
        extra_body={
            "metadata": {
                "transaction": {
                    "template": template,
                    "agent": agent,
                }
            }
        },
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content_piece = chunk.choices[0].delta.content
            full_response += content_piece
            print(content_piece, end="", flush=True)

    print()  # 换行
    return full_response
```

### 6. 配置验证脚本

部署前验证 AegisRouter 路由方案是否正确：

```python
"""验证脚本 — 确认所有 template + agent 组合正常工作。"""
import httpx

AEGIS_URL = "http://localhost:8000/v1/chat/completions"
API_KEY = "sk-your-master-key"

# 所有预期的 (template, agent) 组合
EXPECTED_ROUTES = [
    ("resume_screening", "intent_classifier"),
    ("resume_screening", "resume_parser"),
    ("resume_screening", "skill_matcher"),
    ("resume_screening", "compliance_checker"),
    ("code_review", "code_analyzer"),
    ("code_review", "issue_detector"),
    ("code_review", "fix_suggester"),
    ("supplier_evaluation", "data_collector"),
    ("supplier_evaluation", "performance_scorer"),
    ("supplier_evaluation", "compliance_checker"),
    ("supplier_evaluation", "tier_determiner"),
]


def verify_routes():
    """验证所有路由是否正常。"""
    print("=== AegisRouter 路由验证 ===\n")

    for template, agent in EXPECTED_ROUTES:
        resp = httpx.post(
            AEGIS_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": "placeholder",
                "messages": [{"role": "user", "content": "ping"}],
                "metadata": {"transaction": {"template": template, "agent": agent}},
            },
            timeout=30.0,
        )

        if resp.status_code == 200:
            meta = resp.json().get("aegis_metadata", {})
            model = meta.get("assigned_model", "?")
            warnings = meta.get("warnings", [])
            status = "⚠️" if warnings else "✓"
            print(f"  {status} {template}/{agent} → {model} {warnings}")
        else:
            print(f"  ✗ {template}/{agent} → HTTP {resp.status_code}")

    print("\n验证完成。")


if __name__ == "__main__":
    verify_routes()
```

---

## 常见问题

### Q: `model` 字段应该填什么？

A: 任意值均可（如 `"placeholder"`）。AegisRouter 会根据 `metadata.transaction` 查表后覆盖这个字段。
但如果你不注入 `metadata.transaction`，AegisRouter 会使用配置的 fallback 模型。

### Q: Agent 需要修改代码吗？

A: **不需要**。这是事务级路由的核心设计原则。Agent 正常调用 LLM，
metadata 由 Supervisor 在创建 Agent 实例时统一注入。

### Q: 同一个 Agent 在不同模板中会使用不同模型吗？

A: **是的**。AegisRouter 的路由 key 是 `(template, agent)` 二元组。
同一个 `compliance_checker` 在 `resume_screening` 和 `supplier_evaluation` 中
可以根据各自模板的 Profile 设置分配到不同模型。

### Q: 如何确认请求被路由到了正确的模型？

A: 检查响应中的 `aegis_metadata.assigned_model` 字段。
也可以查看 AegisRouter 启动日志中的方案表。

### Q: 如果引用了不存在的模板会怎样？

A: AegisRouter 返回 HTTP 400，错误信息为 `Template 'xxx' not found`。
建议在 Supervisor 中做好错误处理。

### Q: 如果引用了模板中不存在的 Agent 会怎样？

A: 请求不会失败。AegisRouter 使用 fallback 模型处理请求，
并在响应的 `aegis_metadata.warnings` 中包含 `UNKNOWN_AGENT` 警告。

### Q: 配置变更后需要重启 Supervisor 吗？

A: **不需要**。AegisRouter 支持配置热更新，修改 `transaction_templates.yaml`
或 `capability_profiles.yaml` 后方案自动重算。Supervisor 端无需任何操作，
下一次请求即使用新方案。
