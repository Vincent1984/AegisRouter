#!/usr/bin/env python3
"""
AegisRouter Supervisor 集成示例脚本
===================================

本脚本演示如何在 Supervisor 编排代码中注入 `metadata.transaction`，
实现事务级路由的完整流程。

功能覆盖：
  1. 单模板多 Agent 流水线 (resume_screening)
  2. 多模板路由 (code_review, supplier_evaluation)
  3. 流式响应处理
  4. 错误处理（未知模板、未知 Agent）
  5. 响应 aegis_metadata 检查

运行方式：
  # 实际连接 AegisRouter（需要启动 AegisRouter 实例）
  python examples/supervisor_integration_example.py

  # Dry-run 模式（不需要 AegisRouter 实例，使用模拟响应）
  python examples/supervisor_integration_example.py --dry-run

依赖：
  pip install openai httpx

环境变量（可选）：
  AEGIS_BASE_URL  - AegisRouter 地址（默认 http://localhost:8000/v1）
  AEGIS_API_KEY   - API Key（默认 sk-aegis-master-key）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Generator

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

AEGIS_BASE_URL = os.environ.get("AEGIS_BASE_URL", "http://localhost:8000/v1")
AEGIS_API_KEY = os.environ.get("AEGIS_API_KEY", "sk-aegis-master-key")


# ---------------------------------------------------------------------------
# Dry-run 模拟层
# ---------------------------------------------------------------------------


class MockResponse:
    """模拟 OpenAI SDK 的 ChatCompletion 响应对象。"""

    def __init__(self, template: str, agent: str, content: str, model: str):
        self.choices = [MockChoice(content)]
        self.model = model
        # 模拟 aegis_metadata（实际由 AegisRouter 注入到响应中）
        self.aegis_metadata = {
            "template": template,
            "agent": agent,
            "assigned_model": model,
            "routing_plugin": "transaction",
            "warnings": [],
        }

    def model_dump(self) -> dict:
        return {
            "choices": [{"message": {"role": "assistant", "content": self.choices[0].message.content}}],
            "model": self.model,
            "aegis_metadata": self.aegis_metadata,
        }


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content


class MockStreamChunk:
    """模拟流式响应的 chunk。"""

    def __init__(self, content: str | None):
        self.choices = [MockStreamChoice(content)]


class MockStreamChoice:
    def __init__(self, content: str | None):
        self.delta = MockDelta(content)


class MockDelta:
    def __init__(self, content: str | None):
        self.content = content


# 模拟模型分配表（与 transaction_templates.yaml 对应）
MOCK_MODEL_MAP = {
    ("resume_screening", "intent_classifier"): "gemini-2.5-flash",
    ("resume_screening", "resume_parser"): "gemini-2.5-pro",
    ("resume_screening", "skill_matcher"): "o3",
    ("resume_screening", "compliance_checker"): "gpt-4.1",
    ("code_review", "code_analyzer"): "claude-sonnet-4-20250514",
    ("code_review", "issue_detector"): "o3",
    ("code_review", "fix_suggester"): "claude-sonnet-4-20250514",
    ("supplier_evaluation", "data_collector"): "gemini-2.5-flash",
    ("supplier_evaluation", "performance_scorer"): "gpt-4.1",
    ("supplier_evaluation", "compliance_checker"): "o3",
    ("supplier_evaluation", "tier_determiner"): "o3",
}


def mock_completion(template: str, agent: str, content: str) -> MockResponse:
    """生成模拟响应。"""
    model = MOCK_MODEL_MAP.get((template, agent), "fallback-model")
    reply = f"[DRY-RUN] Agent '{agent}' (template='{template}') 已处理请求。分配模型: {model}"
    return MockResponse(template=template, agent=agent, content=reply, model=model)


def mock_stream(template: str, agent: str, content: str) -> Generator[MockStreamChunk, None, None]:
    """生成模拟流式响应。"""
    model = MOCK_MODEL_MAP.get((template, agent), "fallback-model")
    text = f"[DRY-RUN 流式] Agent '{agent}' 处理中 (model={model})... 完成。"
    # 模拟逐字输出
    for char in text:
        yield MockStreamChunk(char)
    # 结束 chunk
    yield MockStreamChunk(None)


# ---------------------------------------------------------------------------
# AegisRouter 客户端封装
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """单个 Agent 调用的结果。"""

    agent: str
    template: str
    content: str
    assigned_model: str
    warnings: list[str]


class AegisClient:
    """AegisRouter 客户端 — 封装 transaction metadata 注入逻辑。

    支持两种模式:
      - 实际模式: 通过 OpenAI SDK 连接 AegisRouter
      - Dry-run 模式: 使用模拟响应，无需真实实例
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        if not dry_run:
            try:
                from openai import OpenAI
            except ImportError:
                print("错误: 实际模式需要 openai 库。请运行: pip install openai")
                print("或使用 --dry-run 模式运行。")
                sys.exit(1)
            self.client = OpenAI(api_key=AEGIS_API_KEY, base_url=AEGIS_BASE_URL)
        else:
            self.client = None

    def call_agent(
        self,
        template: str,
        agent: str,
        messages: list[dict[str, str]],
        stream: bool = False,
    ) -> AgentResult | Generator[str, None, None]:
        """调用指定 template + agent，注入 transaction metadata。

        Args:
            template: 模板名称（对应 transaction_templates.yaml 中的 key）
            agent: Agent 标识（对应模板中 agents 列表的 name）
            messages: OpenAI 格式的消息列表
            stream: 是否启用流式响应

        Returns:
            AgentResult 或流式 Generator
        """
        if stream:
            return self._call_stream(template, agent, messages)
        return self._call_sync(template, agent, messages)

    def _call_sync(self, template: str, agent: str, messages: list[dict]) -> AgentResult:
        """同步调用。"""
        if self.dry_run:
            content = messages[-1]["content"] if messages else ""
            resp = mock_completion(template, agent, content)
            return AgentResult(
                agent=agent,
                template=template,
                content=resp.choices[0].message.content,
                assigned_model=resp.aegis_metadata["assigned_model"],
                warnings=resp.aegis_metadata["warnings"],
            )

        # 实际调用 AegisRouter — 通过 extra_body 注入 metadata.transaction
        response = self.client.chat.completions.create(
            model="placeholder",  # 会被 AegisRouter 覆盖
            messages=messages,
            extra_body={
                "metadata": {
                    "transaction": {
                        "template": template,
                        "agent": agent,
                    }
                }
            },
        )

        # 解析 aegis_metadata（从原始响应中获取）
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        aegis_meta = raw.get("aegis_metadata", {})

        return AgentResult(
            agent=agent,
            template=template,
            content=response.choices[0].message.content,
            assigned_model=aegis_meta.get("assigned_model", response.model),
            warnings=aegis_meta.get("warnings", []),
        )

    def _call_stream(self, template: str, agent: str, messages: list[dict]) -> Generator[str, None, None]:
        """流式调用 — 逐 chunk 返回内容。"""
        if self.dry_run:
            content = messages[-1]["content"] if messages else ""
            for chunk in mock_stream(template, agent, content):
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        # 实际流式调用
        stream = self.client.chat.completions.create(
            model="placeholder",
            messages=messages,
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

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ---------------------------------------------------------------------------
# 示例 1: 单模板多 Agent 流水线 (resume_screening)
# ---------------------------------------------------------------------------


def demo_single_template_pipeline(client: AegisClient) -> None:
    """演示单模板多 Agent 流水线。

    模板: resume_screening
    流程: intent_classifier → resume_parser → skill_matcher → compliance_checker
    """
    print("\n" + "=" * 70)
    print("示例 1: 单模板多 Agent 流水线 (resume_screening)")
    print("=" * 70)

    resume_text = (
        "张三，男，28岁。5年 Python 后端开发经验。\n"
        "技能: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes\n"
        "学历: 北京大学计算机科学学士\n"
        "经历: 某大厂高级工程师 (2021-至今)"
    )

    # Step 1: 意图分类 — 路由到 lightweight 模型
    print("\n[Step 1] 意图分类 (intent_classifier)")
    result = client.call_agent(
        template="resume_screening",
        agent="intent_classifier",
        messages=[{"role": "user", "content": f"判断以下文本是否为简历:\n{resume_text[:100]}"}],
    )
    print(f"  → 分配模型: {result.assigned_model}")
    print(f"  → 结果: {result.content[:80]}...")

    # Step 2: 简历解析 — 路由到 long_context 模型
    print("\n[Step 2] 简历解析 (resume_parser)")
    result = client.call_agent(
        template="resume_screening",
        agent="resume_parser",
        messages=[{"role": "user", "content": f"解析以下简历的结构化信息:\n{resume_text}"}],
    )
    print(f"  → 分配模型: {result.assigned_model}")
    print(f"  → 结果: {result.content[:80]}...")
    parsed_content = result.content

    # Step 3: 技能匹配 — 路由到 strong_reasoning 模型
    print("\n[Step 3] 技能匹配 (skill_matcher)")
    result = client.call_agent(
        template="resume_screening",
        agent="skill_matcher",
        messages=[{"role": "user", "content": f"从以下信息中匹配技能:\n{parsed_content[:200]}"}],
    )
    print(f"  → 分配模型: {result.assigned_model}")
    print(f"  → 结果: {result.content[:80]}...")

    # Step 4: 合规检查 — 路由到 medium 模型
    print("\n[Step 4] 合规检查 (compliance_checker)")
    result = client.call_agent(
        template="resume_screening",
        agent="compliance_checker",
        messages=[{"role": "user", "content": f"检查合规性:\n{result.content[:200]}"}],
    )
    print(f"  → 分配模型: {result.assigned_model}")
    print(f"  → 结果: {result.content[:80]}...")
    print(f"  → 警告: {result.warnings}")

    print("\n✓ 简历筛选流水线完成")


# ---------------------------------------------------------------------------
# 示例 2: 多模板路由
# ---------------------------------------------------------------------------


def demo_multi_template_routing(client: AegisClient) -> None:
    """演示多模板路由 — 同一 Supervisor 根据意图路由到不同模板。

    展示同一个 Agent 名称 (compliance_checker) 在不同模板中的路由差异。
    """
    print("\n" + "=" * 70)
    print("示例 2: 多模板路由 (同一 Agent 不同模板)")
    print("=" * 70)

    # 在 resume_screening 中调用 compliance_checker → medium 模型
    print("\n[A] resume_screening/compliance_checker")
    result_a = client.call_agent(
        template="resume_screening",
        agent="compliance_checker",
        messages=[{"role": "user", "content": "检查简历信息合规性"}],
    )
    print(f"  → 分配模型: {result_a.assigned_model}")

    # 在 supplier_evaluation 中调用 compliance_checker → strong_reasoning 模型
    print("\n[B] supplier_evaluation/compliance_checker")
    result_b = client.call_agent(
        template="supplier_evaluation",
        agent="compliance_checker",
        messages=[{"role": "user", "content": "检查供应商合规性（复杂法规分析）"}],
    )
    print(f"  → 分配模型: {result_b.assigned_model}")

    # 使用 code_review 模板
    print("\n[C] code_review/code_analyzer")
    result_c = client.call_agent(
        template="code_review",
        agent="code_analyzer",
        messages=[{"role": "user", "content": "分析以下代码:\ndef foo(): return None"}],
    )
    print(f"  → 分配模型: {result_c.assigned_model}")

    print("\n✓ 多模板路由演示完成")
    print("  注意: 同名 Agent 'compliance_checker' 在不同模板中被路由到不同模型")


# ---------------------------------------------------------------------------
# 示例 3: 流式响应
# ---------------------------------------------------------------------------


def demo_streaming(client: AegisClient) -> None:
    """演示流式响应处理。

    流式模式适用于需要实时展示输出的场景（如 Web UI 逐字显示）。
    """
    print("\n" + "=" * 70)
    print("示例 3: 流式响应")
    print("=" * 70)

    print("\n[流式输出] code_review/fix_suggester:")
    print("  ", end="")

    full_response = ""
    for chunk in client.call_agent(
        template="code_review",
        agent="fix_suggester",
        messages=[{"role": "user", "content": "为以下代码提供修复建议: def foo(): return None"}],
        stream=True,
    ):
        full_response += chunk
        print(chunk, end="", flush=True)

    print()  # 换行
    print(f"\n✓ 流式响应完成 (共 {len(full_response)} 字符)")


# ---------------------------------------------------------------------------
# 示例 4: 错误处理
# ---------------------------------------------------------------------------


def demo_error_handling(client: AegisClient) -> None:
    """演示错误处理场景。

    场景 A: 未知模板 → HTTP 400
    场景 B: 未知 Agent → fallback 模型 + UNKNOWN_AGENT 警告
    """
    print("\n" + "=" * 70)
    print("示例 4: 错误处理")
    print("=" * 70)

    # 场景 A: 未知模板
    print("\n[A] 调用不存在的模板 'nonexistent_template'")
    if client.dry_run:
        # Dry-run 模式下模拟 HTTP 400 错误
        print("  → [DRY-RUN] 模拟 HTTP 400: Template 'nonexistent_template' not found")
        print("  → 正确行为: AegisRouter 返回 HTTP 400，Supervisor 应捕获此错误")
    else:
        try:
            client.call_agent(
                template="nonexistent_template",
                agent="some_agent",
                messages=[{"role": "user", "content": "test"}],
            )
        except Exception as e:
            print(f"  → 捕获错误: {type(e).__name__}: {e}")
            print("  → 正确行为: Supervisor 应处理此错误并反馈给用户")

    # 场景 B: 未知 Agent（模板存在但 Agent 不在其中）
    print("\n[B] 调用模板中不存在的 Agent 'unknown_agent'")
    if client.dry_run:
        # 模拟 UNKNOWN_AGENT 警告
        result = AgentResult(
            agent="unknown_agent",
            template="resume_screening",
            content="[DRY-RUN] 使用 fallback 模型处理的响应",
            assigned_model="fallback-model",
            warnings=["UNKNOWN_AGENT"],
        )
        print(f"  → 分配模型: {result.assigned_model} (fallback)")
        print(f"  → 警告: {result.warnings}")
        print("  → 正确行为: 请求不会失败，但使用 fallback 模型且带有警告")
    else:
        result = client.call_agent(
            template="resume_screening",
            agent="unknown_agent",
            messages=[{"role": "user", "content": "test"}],
        )
        print(f"  → 分配模型: {result.assigned_model}")
        print(f"  → 警告: {result.warnings}")
        if "UNKNOWN_AGENT" in result.warnings:
            print("  → 检测到 UNKNOWN_AGENT 警告，已降级到 fallback 模型")

    print("\n✓ 错误处理演示完成")


# ---------------------------------------------------------------------------
# 示例 5: 响应 metadata 检查
# ---------------------------------------------------------------------------


def demo_metadata_inspection(client: AegisClient) -> None:
    """演示如何检查响应中的 aegis_metadata 进行路由追踪和审计。

    aegis_metadata 包含:
      - template: 使用的模板名
      - agent: Agent 标识
      - assigned_model: 实际分配的模型
      - routing_plugin: 路由插件类型 (transaction)
      - warnings: 警告列表
    """
    print("\n" + "=" * 70)
    print("示例 5: 响应 metadata 检查 (路由追踪)")
    print("=" * 70)

    # 验证所有 resume_screening 的 Agent 路由
    agents = ["intent_classifier", "resume_parser", "skill_matcher", "compliance_checker"]

    print("\n[路由方案验证] template=resume_screening")
    print("-" * 50)
    print(f"  {'Agent':<22} {'分配模型':<25} {'警告'}")
    print("-" * 50)

    for agent_name in agents:
        result = client.call_agent(
            template="resume_screening",
            agent=agent_name,
            messages=[{"role": "user", "content": "ping"}],
        )
        warnings_str = ", ".join(result.warnings) if result.warnings else "无"
        print(f"  {agent_name:<22} {result.assigned_model:<25} {warnings_str}")

    print("-" * 50)
    print("\n✓ 路由追踪演示完成")
    print("  提示: 在生产环境中，应记录每次调用的 aegis_metadata 用于审计")


# ---------------------------------------------------------------------------
# 示例 6: httpx 直接调用（不依赖 OpenAI SDK）
# ---------------------------------------------------------------------------


def demo_httpx_direct(client: AegisClient) -> None:
    """演示使用 httpx 直接调用 AegisRouter（不依赖 OpenAI SDK）。

    适用于不使用 OpenAI SDK 的自定义框架。
    """
    print("\n" + "=" * 70)
    print("示例 6: httpx 直接调用")
    print("=" * 70)

    if client.dry_run:
        # Dry-run 模式下展示请求结构
        payload = {
            "model": "placeholder",
            "messages": [{"role": "user", "content": "分析代码质量"}],
            "metadata": {
                "transaction": {
                    "template": "code_review",
                    "agent": "code_analyzer",
                }
            },
        }
        print("\n[DRY-RUN] 请求体结构:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        # 模拟响应
        mock_resp = {
            "choices": [{"message": {"role": "assistant", "content": "代码分析结果..."}}],
            "model": "claude-sonnet-4-20250514",
            "aegis_metadata": {
                "template": "code_review",
                "agent": "code_analyzer",
                "assigned_model": "claude-sonnet-4-20250514",
                "routing_plugin": "transaction",
                "warnings": [],
            },
        }
        print("\n[DRY-RUN] 响应体结构:")
        print(json.dumps(mock_resp, indent=2, ensure_ascii=False))
    else:
        try:
            import httpx
        except ImportError:
            print("  跳过: httpx 未安装 (pip install httpx)")
            return

        url = f"{AEGIS_BASE_URL}/chat/completions"
        payload = {
            "model": "placeholder",
            "messages": [{"role": "user", "content": "分析代码质量"}],
            "metadata": {
                "transaction": {
                    "template": "code_review",
                    "agent": "code_analyzer",
                }
            },
        }

        print(f"\n[POST] {url}")
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {AEGIS_API_KEY}"},
            json=payload,
            timeout=60.0,
        )

        if resp.status_code == 200:
            data = resp.json()
            aegis_meta = data.get("aegis_metadata", {})
            print(f"  → HTTP {resp.status_code}")
            print(f"  → 分配模型: {aegis_meta.get('assigned_model', 'N/A')}")
            print(f"  → 内容: {data['choices'][0]['message']['content'][:80]}...")
        else:
            print(f"  → HTTP {resp.status_code}: {resp.text[:200]}")

    print("\n✓ httpx 直接调用演示完成")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AegisRouter Supervisor 集成示例脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python examples/supervisor_integration_example.py --dry-run
  python examples/supervisor_integration_example.py
  python examples/supervisor_integration_example.py --demo streaming
  python examples/supervisor_integration_example.py --base-url http://my-aegis:8000/v1
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="使用模拟响应运行（不需要 AegisRouter 实例）",
    )
    parser.add_argument(
        "--base-url",
        default=AEGIS_BASE_URL,
        help=f"AegisRouter base URL（默认: {AEGIS_BASE_URL}）",
    )
    parser.add_argument(
        "--api-key",
        default=AEGIS_API_KEY,
        help="API Key",
    )
    parser.add_argument(
        "--demo",
        choices=["all", "pipeline", "multi-template", "streaming", "errors", "metadata", "httpx"],
        default="all",
        help="选择要运行的演示（默认: all）",
    )

    args = parser.parse_args()

    # 使用命令行参数覆盖配置
    base_url = args.base_url
    api_key = args.api_key

    # 创建客户端
    client = AegisClient(dry_run=args.dry_run)

    mode_label = "DRY-RUN (模拟模式)" if args.dry_run else f"LIVE ({base_url})"
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        AegisRouter Supervisor 集成示例                              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n运行模式: {mode_label}")
    print(f"API Key:  {api_key[:10]}...")

    # 运行选定的演示
    demos = {
        "pipeline": demo_single_template_pipeline,
        "multi-template": demo_multi_template_routing,
        "streaming": demo_streaming,
        "errors": demo_error_handling,
        "metadata": demo_metadata_inspection,
        "httpx": demo_httpx_direct,
    }

    if args.demo == "all":
        for demo_fn in demos.values():
            demo_fn(client)
    else:
        demos[args.demo](client)

    print("\n" + "=" * 70)
    print("全部演示完成！")
    print("=" * 70)
    print("\n提示:")
    print("  - 实际使用时将 AegisRouter 启动后运行本脚本（去掉 --dry-run）")
    print("  - 配置模板: config/transaction_templates.yaml")
    print("  - 能力 Profile: config/capability_profiles.yaml")
    print("  - 集成指南: docs/supervisor_integration_guide.md")


if __name__ == "__main__":
    main()
