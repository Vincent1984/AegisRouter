"""Tests for AgentWorkbuddyCallback — Agent-WorkBuddy 路由回调。

Tests cover:
- V2-1: 单条 user 消息正确提取 agent 字段
- V2-2: 多条 user 消息取最后一条的 agent
- V2-3: user 消息无 agent 时使用 metadata.agent 备选
- V2-4: 均无 agent → fallback + NO_AGENT 警告
- V2-5: 非法 agent 名称 → fallback + INVALID_AGENT 警告
- V2-6: 未知 agent → fallback + UNKNOWN_AGENT 警告
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from aegis_router.callbacks.agent_workbuddy_router import AgentWorkbuddyCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.agent_plan_store import AgentPlanStore


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_store():
    """Create an AgentPlanStore with test data."""
    store = AgentPlanStore()
    store.set_model("intent_classifier", "deepseek-v4-pro")
    store.set_model("document_parser", "gpt-5.5")
    store.set_model("reasoning_engine", "gpt-5.5")
    store.set_model("code_assistant", "codex-mini")
    return store


@pytest.fixture
def router(plan_store):
    """Create an AgentWorkbuddyCallback with test store."""
    with patch.object(ClawVaultPool, "__init__", return_value=None):
        return AgentWorkbuddyCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
        )


# ---------------------------------------------------------------------------
# Test: V2-1 — 单条 user 消息正确提取 agent 字段
# ---------------------------------------------------------------------------


class TestExtractAgentSingleUserMessage:
    """V2-1: 单条 user 消息正确提取 agent 字段。

    验证 _extract_agent 方法从单条 role: "user" 消息中提取 agent 字段。
    需求: FR-1.1
    """

    def test_single_user_message_extracts_agent(self, router):
        """单条 user 消息含 agent 字段 → 正确返回 agent 名称。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我分类这个意图",
                    "agent": "intent_classifier",
                }
            ]
        }

        result = router._extract_agent(data)

        assert result == "intent_classifier"

    def test_single_user_message_different_agent(self, router):
        """单条 user 消息含不同 agent 名称 → 正确返回对应名称。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Parse this document.",
                    "agent": "document_parser",
                }
            ]
        }

        result = router._extract_agent(data)

        assert result == "document_parser"

    def test_single_user_message_with_system_message(self, router):
        """system + 单条 user 消息 → 从 user 消息中提取 agent。"""
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个意图分类助手",
                },
                {
                    "role": "user",
                    "content": "帮我分析代码",
                    "agent": "code_assistant",
                },
            ]
        }

        result = router._extract_agent(data)

        assert result == "code_assistant"

    def test_single_user_message_with_hyphenated_agent(self, router):
        """agent 名称含连字符 → 正确提取。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test request",
                    "agent": "my-agent-v2",
                }
            ]
        }

        result = router._extract_agent(data)

        assert result == "my-agent-v2"

    def test_single_user_message_with_underscore_agent(self, router):
        """agent 名称含下划线 → 正确提取。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test request",
                    "agent": "reasoning_engine",
                }
            ]
        }

        result = router._extract_agent(data)

        assert result == "reasoning_engine"


# ---------------------------------------------------------------------------
# Test: V2-2 — 多条 user 消息取最后一条的 agent
# ---------------------------------------------------------------------------


class TestExtractAgentMultipleUserMessages:
    """V2-2: 多条 user 消息取最后一条的 agent。

    验证 _extract_agent 方法在多条 role: "user" 消息存在时，
    仅使用最后一条 user 消息的 agent 字段。
    需求: FR-1.2
    """

    def test_multiple_user_messages_returns_last_agent(self, router):
        """TC-WB-002: 多条 user 消息，最后一条有 agent → 返回最后一条的 agent。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "第一条消息",
                    "agent": "intent_classifier",
                },
                {
                    "role": "user",
                    "content": "第二条消息",
                    "agent": "document_parser",
                },
            ]
        }

        result = router._extract_agent(data)

        assert result == "document_parser"

    def test_multiple_user_messages_different_agents_returns_last(self, router):
        """多条 user 消息含不同 agent → 返回最后一条的 agent。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "早期消息",
                    "agent": "code_assistant",
                },
                {
                    "role": "user",
                    "content": "中间消息",
                    "agent": "intent_classifier",
                },
                {
                    "role": "user",
                    "content": "最后消息",
                    "agent": "reasoning_engine",
                },
            ]
        }

        result = router._extract_agent(data)

        assert result == "reasoning_engine"

    def test_last_user_message_no_agent_earlier_has_agent_returns_none(self, router):
        """最后一条 user 消息无 agent 但早期有 → 返回 None（降级到 metadata.agent）。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "早期消息有 agent",
                    "agent": "intent_classifier",
                },
                {
                    "role": "user",
                    "content": "最后消息无 agent",
                },
            ]
        }

        result = router._extract_agent(data)

        # 最后一条 user 消息无 agent，且无 metadata → 返回 None
        assert result is None

    def test_last_user_message_no_agent_falls_through_to_metadata(self, router):
        """最后一条 user 消息无 agent → 降级到 metadata.agent。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "早期消息有 agent",
                    "agent": "intent_classifier",
                },
                {
                    "role": "user",
                    "content": "最后消息无 agent",
                },
            ],
            "metadata": {"agent": "fallback_agent"},
        }

        result = router._extract_agent(data)

        assert result == "fallback_agent"

    def test_multiple_user_messages_interleaved_with_assistant(self, router):
        """user 消息与 assistant 消息交错 → 仍取最后一条 user 消息的 agent。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "第一轮用户消息",
                    "agent": "intent_classifier",
                },
                {
                    "role": "assistant",
                    "content": "第一轮助手回复",
                },
                {
                    "role": "user",
                    "content": "第二轮用户消息",
                    "agent": "document_parser",
                },
                {
                    "role": "assistant",
                    "content": "第二轮助手回复",
                },
                {
                    "role": "user",
                    "content": "第三轮用户消息",
                    "agent": "code_assistant",
                },
            ]
        }

        result = router._extract_agent(data)

        assert result == "code_assistant"

    def test_three_user_messages_always_picks_last(self, router):
        """三条以上 user 消息 → 始终取最后一条。"""
        data = {
            "messages": [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "msg1", "agent": "intent_classifier"},
                {"role": "assistant", "content": "reply1"},
                {"role": "user", "content": "msg2", "agent": "document_parser"},
                {"role": "assistant", "content": "reply2"},
                {"role": "user", "content": "msg3", "agent": "reasoning_engine"},
                {"role": "assistant", "content": "reply3"},
                {"role": "user", "content": "msg4", "agent": "code_assistant"},
            ]
        }

        result = router._extract_agent(data)

        assert result == "code_assistant"


# ---------------------------------------------------------------------------
# Test: V2-3 — user 消息无 agent 时使用 metadata.agent 备选
# ---------------------------------------------------------------------------


class TestExtractAgentMetadataFallback:
    """V2-3: user 消息无 agent 时使用 metadata.agent 备选。

    验证 _extract_agent 方法在 user 消息中无 agent 字段时，
    正确降级到 metadata.agent 作为兼容备选。
    需求: FR-1.3

    **Validates: Requirements 1.3**
    """

    def test_single_user_no_agent_with_metadata_returns_metadata_agent(self, router):
        """单条 user 消息无 agent 字段 + metadata.agent 存在 → 返回 metadata.agent。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理这个任务",
                }
            ],
            "metadata": {"agent": "metadata_fallback_agent"},
        }

        result = router._extract_agent(data)

        assert result == "metadata_fallback_agent"

    def test_no_user_messages_with_metadata_returns_metadata_agent(self, router):
        """无 user 消息（仅 system/assistant）+ metadata.agent 存在 → 返回 metadata.agent。"""
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个助手",
                },
                {
                    "role": "assistant",
                    "content": "好的，有什么可以帮您的？",
                },
            ],
            "metadata": {"agent": "system_only_agent"},
        }

        result = router._extract_agent(data)

        assert result == "system_only_agent"

    def test_user_message_empty_string_agent_with_metadata_returns_metadata_agent(
        self, router
    ):
        """user 消息 agent 为空字符串 + metadata.agent 存在 → 返回 metadata.agent。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "空 agent 字段测试",
                    "agent": "",
                }
            ],
            "metadata": {"agent": "empty_string_fallback"},
        }

        result = router._extract_agent(data)

        assert result == "empty_string_fallback"

    def test_user_message_no_agent_no_metadata_returns_none(self, router):
        """user 消息无 agent 字段 + 无 metadata → 返回 None。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "没有 agent 也没有 metadata",
                }
            ],
        }

        result = router._extract_agent(data)

        assert result is None

    def test_multiple_user_messages_all_no_agent_with_metadata_returns_metadata_agent(
        self, router
    ):
        """多条 user 消息均无 agent 字段 + metadata.agent 存在 → 返回 metadata.agent。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "第一条消息无 agent",
                },
                {
                    "role": "assistant",
                    "content": "助手回复",
                },
                {
                    "role": "user",
                    "content": "第二条消息也无 agent",
                },
            ],
            "metadata": {"agent": "multi_msg_fallback"},
        }

        result = router._extract_agent(data)

        assert result == "multi_msg_fallback"


# ---------------------------------------------------------------------------
# Test: V2-4 — 均无 agent → fallback + NO_AGENT 警告
# ---------------------------------------------------------------------------


class TestExecuteRoutingNoAgent:
    """V2-4: 均无 agent → fallback + NO_AGENT 警告。

    验证 _execute_routing 方法在请求中完全没有 agent 标识时，
    正确设置 fallback 模型并发出 NO_AGENT 警告。
    需求: FR-1.4

    **Validates: Requirements 1.4**
    """

    async def test_user_message_without_agent_no_metadata_routes_to_fallback(
        self, router
    ):
        """user 消息存在但无 agent 字段且无 metadata.agent → fallback + NO_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理这个任务",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["NO_AGENT"]
        assert data["metadata"]["transaction_template"] == ""
        assert data["metadata"]["transaction_agent"] == ""

    async def test_no_user_messages_at_all_routes_to_fallback(self, router):
        """无 user 消息（仅 system/assistant）→ fallback + NO_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个助手",
                },
                {
                    "role": "assistant",
                    "content": "好的，有什么可以帮您的？",
                },
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["NO_AGENT"]
        assert data["metadata"]["transaction_template"] == ""
        assert data["metadata"]["transaction_agent"] == ""

    async def test_empty_messages_list_routes_to_fallback(self, router):
        """空 messages 列表 → fallback + NO_AGENT。"""
        data = {
            "messages": [],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["NO_AGENT"]
        assert data["metadata"]["transaction_template"] == ""
        assert data["metadata"]["transaction_agent"] == ""

    async def test_no_metadata_key_in_data_routes_to_fallback(self, router):
        """data 中无 metadata 键 → fallback + NO_AGENT，且自动创建 metadata。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "没有 metadata 的请求",
                }
            ],
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert "metadata" in data
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["NO_AGENT"]
        assert data["metadata"]["transaction_template"] == ""
        assert data["metadata"]["transaction_agent"] == ""

    async def test_no_agent_logs_warning(self, router, caplog):
        """无 agent 时记录 NO_AGENT 警告日志。"""
        import logging

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "普通请求",
                }
            ],
            "metadata": {},
        }

        with caplog.at_level(logging.WARNING):
            await router._execute_routing(data, "masked", "original", "abc123hash")

        assert any("NO_AGENT" in record.message for record in caplog.records)
        assert any("fallback" in record.message.lower() for record in caplog.records)

    async def test_user_message_with_empty_string_agent_no_metadata_routes_to_fallback(
        self, router
    ):
        """user 消息 agent 为空字符串且无 metadata.agent → fallback + NO_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "空 agent 字符串",
                    "agent": "",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["NO_AGENT"]


# ---------------------------------------------------------------------------
# Test: V2-5 — 非法 agent 名称 → fallback + INVALID_AGENT 警告
# ---------------------------------------------------------------------------


class TestExecuteRoutingInvalidAgent:
    """V2-5: 非法 agent 名称 → fallback + INVALID_AGENT 警告。

    验证 _execute_routing 方法在 agent 名称包含非法字符时，
    正确设置 fallback 模型并发出 INVALID_AGENT 警告。
    合法字符范围: [a-zA-Z0-9_-]
    需求: FR-1.5, FR-1.6

    **Validates: Requirements 1.5, 1.6**
    """

    # --- 非法名称测试 ---

    async def test_agent_name_with_spaces_routes_to_fallback(self, router):
        """agent 名称含空格 → fallback + INVALID_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "my agent",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["INVALID_AGENT"]
        assert data["metadata"]["transaction_agent"] == "my agent"

    async def test_agent_name_with_at_sign_routes_to_fallback(self, router):
        """agent 名称含 @ 符号 → fallback + INVALID_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "agent@name",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["INVALID_AGENT"]
        assert data["metadata"]["transaction_agent"] == "agent@name"

    async def test_agent_name_with_dot_routes_to_fallback(self, router):
        """agent 名称含点号 → fallback + INVALID_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "agent.name",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["INVALID_AGENT"]
        assert data["metadata"]["transaction_agent"] == "agent.name"

    async def test_agent_name_with_slash_routes_to_fallback(self, router):
        """agent 名称含斜杠 → fallback + INVALID_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "agent/name",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["INVALID_AGENT"]
        assert data["metadata"]["transaction_agent"] == "agent/name"

    async def test_agent_name_with_chinese_characters_routes_to_fallback(self, router):
        """agent 名称含中文字符 → fallback + INVALID_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "代理",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["INVALID_AGENT"]
        assert data["metadata"]["transaction_agent"] == "代理"

    async def test_agent_name_empty_string_via_metadata_routes_to_fallback(
        self, router
    ):
        """agent 名称为空字符串(通过 metadata 传入) → 无法匹配正则 → fallback + INVALID_AGENT。

        注意: 空字符串不匹配 ^[a-zA-Z0-9_-]+$ (因为 + 要求至少一个字符)。
        """
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "系统消息",
                }
            ],
            "metadata": {"agent": " "},  # 仅空格，非法
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["target_model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"
        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"
        assert data["metadata"]["_routing_warnings"] == ["INVALID_AGENT"]

    async def test_invalid_agent_logs_warning(self, router, caplog):
        """非法 agent 名称时记录 INVALID_AGENT 警告日志。"""
        import logging

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "agent@inject",
                }
            ],
            "metadata": {},
        }

        with caplog.at_level(logging.WARNING):
            await router._execute_routing(data, "masked", "original", "abc123hash")

        assert any("INVALID_AGENT" in record.message for record in caplog.records)
        assert any("非法字符" in record.message for record in caplog.records)

    # --- 合法名称正向控制测试 ---

    async def test_valid_agent_name_with_hyphen_does_not_trigger_invalid(self, router):
        """合法 agent 名称含连字符 → 不触发 INVALID_AGENT（正向控制）。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "my-agent",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        # 名称合法但不在方案表中 → UNKNOWN_AGENT，非 INVALID_AGENT
        assert data["metadata"].get("_routing_warnings") != ["INVALID_AGENT"]

    async def test_valid_agent_name_with_underscore_does_not_trigger_invalid(
        self, router
    ):
        """合法 agent 名称含下划线 → 不触发 INVALID_AGENT（正向控制）。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "my_agent",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"].get("_routing_warnings") != ["INVALID_AGENT"]

    async def test_valid_agent_name_with_numbers_does_not_trigger_invalid(self, router):
        """合法 agent 名称含数字 → 不触发 INVALID_AGENT（正向控制）。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "agent123",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"].get("_routing_warnings") != ["INVALID_AGENT"]

    async def test_valid_agent_in_plan_store_routes_normally(self, router):
        """合法 agent 名称且在方案表中 → 正常路由，无 INVALID_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理任务",
                    "agent": "intent_classifier",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v4-pro"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["_routing_warnings"] == []



# ---------------------------------------------------------------------------
# Test: V2-6 — 未知 agent → fallback + UNKNOWN_AGENT 警告
# ---------------------------------------------------------------------------


class TestExecuteRoutingUnknownAgent:
    """V2-6: 未知 agent → fallback + UNKNOWN_AGENT 警告。

    验证 _execute_routing 方法在 agent 名称合法但不在 AgentPlanStore 中时，
    正确设置 fallback 模型并发出 UNKNOWN_AGENT 警告。
    需求: FR-3.2

    **Validates: Requirements 3.2**
    """

    async def test_unknown_agent_routes_to_fallback(self, router):
        """合法但未注册的 agent 名称 → fallback 模型。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理任务",
                    "agent": "unknown_agent_xyz",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v3"

    async def test_unknown_agent_metadata_contains_unknown_agent_warning(self, router):
        """未知 agent → metadata._routing_warnings 包含 UNKNOWN_AGENT。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理任务",
                    "agent": "nonexistent_agent",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["_routing_warnings"] == ["UNKNOWN_AGENT"]

    async def test_unknown_agent_metadata_route_reason(self, router):
        """未知 agent → metadata.route_reason 为 'unknown_agent'。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理任务",
                    "agent": "unknown_agent_xyz",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["route_reason"] == "unknown_agent"

    async def test_unknown_agent_metadata_routing_plugin(self, router):
        """未知 agent → metadata.routing_plugin 为 'agent_workbuddy'。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理任务",
                    "agent": "unknown_agent_xyz",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"

    async def test_unknown_agent_logs_warning(self, router, caplog):
        """未知 agent → 记录包含 UNKNOWN_AGENT 的 WARNING 日志。"""
        import logging

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理任务",
                    "agent": "unknown_agent_xyz",
                }
            ],
            "metadata": {},
        }

        with caplog.at_level(logging.WARNING):
            await router._execute_routing(data, "masked", "original", "abc123hash")

        assert any("UNKNOWN_AGENT" in record.message for record in caplog.records)

    async def test_multiple_unknown_agents_all_route_to_fallback(self, router):
        """多个不同的未知 agent 名称都路由到 fallback 模型。"""
        unknown_agents = [
            "ghost_agent",
            "missing_worker",
            "unregistered_bot",
            "phantom_v2",
        ]

        for agent_name in unknown_agents:
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"请求来自 {agent_name}",
                        "agent": agent_name,
                    }
                ],
                "metadata": {},
            }

            await router._execute_routing(data, "masked", "original", "abc123hash")

            assert data["model"] == "deepseek-v3", (
                f"agent '{agent_name}' should route to fallback"
            )
            assert data["metadata"]["_routing_warnings"] == ["UNKNOWN_AGENT"], (
                f"agent '{agent_name}' should have UNKNOWN_AGENT warning"
            )
            assert data["metadata"]["route_reason"] == "unknown_agent", (
                f"agent '{agent_name}' should have route_reason='unknown_agent'"
            )

    async def test_unknown_agent_transaction_agent_field_set(self, router):
        """未知 agent → metadata.transaction_agent 设为该 agent 名称。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我处理任务",
                    "agent": "unknown_agent_xyz",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["transaction_agent"] == "unknown_agent_xyz"


# ---------------------------------------------------------------------------
# Test: V2-7 — 已知 agent → 正确路由到方案表中的模型
# ---------------------------------------------------------------------------


class TestExecuteRoutingKnownAgent:
    """V2-7: 已知 agent → 正确路由到方案表中的模型。

    验证 _execute_routing 方法在 agent 名称合法且存在于 AgentPlanStore 中时，
    正确设置 data["model"] 为方案表中预计算的模型，并正确填充 metadata 各字段。
    需求: FR-3.1, FR-3.3, FR-5.1

    **Validates: Requirements 3.1, 3.3, 5.1**
    """

    async def test_known_agent_routes_to_correct_model(self, router):
        """intent_classifier → deepseek-v4-pro。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请帮我分类意图",
                    "agent": "intent_classifier",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "deepseek-v4-pro"

    async def test_known_agent_document_parser_routes_correctly(self, router):
        """document_parser → gpt-5.5。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "解析这个文档",
                    "agent": "document_parser",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "gpt-5.5"

    async def test_known_agent_code_assistant_routes_correctly(self, router):
        """code_assistant → codex-mini。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "帮我写一段代码",
                    "agent": "code_assistant",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["model"] == "codex-mini"

    async def test_known_agent_sets_route_reason_plan(self, router):
        """已知 agent → metadata["route_reason"] == "plan"。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理",
                    "agent": "intent_classifier",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["route_reason"] == "plan"

    async def test_known_agent_sets_routing_plugin(self, router):
        """已知 agent → metadata["routing_plugin"] == "agent_workbuddy"。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理",
                    "agent": "document_parser",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["routing_plugin"] == "agent_workbuddy"

    async def test_known_agent_no_warnings(self, router):
        """已知 agent → metadata["_routing_warnings"] == []。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理",
                    "agent": "reasoning_engine",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["_routing_warnings"] == []

    async def test_known_agent_sets_target_model(self, router):
        """已知 agent → metadata["target_model"] == 分配的模型。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理",
                    "agent": "code_assistant",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["target_model"] == "codex-mini"

    async def test_known_agent_sets_transaction_agent(self, router):
        """已知 agent → metadata["transaction_agent"] == agent 名称。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理",
                    "agent": "document_parser",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["transaction_agent"] == "document_parser"

    async def test_known_agent_sets_transaction_template_empty(self, router):
        """已知 agent → metadata["transaction_template"] == ""。"""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请处理",
                    "agent": "intent_classifier",
                }
            ],
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["metadata"]["transaction_template"] == ""

    async def test_known_agent_with_failover_chain_stores_chain_in_metadata(
        self, plan_store
    ):
        """已知 agent + failover_enabled + 模型有链 → metadata 中存储 failover 链信息。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router_with_failover = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请分类意图",
                    "agent": "intent_classifier",
                }
            ],
            "metadata": {},
        }

        await router_with_failover._execute_routing(
            data, "masked", "original", "abc123hash"
        )

        assert data["model"] == "deepseek-v4-pro"
        assert data["metadata"]["_failover_chain"] == ["gpt-5.5", "gpt-4o"]
        assert data["metadata"]["_failover_index"] == 0
        assert data["metadata"]["_original_model"] == "deepseek-v4-pro"

    async def test_known_agent_without_failover_chain_no_failover_metadata(
        self, plan_store
    ):
        """已知 agent + failover_enabled 但模型无链 → metadata 中无 _failover_chain。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router_with_failover = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "帮我写代码",
                    "agent": "code_assistant",  # codex-mini 不在 failover_chains 中
                }
            ],
            "metadata": {},
        }

        await router_with_failover._execute_routing(
            data, "masked", "original", "abc123hash"
        )

        assert data["model"] == "codex-mini"
        assert "_failover_chain" not in data["metadata"]
        assert "_failover_index" not in data["metadata"]
        assert "_original_model" not in data["metadata"]

    async def test_known_agent_preserves_original_messages(self, router):
        """已知 agent 路由后 data["messages"] 保持不变。"""
        original_messages = [
            {
                "role": "system",
                "content": "你是一个助手",
            },
            {
                "role": "user",
                "content": "请帮我分析代码",
                "agent": "code_assistant",
            },
        ]
        data = {
            "messages": list(original_messages),  # 浅拷贝
            "metadata": {},
        }

        await router._execute_routing(data, "masked", "original", "abc123hash")

        assert data["messages"] == original_messages


# ---------------------------------------------------------------------------
# Test: V2-9a — TC-WB-009: failover 链在 LLM 错误时触发
# ---------------------------------------------------------------------------


class TestFailoverChainTriggersOnLLMError:
    """TC-WB-009: failover 链在 LLM 错误时触发。

    验证 async_log_failure_event 在 LLM 调用失败时:
    1. 正确从 metadata 中读取 failover 链状态
    2. 选择链中下一个模型
    3. 递增 _failover_index
    4. 存储 _failover_model 和 _failover_from

    需求: FR-5.1, FR-5.2

    **Validates: Requirements 5.1, 5.2**
    """

    async def test_first_failure_selects_first_failover_model(self, plan_store):
        """首次 LLM 失败 → 选择 failover 链中第一个模型。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert metadata["_failover_model"] == "gpt-5.5"
        assert metadata["_failover_from"] == "deepseek-v4-pro"
        assert metadata["_failover_index"] == 1

    async def test_second_failure_selects_second_failover_model(self, plan_store):
        """第二次 LLM 失败 → 选择 failover 链中第二个模型。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 1,  # 第一个已尝试
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "gpt-5.5", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert metadata["_failover_model"] == "gpt-4o"
        assert metadata["_failover_from"] == "gpt-5.5"
        assert metadata["_failover_index"] == 2

    async def test_chain_exhausted_does_not_select_model(self, plan_store):
        """failover 链耗尽 → 不选择新模型，不写入 _failover_model。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 2,  # 已超出链长度
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "gpt-4o", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # 链耗尽，不应该写入新的 failover_model
        assert "_failover_model" not in metadata
        assert "_failover_from" not in metadata
        # index 保持不变
        assert metadata["_failover_index"] == 2

    async def test_no_metadata_failover_chain_uses_config_chains(self, plan_store):
        """metadata 中无 _failover_chain → 从实例配置 failover_chains 查找。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        # metadata 中没有 _failover_chain 字段
        metadata = {}
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # 应该从实例配置找到链并选择第一个
        assert metadata["_failover_model"] == "gpt-5.5"
        assert metadata["_failover_from"] == "deepseek-v4-pro"
        assert metadata["_failover_index"] == 1

    async def test_model_not_in_any_chain_no_failover(self, plan_store):
        """失败模型不在任何 failover 链中 → 不做 failover。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        metadata = {}
        kwargs = {"model": "codex-mini", "metadata": metadata}  # codex-mini 不在链中

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert "_failover_model" not in metadata
        assert "_failover_from" not in metadata

    async def test_failover_logs_warning(self, plan_store, caplog):
        """failover 触发时记录 AGENT_FAILOVER 警告日志。"""
        import logging

        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        with caplog.at_level(logging.WARNING):
            await router.async_log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=None,
                end_time=None,
            )

        assert any("AGENT_FAILOVER" in record.message for record in caplog.records)

    async def test_metadata_from_litellm_params_fallback(self, plan_store):
        """kwargs 中无直接 metadata 时从 litellm_params.metadata 获取。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        inner_metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {
            "model": "deepseek-v4-pro",
            "litellm_params": {"metadata": inner_metadata},
        }

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert inner_metadata["_failover_model"] == "gpt-5.5"
        assert inner_metadata["_failover_index"] == 1


# ---------------------------------------------------------------------------
# Test: V2-9b — TC-WB-010: failover 不修改全局方案
# ---------------------------------------------------------------------------


class TestFailoverDoesNotModifyGlobalPlanStore:
    """TC-WB-010: failover 不修改全局方案。

    验证 failover 过程中:
    1. AgentPlanStore 的内容在 failover 前后完全一致
    2. failover_chains 配置不被修改
    3. 仅 metadata 中的副本被更新，全局状态不变

    需求: FR-5.2

    **Validates: Requirements 5.2**
    """

    async def test_plan_store_unchanged_after_failover(self, plan_store):
        """failover 触发后 AgentPlanStore 内容完全不变。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        # 记录 failover 前的方案
        plans_before = router.plan_store.get_all_plans()

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # failover 后方案表应完全一致
        plans_after = router.plan_store.get_all_plans()
        assert plans_before == plans_after

    async def test_plan_store_unchanged_after_multiple_failovers(self, plan_store):
        """多次 failover 触发后 AgentPlanStore 内容仍然不变。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        plans_before = router.plan_store.get_all_plans()

        # 模拟连续 failover（链中两个模型依次失败）
        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }

        # 第一次 failover
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}
        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # 第二次 failover
        kwargs = {"model": "gpt-5.5", "metadata": metadata}
        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # 方案表仍然不变
        plans_after = router.plan_store.get_all_plans()
        assert plans_before == plans_after

    async def test_plan_store_unchanged_after_chain_exhaustion(self, plan_store):
        """failover 链耗尽后 AgentPlanStore 内容仍然不变。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        plans_before = router.plan_store.get_all_plans()

        # 链已耗尽
        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 2,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "gpt-4o", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        plans_after = router.plan_store.get_all_plans()
        assert plans_before == plans_after

    async def test_failover_chains_config_unchanged_after_failover(self, plan_store):
        """failover 触发后实例的 failover_chains 配置不被修改。"""
        original_chains = {"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]}
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains=original_chains,
                failover_enabled=True,
            )

        # 记录 failover_chains 的深拷贝
        import copy
        chains_before = copy.deepcopy(router.failover_chains)

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # 实例的 failover_chains 配置不应被修改
        assert router.failover_chains == chains_before

    async def test_original_model_still_in_plan_store_after_failover(self, plan_store):
        """failover 后原始模型对应的 agent 在方案表中仍映射到原模型。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # intent_classifier 仍然映射到原始模型 deepseek-v4-pro
        assert router.plan_store.get_model("intent_classifier") == "deepseek-v4-pro"

    async def test_concurrent_requests_failover_isolation(self, plan_store):
        """并发请求的 failover 互不影响（metadata 隔离验证）。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=True,
            )

        # 模拟两个并发请求各自的 metadata
        metadata_req1 = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        metadata_req2 = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }

        # 请求1 failover
        kwargs1 = {"model": "deepseek-v4-pro", "metadata": metadata_req1}
        await router.async_log_failure_event(
            kwargs=kwargs1, response_obj=None, start_time=None, end_time=None
        )

        # 请求2 的 metadata 不受请求1的影响
        assert metadata_req2["_failover_index"] == 0
        assert "_failover_model" not in metadata_req2

        # 请求2 failover
        kwargs2 = {"model": "deepseek-v4-pro", "metadata": metadata_req2}
        await router.async_log_failure_event(
            kwargs=kwargs2, response_obj=None, start_time=None, end_time=None
        )

        # 两个请求各自独立 failover
        assert metadata_req1["_failover_model"] == "gpt-5.5"
        assert metadata_req2["_failover_model"] == "gpt-5.5"
        assert metadata_req1["_failover_index"] == 1
        assert metadata_req2["_failover_index"] == 1


# ---------------------------------------------------------------------------
# Test: V2-10 — failover_enabled=False 时不尝试替代模型
# ---------------------------------------------------------------------------


class TestFailoverDisabledNoFalloverAttempt:
    """V2-10: failover_enabled=False 时不尝试替代模型。

    验证当 failover_enabled=False 时:
    1. _execute_routing 不在 metadata 中存储 failover 链信息
    2. async_log_failure_event 不选择备选模型（早返回）
    3. get_next_failover_model 返回 None

    需求: FR-5.3

    **Validates: Requirements 5.3**
    """

    async def test_failover_disabled_no_chain_stored_in_metadata(self, plan_store):
        """V2-10: failover_enabled=False 时，即使模型有链配置，metadata 中不存储 failover 信息。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router_no_failover = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=False,
            )

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请分类意图",
                    "agent": "intent_classifier",
                }
            ],
            "metadata": {},
        }

        await router_no_failover._execute_routing(
            data, "masked", "original", "abc123hash"
        )

        # Model should still be correctly routed
        assert data["model"] == "deepseek-v4-pro"
        # But no failover metadata should be stored
        assert "_failover_chain" not in data["metadata"]
        assert "_failover_index" not in data["metadata"]
        assert "_original_model" not in data["metadata"]

    async def test_failover_disabled_failure_event_does_not_select_next_model(
        self, plan_store
    ):
        """V2-10: failover_enabled=False 时 async_log_failure_event 不选择备选模型。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router_no_failover = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=False,
            )

        metadata = {
            "_failover_chain": ["gpt-5.5", "gpt-4o"],
            "_failover_index": 0,
            "_original_model": "deepseek-v4-pro",
        }
        kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

        await router_no_failover.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        # failover_index should NOT be incremented (early return)
        assert metadata["_failover_index"] == 0
        assert "_failover_model" not in metadata
        assert "_failover_from" not in metadata

    async def test_failover_disabled_get_next_failover_model_returns_none(
        self, plan_store
    ):
        """V2-10: failover_enabled=False 时 get_next_failover_model 返回 None。"""
        with patch.object(ClawVaultPool, "__init__", return_value=None):
            router_no_failover = AgentWorkbuddyCallback(
                plan_store=plan_store,
                fallback_model="deepseek-v3",
                failover_chains={"deepseek-v4-pro": ["gpt-5.5", "gpt-4o"]},
                failover_enabled=False,
            )

        result = router_no_failover.get_next_failover_model(
            failed_model="deepseek-v4-pro",
            metadata={
                "_failover_chain": ["gpt-5.5", "gpt-4o"],
                "_failover_index": 0,
            },
        )
        assert result is None
