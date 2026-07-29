"""Tests for Plugin Switch (Phase 6.5 — 插件互切测试)

验证:
- TC-SWITCH-001: conversation → transaction 切换后事务路由生效
- TC-SWITCH-002: transaction → conversation 切换后对话级路由恢复
- TC-SWITCH-003: 切换过程中进行中的请求不异常中断
- TC-SWITCH-004: 事务级插件下，metadata.transaction 被正确处理
- TC-SWITCH-005: 对话级插件下，metadata.transaction 被忽略
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from aegis_router.callbacks.base_router import BaseRouterCallback
from aegis_router.callbacks.plugin_loader import load_routing_plugin
from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_temp_config(config_data: dict) -> str:
    """Create a temp directory with a config.yaml containing given data.

    Returns the temp directory path.
    """
    tmpdir = tempfile.mkdtemp()
    config_path = Path(tmpdir) / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    return tmpdir


def _make_mock_pool() -> MagicMock:
    """Create a mock ClawVaultPool that simulates successful compliance + masking."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.max_connections = 10

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            return {
                "masked_text": params.get("text", ""),
                "entities_found": [],
            }
        elif method == "restore":
            return {"restored_text": params.get("text", "")}
        elif method == "get_mapping":
            return {"mapping": {}}
        return None

    pool.call = AsyncMock(side_effect=mock_call)
    return pool


def _make_plan_store() -> RoutingPlanStore:
    """Create a RoutingPlanStore with test data."""
    store = RoutingPlanStore()
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("resume_screening", "skill_matcher", "gpt-5.5")
    store.set_model("code_review", "code_analyzer", "codex-mini")
    return store


# ---------------------------------------------------------------------------
# TC-SWITCH-001: conversation → transaction 切换后事务路由生效
# ---------------------------------------------------------------------------


class TestSwitchConversationToTransaction:
    """TC-SWITCH-001: 从 conversation 切换到 transaction 后事务路由生效."""

    def test_load_conversation_then_transaction(self):
        """先加载 conversation 插件，再加载 transaction 插件，类型正确切换."""
        # Load conversation plugin
        tmpdir_conv = _create_temp_config({"routing_plugin": "conversation"})
        plugin_conv = load_routing_plugin(config_dir=tmpdir_conv, enable_routing=False)
        assert isinstance(plugin_conv, SmartRouterCallback)
        assert isinstance(plugin_conv, BaseRouterCallback)

        # Load transaction plugin
        tmpdir_txn = _create_temp_config({"routing_plugin": "transaction"})
        plugin_txn = load_routing_plugin(config_dir=tmpdir_txn)
        assert isinstance(plugin_txn, TransactionRouterCallback)
        assert isinstance(plugin_txn, BaseRouterCallback)

    @pytest.mark.asyncio
    async def test_transaction_routing_works_after_switch(self):
        """切换到 transaction 后，事务路由能正确根据 plan_store 分发."""
        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        # Simulate: first had conversation plugin
        conv_plugin = SmartRouterCallback(pool=mock_pool, enable_routing=False)
        assert isinstance(conv_plugin, SmartRouterCallback)

        # Now switch to transaction plugin
        txn_plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        data = {
            "messages": [{"role": "user", "content": "Parse this resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        await txn_plugin.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "gemini-2.5-pro"
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["routing_plugin"] == "transaction"


# ---------------------------------------------------------------------------
# TC-SWITCH-002: transaction → conversation 切换后对话级路由恢复
# ---------------------------------------------------------------------------


class TestSwitchTransactionToConversation:
    """TC-SWITCH-002: 从 transaction 切换到 conversation 后对话级路由恢复."""

    def test_load_transaction_then_conversation(self):
        """先加载 transaction 插件，再加载 conversation 插件，类型正确切换."""
        # Load transaction plugin
        tmpdir_txn = _create_temp_config({"routing_plugin": "transaction"})
        plugin_txn = load_routing_plugin(config_dir=tmpdir_txn)
        assert isinstance(plugin_txn, TransactionRouterCallback)

        # Load conversation plugin
        tmpdir_conv = _create_temp_config({"routing_plugin": "conversation"})
        plugin_conv = load_routing_plugin(config_dir=tmpdir_conv, enable_routing=False)
        assert isinstance(plugin_conv, SmartRouterCallback)
        assert isinstance(plugin_conv, BaseRouterCallback)

    @pytest.mark.asyncio
    async def test_conversation_routing_works_after_switch(self):
        """切换回 conversation 后，对话级路由插件正确工作（使用 fallback/评分）."""
        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        # Simulate: first had transaction plugin
        txn_plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )
        assert isinstance(txn_plugin, TransactionRouterCallback)

        # Now switch to conversation plugin (routing disabled for test simplicity)
        conv_plugin = SmartRouterCallback(pool=mock_pool, enable_routing=False)
        assert isinstance(conv_plugin, SmartRouterCallback)
        assert isinstance(conv_plugin, BaseRouterCallback)

        # Verify that conversation plugin doesn't do transaction-style routing
        data = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        # With routing disabled, conversation plugin won't set model
        await conv_plugin.async_pre_call_hook({}, None, data, "completion")

        # The key point: conversation plugin does NOT do table lookup
        # With routing disabled, no model is set based on transaction metadata
        assert data.get("model") != "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# TC-SWITCH-003: 切换过程中进行中的请求不异常中断
# ---------------------------------------------------------------------------


class TestSwitchInFlightRequests:
    """TC-SWITCH-003: 切换过程中进行中的请求不异常中断."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_during_plugin_switch(self):
        """模拟并发请求：切换插件引用时已发出的请求不崩溃."""
        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        # Start with transaction plugin
        active_plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        results = []
        errors = []

        async def make_request(plugin, request_id: int):
            """Simulate a request using the given plugin instance."""
            try:
                data = {
                    "messages": [
                        {"role": "user", "content": f"Request {request_id}"}
                    ],
                    "metadata": {
                        "transaction": {
                            "template": "resume_screening",
                            "agent": "resume_parser",
                        }
                    },
                }
                await plugin.async_pre_call_hook({}, None, data, "completion")
                results.append(
                    {"id": request_id, "model": data.get("model"), "success": True}
                )
            except Exception as e:
                errors.append({"id": request_id, "error": str(e)})

        # Fire multiple concurrent requests
        tasks = []
        for i in range(5):
            tasks.append(asyncio.create_task(make_request(active_plugin, i)))

        # Simulate plugin switch mid-flight by replacing the reference
        new_plugin = SmartRouterCallback(pool=mock_pool, enable_routing=False)

        # The old plugin reference is still being used by in-flight tasks
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Key assertion: no errors occurred during concurrent execution
        assert len(errors) == 0
        assert len(results) == 5

        # All in-flight requests completed successfully with original plugin
        for result in results:
            assert result["success"] is True
            assert result["model"] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_mixed_plugin_requests_no_crash(self):
        """两个插件实例同时处理请求，互不干扰."""
        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        txn_plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )
        conv_plugin = SmartRouterCallback(pool=mock_pool, enable_routing=False)

        results = []
        errors = []

        async def txn_request(idx):
            try:
                data = {
                    "messages": [{"role": "user", "content": f"TXN request {idx}"}],
                    "metadata": {
                        "transaction": {
                            "template": "code_review",
                            "agent": "code_analyzer",
                        }
                    },
                }
                await txn_plugin.async_pre_call_hook({}, None, data, "completion")
                results.append({"type": "txn", "model": data.get("model")})
            except Exception as e:
                errors.append({"type": "txn", "error": str(e)})

        async def conv_request(idx):
            try:
                data = {
                    "messages": [{"role": "user", "content": f"CONV request {idx}"}],
                    "metadata": {},
                }
                await conv_plugin.async_pre_call_hook({}, None, data, "completion")
                results.append({"type": "conv", "success": True})
            except Exception as e:
                errors.append({"type": "conv", "error": str(e)})

        # Mix transaction and conversation requests concurrently
        tasks = []
        for i in range(3):
            tasks.append(asyncio.create_task(txn_request(i)))
            tasks.append(asyncio.create_task(conv_request(i)))

        await asyncio.gather(*tasks)

        # No errors
        assert len(errors) == 0
        assert len(results) == 6

        # Transaction requests got correct routing
        txn_results = [r for r in results if r["type"] == "txn"]
        for r in txn_results:
            assert r["model"] == "codex-mini"


# ---------------------------------------------------------------------------
# TC-SWITCH-004: 事务级插件下，metadata.transaction 被正确处理
# ---------------------------------------------------------------------------


class TestTransactionMetadataHandling:
    """TC-SWITCH-004: 事务级插件下，metadata.transaction 被正确处理."""

    @pytest.mark.asyncio
    async def test_transaction_metadata_routes_by_plan(self):
        """事务级插件读取 metadata.transaction 进行查表路由."""
        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        data = {
            "messages": [{"role": "user", "content": "Analyze this code."}],
            "metadata": {
                "transaction": {
                    "template": "code_review",
                    "agent": "code_analyzer",
                }
            },
        }

        await plugin.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "codex-mini"
        assert data["metadata"]["target_model"] == "codex-mini"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["transaction_template"] == "code_review"
        assert data["metadata"]["transaction_agent"] == "code_analyzer"
        assert data["metadata"]["routing_plugin"] == "transaction"

    @pytest.mark.asyncio
    async def test_transaction_metadata_different_agents_different_models(self):
        """同一模板下不同 Agent 路由到不同模型."""
        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        # Agent 1: resume_parser → gemini-2.5-pro
        data1 = {
            "messages": [{"role": "user", "content": "Parse resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }
        await plugin.async_pre_call_hook({}, None, data1, "completion")
        assert data1["model"] == "gemini-2.5-pro"

        # Agent 2: intent_classifier → local-7b
        data2 = {
            "messages": [{"role": "user", "content": "Classify intent."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "intent_classifier",
                }
            },
        }
        await plugin.async_pre_call_hook({}, None, data2, "completion")
        assert data2["model"] == "local-7b"

        # Agent 3: skill_matcher → gpt-5.5
        data3 = {
            "messages": [{"role": "user", "content": "Match skills."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                }
            },
        }
        await plugin.async_pre_call_hook({}, None, data3, "completion")
        assert data3["model"] == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_transaction_metadata_no_template_uses_fallback(self):
        """有 transaction metadata 但无匹配模板 → HTTP 400."""
        from aegis_router.callbacks.exceptions import TemplateNotFoundError

        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        data = {
            "messages": [{"role": "user", "content": "Test."}],
            "metadata": {
                "transaction": {
                    "template": "nonexistent_template",
                    "agent": "some_agent",
                }
            },
        }

        with pytest.raises(TemplateNotFoundError) as exc_info:
            await plugin.async_pre_call_hook({}, None, data, "completion")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_no_transaction_metadata_uses_fallback(self):
        """无 transaction metadata → fallback 模型."""
        mock_pool = _make_mock_pool()
        plan_store = _make_plan_store()

        plugin = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        data = {
            "messages": [{"role": "user", "content": "Hello."}],
            "metadata": {},
        }

        await plugin.async_pre_call_hook({}, None, data, "completion")

        assert data["model"] == "deepseek-v3"
        assert data["metadata"]["route_reason"] == "fallback"


# ---------------------------------------------------------------------------
# TC-SWITCH-005: 对话级插件下，metadata.transaction 被忽略
# ---------------------------------------------------------------------------


class TestConversationIgnoresTransactionMetadata:
    """TC-SWITCH-005: 对话级插件下，metadata.transaction 被忽略."""

    @pytest.mark.asyncio
    async def test_conversation_plugin_ignores_transaction_metadata(self):
        """SmartRouterCallback 不读取 metadata.transaction，使用自身路由逻辑.

        当 routing 禁用时，不设置 model（忽略 transaction metadata）。
        当 routing 启用时，使用 RouteLLM 打分/规则引擎，而非查表。
        """
        mock_pool = _make_mock_pool()

        # SmartRouterCallback with routing disabled
        conv_plugin = SmartRouterCallback(pool=mock_pool, enable_routing=False)

        data = {
            "messages": [{"role": "user", "content": "Parse this resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        await conv_plugin.async_pre_call_hook({}, None, data, "completion")

        # Conversation plugin does NOT do table lookup based on transaction metadata
        # With routing disabled, no model is set
        assert data.get("model") != "gemini-2.5-pro"
        # The transaction metadata is still present but unused
        assert data["metadata"]["transaction"]["template"] == "resume_screening"
        assert data["metadata"]["transaction"]["agent"] == "resume_parser"

    @pytest.mark.asyncio
    async def test_conversation_plugin_routes_by_scoring_not_table(self):
        """对话级插件使用评分/规则路由，不使用事务查表。

        验证即使有 transaction metadata，SmartRouterCallback 也不会
        根据 (template, agent) 查表路由。
        """
        mock_pool = _make_mock_pool()

        # SmartRouterCallback with routing enabled but no classifier
        # → will use fallback model (no_classifier reason)
        conv_plugin = SmartRouterCallback(
            pool=mock_pool,
            enable_routing=True,
            rule_engine=MagicMock(check=MagicMock(return_value=MagicMock(matched=False))),
            classifier=None,
        )
        # Set a known fallback
        conv_plugin._routing_config = MagicMock()
        conv_plugin._routing_config.fallback_model = "deepseek-v3"
        conv_plugin._routing_config.score_input = "masked"
        conv_plugin._routing_config.session_policy = "sticky"

        data = {
            "messages": [{"role": "user", "content": "Parse this resume."}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        await conv_plugin.async_pre_call_hook({}, None, data, "completion")

        # Should route to fallback via conversation logic (no_classifier), NOT via plan table
        assert data["model"] == "deepseek-v3"
        # metadata.routing_plugin should be "conversation", not "transaction"
        assert data["metadata"]["routing_plugin"] == "conversation"
        # The transaction metadata is ignored — no transaction-specific routing fields
        assert data["metadata"].get("transaction_template") is None
        assert data["metadata"].get("transaction_agent") is None

    @pytest.mark.asyncio
    async def test_both_plugins_inherit_base_router(self):
        """两个插件都继承 BaseRouterCallback，公共管道一致."""
        assert issubclass(SmartRouterCallback, BaseRouterCallback)
        assert issubclass(TransactionRouterCallback, BaseRouterCallback)

        mock_pool = _make_mock_pool()

        conv = SmartRouterCallback(pool=mock_pool, enable_routing=False)
        txn = TransactionRouterCallback(
            plan_store=_make_plan_store(),
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        # Both are instances of BaseRouterCallback
        assert isinstance(conv, BaseRouterCallback)
        assert isinstance(txn, BaseRouterCallback)

        # Both have the abstract method implemented
        assert hasattr(conv, "_execute_routing")
        assert hasattr(txn, "_execute_routing")
