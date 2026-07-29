"""Tests for TransactionRouterCallback — 事务级路由回调。

Tests cover:
- TC-TXN-ROUTE-001: 正确 template + agent → 查表命中，路由到预计算模型
- TC-TXN-ROUTE-002: 无 transaction metadata → fallback 模型
- TC-TXN-ROUTE-003: 未知模板 → raises Exception with "not found" in message
- TC-TXN-ROUTE-004: 未知 Agent → fallback + UNKNOWN_AGENT 警告
- TC-TXN-ROUTE-005: PII 管道正常工作（基类处理）
- TC-TXN-ROUTE-006: 流式响应 + 事务路由同时工作
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.router.routing_plan_store import RoutingPlanStore


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def plan_store():
    """Create a RoutingPlanStore with test data."""
    store = RoutingPlanStore()
    store.set_model("resume_screening", "resume_parser", "gemini-2.5-pro")
    store.set_model("resume_screening", "intent_classifier", "local-7b")
    store.set_model("resume_screening", "skill_matcher", "gpt-5.5")
    store.set_model("code_review", "code_analyzer", "codex-mini")
    store.set_model("code_review", "issue_detector", "gpt-5.5")
    return store


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool that simulates successful PII masking."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.max_connections = 10

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            return {
                "masked_text": params["text"],
                "entities_found": [],
            }
        elif method == "restore":
            return {"restored_text": params["text"]}
        elif method == "get_mapping":
            return {"mapping": {}}
        return None

    pool.call = AsyncMock(side_effect=mock_call)
    return pool


@pytest.fixture
def router(plan_store, mock_pool):
    """Create a TransactionRouterCallback with test store and mock pool."""
    return TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        pool=mock_pool,
    )


# ---------------------------------------------------------------------------
# Test: Normal routing (TC-TXN-ROUTE-001)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_routing_correct_template_agent(router):
    """正确 template + agent → 查表命中，路由到预计算模型."""
    data = {
        "messages": [{"role": "user", "content": "Parse this resume."}],
        "metadata": {
            "transaction": {
                "template": "resume_screening",
                "agent": "resume_parser",
            }
        },
    }

    await router.async_pre_call_hook({}, None, data, "completion")

    assert data["model"] == "gemini-2.5-pro"
    assert data["metadata"]["target_model"] == "gemini-2.5-pro"
    assert data["metadata"]["route_reason"] == "plan"
    assert data["metadata"]["transaction_template"] == "resume_screening"
    assert data["metadata"]["transaction_agent"] == "resume_parser"


@pytest.mark.asyncio
async def test_normal_routing_different_template(router):
    """不同模板+agent → 路由到对应模型."""
    data = {
        "messages": [{"role": "user", "content": "Review this code."}],
        "metadata": {
            "transaction": {
                "template": "code_review",
                "agent": "code_analyzer",
            }
        },
    }

    await router.async_pre_call_hook({}, None, data, "completion")

    assert data["model"] == "codex-mini"
    assert data["metadata"]["target_model"] == "codex-mini"
    assert data["metadata"]["route_reason"] == "plan"


# ---------------------------------------------------------------------------
# Test: No transaction metadata (TC-TXN-ROUTE-002)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_transaction_metadata_uses_fallback(router):
    """无 transaction metadata 的请求 → fallback 模型."""
    data = {
        "messages": [{"role": "user", "content": "Hello world."}],
        "metadata": {},
    }

    await router.async_pre_call_hook({}, None, data, "completion")

    assert data["model"] == "deepseek-v3"
    assert data["metadata"]["target_model"] == "deepseek-v3"
    assert data["metadata"]["route_reason"] == "fallback"


@pytest.mark.asyncio
async def test_no_metadata_at_all_uses_fallback(router):
    """完全没有 metadata 的请求 → fallback 模型."""
    data = {
        "messages": [{"role": "user", "content": "Hello world."}],
    }

    await router.async_pre_call_hook({}, None, data, "completion")

    assert data["model"] == "deepseek-v3"
    assert data["metadata"]["route_reason"] == "fallback"


# ---------------------------------------------------------------------------
# Test: Unknown template (TC-TXN-ROUTE-003)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_template_raises_exception(router):
    """引用不存在的模板 → 抛出 Exception，消息包含 'not found'."""
    data = {
        "messages": [{"role": "user", "content": "Test."}],
        "metadata": {
            "transaction": {
                "template": "nonexistent_template",
                "agent": "some_agent",
            }
        },
    }

    with pytest.raises(Exception, match="not found"):
        await router.async_pre_call_hook({}, None, data, "completion")


@pytest.mark.asyncio
async def test_unknown_template_error_contains_template_name(router):
    """错误消息包含模板名称."""
    data = {
        "messages": [{"role": "user", "content": "Test."}],
        "metadata": {
            "transaction": {
                "template": "my_missing_template",
                "agent": "some_agent",
            }
        },
    }

    with pytest.raises(Exception, match="my_missing_template"):
        await router.async_pre_call_hook({}, None, data, "completion")


# ---------------------------------------------------------------------------
# Test: V4-5 — Unknown template → HTTP 400 (TemplateNotFoundError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_5_unknown_template_returns_http_400(router):
    """V4-5: 引用不存在的模板 → HTTP 400.

    验证:
    1. 抛出 TemplateNotFoundError（非通用 Exception）
    2. 异常 status_code == 400（LiteLLM Proxy 映射为 HTTP 400）
    3. 异常 message 包含模板名和 "not found"
    4. 异常 type == "invalid_request_error"
    """
    from aegis_router.callbacks.exceptions import TemplateNotFoundError

    data = {
        "messages": [{"role": "user", "content": "Test."}],
        "metadata": {
            "transaction": {
                "template": "fantasy_template",
                "agent": "some_agent",
            }
        },
    }

    with pytest.raises(TemplateNotFoundError) as exc_info:
        await router.async_pre_call_hook({}, None, data, "completion")

    err = exc_info.value
    assert err.status_code == 400
    assert "fantasy_template" in err.message
    assert "not found" in err.message
    assert err.type == "invalid_request_error"


# ---------------------------------------------------------------------------
# Test: Unknown agent (TC-TXN-ROUTE-004)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_agent_uses_fallback(router):
    """模板存在但 Agent 不在其中 → fallback 模型."""
    data = {
        "messages": [{"role": "user", "content": "Test."}],
        "metadata": {
            "transaction": {
                "template": "resume_screening",
                "agent": "unknown_agent",
            }
        },
    }

    await router.async_pre_call_hook({}, None, data, "completion")

    assert data["model"] == "deepseek-v3"
    assert data["metadata"]["target_model"] == "deepseek-v3"
    assert data["metadata"]["route_reason"] == "unknown_agent"


@pytest.mark.asyncio
async def test_unknown_agent_logs_warning(router, caplog):
    """未知 Agent 时记录 UNKNOWN_AGENT 警告日志."""
    import logging

    data = {
        "messages": [{"role": "user", "content": "Test."}],
        "metadata": {
            "transaction": {
                "template": "resume_screening",
                "agent": "nonexistent_agent",
            }
        },
    }

    with caplog.at_level(logging.WARNING):
        await router.async_pre_call_hook({}, None, data, "completion")

    assert "UNKNOWN_AGENT" in caplog.text
    assert "nonexistent_agent" in caplog.text


# ---------------------------------------------------------------------------
# Test: PII pipeline works with transaction routing (TC-TXN-ROUTE-005)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pii_pipeline_still_works(plan_store):
    """公共管道（PII 脱敏）与事务路由同时工作."""
    mock_pool = MagicMock(spec=ClawVaultPool)
    mock_pool.max_connections = 10

    masked_content = "Hello [PII_PLACEHOLDER], please process."
    original_content = "Hello 张三, please process."

    call_count = {"mask": 0}

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            call_count["mask"] += 1
            return {
                "masked_text": masked_content,
                "entities_found": [{"type": "PERSON_NAME", "value": "张三"}],
            }
        elif method == "restore":
            return {"restored_text": original_content}
        elif method == "get_mapping":
            return {"mapping": {"[PII_PLACEHOLDER]": "张三"}}
        return None

    mock_pool.call = AsyncMock(side_effect=mock_call)

    # Patch redis health check to avoid actual Redis dependency
    with patch(
        "aegis_router.callbacks.degradation.DegradationManager.check_redis_health",
        new_callable=AsyncMock,
        return_value=MagicMock(value="healthy"),
    ):
        router = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        data = {
            "messages": [{"role": "user", "content": original_content}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

    # Verify PII masking was called
    assert call_count["mask"] > 0

    # Verify routing still happened correctly
    assert data["model"] == "gemini-2.5-pro"
    assert data["metadata"]["target_model"] == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Test: V4-3 Integration — intent_classifier routes to local-7b
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_3_intent_classifier_routes_to_local_7b(router):
    """V4-3: 请求 {"template": "resume_screening", "agent": "intent_classifier"} → 路由到 local-7b.

    验证 TransactionRouterCallback 通过 RoutingPlanStore 查表，
    将 lightweight profile 的 intent_classifier 正确路由到 local-7b。
    """
    data = {
        "messages": [{"role": "user", "content": "Classify user intent."}],
        "metadata": {
            "transaction": {
                "template": "resume_screening",
                "agent": "intent_classifier",
            }
        },
    }

    await router.async_pre_call_hook({}, None, data, "completion")

    assert data["model"] == "local-7b"
    assert data["metadata"]["target_model"] == "local-7b"
    assert data["metadata"]["route_reason"] == "plan"
    assert data["metadata"]["transaction_template"] == "resume_screening"
    assert data["metadata"]["transaction_agent"] == "intent_classifier"


# ---------------------------------------------------------------------------
# Test: Constructor defaults
# ---------------------------------------------------------------------------


def test_constructor_default_plan_store():
    """未提供 plan_store 时使用空表."""
    with patch.object(ClawVaultPool, "__init__", return_value=None):
        router = TransactionRouterCallback(fallback_model="test-model")
    assert len(router.plan_store) == 0


def test_constructor_accepts_config_dir():
    """接受 config_dir 参数（兼容 plugin_loader）."""
    with patch.object(ClawVaultPool, "__init__", return_value=None):
        router = TransactionRouterCallback(
            fallback_model="test-model",
            config_dir="./config",
        )
    assert router.fallback_model == "test-model"


def test_plan_store_setter(plan_store):
    """plan_store 属性可以被设置（支持热更新）."""
    with patch.object(ClawVaultPool, "__init__", return_value=None):
        router = TransactionRouterCallback(fallback_model="test-model")
    assert len(router.plan_store) == 0

    router.plan_store = plan_store
    assert len(router.plan_store) == 5


# ---------------------------------------------------------------------------
# Test: Empty messages bypass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_messages_bypass(router):
    """空消息列表 → 基类直接返回，不执行路由."""
    data = {
        "messages": [],
        "metadata": {
            "transaction": {
                "template": "resume_screening",
                "agent": "resume_parser",
            }
        },
    }

    await router.async_pre_call_hook({}, None, data, "completion")

    # No model should be set because pipeline was skipped
    assert "model" not in data


# ---------------------------------------------------------------------------
# Fixtures for Failover Tests (V4-9)
# ---------------------------------------------------------------------------


@pytest.fixture
def router_with_failover(plan_store, mock_pool):
    """Create a TransactionRouterCallback with failover chains configured."""
    return TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        failover_chains={
            "gpt-5.5": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
            "gemini-2.5-pro": ["gemini-3.1-pro", "claude-sonnet", "deepseek-v4-pro"],
        },
        failover_enabled=True,
        pool=mock_pool,
    )


# ---------------------------------------------------------------------------
# Test: V4-9 — Mock LLM 429 → failover 到链中下一个模型，全局方案不变
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_9_failover_selects_next_model_on_failure(router_with_failover):
    """V4-9: async_log_failure_event 选择 failover 链中的下一个模型。

    模拟 gpt-5.5 返回 429，验证 failover 选择链中第一个模型 gpt-5.2。
    """
    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
        "_failover_index": 0,
        "_original_model": "gpt-5.5",
    }
    kwargs = {"model": "gpt-5.5", "metadata": metadata}

    await router_with_failover.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert metadata["_failover_model"] == "gpt-5.2"
    assert metadata["_failover_from"] == "gpt-5.5"
    assert metadata["_failover_index"] == 1


@pytest.mark.asyncio
async def test_v4_9_failover_advances_chain_index(router_with_failover):
    """V4-9: 连续失败时 failover 沿链前进到下一个模型。

    第一次失败后 index=1，第二次失败后 index=2，选择 claude-sonnet。
    """
    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
        "_failover_index": 1,  # 第一个已经试过了
        "_original_model": "gpt-5.5",
    }
    kwargs = {"model": "gpt-5.2", "metadata": metadata}

    await router_with_failover.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    assert metadata["_failover_model"] == "claude-sonnet"
    assert metadata["_failover_from"] == "gpt-5.2"
    assert metadata["_failover_index"] == 2


@pytest.mark.asyncio
async def test_v4_9_failover_chain_exhausted(router_with_failover):
    """V4-9: failover 链耗尽时不再选择新模型。"""
    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
        "_failover_index": 3,  # 已耗尽（链长度为 3）
        "_original_model": "gpt-5.5",
    }
    kwargs = {"model": "deepseek-v4-pro", "metadata": metadata}

    await router_with_failover.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    # 链已耗尽，不应设置新的 failover 模型
    assert "_failover_model" not in metadata


@pytest.mark.asyncio
async def test_v4_9_failover_does_not_modify_global_plan_store(
    router_with_failover, plan_store
):
    """V4-9 (FR-6.3): failover 后全局 RoutingPlanStore 不变。

    验证 async_log_failure_event 仅影响当次请求的 metadata，
    不修改全局方案表中的任何条目。
    """
    # 记录 failover 前方案表的快照
    plans_before = plan_store.get_all_plans().copy()

    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
        "_failover_index": 0,
        "_original_model": "gpt-5.5",
    }
    kwargs = {"model": "gpt-5.5", "metadata": metadata}

    await router_with_failover.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    # 验证方案表未被修改
    plans_after = plan_store.get_all_plans()
    assert plans_before == plans_after

    # 验证具体条目仍然是原来的模型（不是 failover 模型）
    assert plan_store.get_model("resume_screening", "skill_matcher") == "gpt-5.5"
    assert plan_store.get_model("resume_screening", "resume_parser") == "gemini-2.5-pro"
    assert plan_store.get_model("code_review", "issue_detector") == "gpt-5.5"


@pytest.mark.asyncio
async def test_v4_9_failover_logs_agent_failover_warning(
    router_with_failover, caplog
):
    """V4-9: failover 时记录 AGENT_FAILOVER 警告日志。"""
    import logging

    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
        "_failover_index": 0,
        "_original_model": "gpt-5.5",
    }
    kwargs = {"model": "gpt-5.5", "metadata": metadata}

    with caplog.at_level(logging.WARNING):
        await router_with_failover.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    assert "AGENT_FAILOVER" in caplog.text
    assert "gpt-5.5" in caplog.text
    assert "gpt-5.2" in caplog.text


@pytest.mark.asyncio
async def test_v4_9_failover_without_metadata_chain_uses_config(
    router_with_failover,
):
    """V4-9: metadata 中无 _failover_chain 时，从配置中查找 failover 链。

    模拟首次失败（尚未执行过路由注入 failover 链信息的场景）。
    """
    metadata = {}
    kwargs = {"model": "gpt-5.5", "metadata": metadata}

    await router_with_failover.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    # 应从 failover_chains 配置中获取链并选择第一个
    assert metadata["_failover_model"] == "gpt-5.2"
    assert metadata["_failover_from"] == "gpt-5.5"
    assert metadata["_failover_index"] == 1


def test_v4_9_get_next_failover_model_returns_correct_model(
    router_with_failover,
):
    """V4-9: get_next_failover_model 返回链中下一个模型。"""
    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
        "_failover_index": 0,
    }

    next_model = router_with_failover.get_next_failover_model(
        "gpt-5.5", metadata=metadata
    )
    assert next_model == "gpt-5.2"


def test_v4_9_get_next_failover_model_chain_exhausted(router_with_failover):
    """V4-9: get_next_failover_model 链耗尽时返回 None。"""
    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet", "deepseek-v4-pro"],
        "_failover_index": 3,
    }

    next_model = router_with_failover.get_next_failover_model(
        "deepseek-v4-pro", metadata=metadata
    )
    assert next_model is None


def test_v4_9_get_next_failover_model_no_metadata(router_with_failover):
    """V4-9: get_next_failover_model 无 metadata 时从配置中查找。"""
    next_model = router_with_failover.get_next_failover_model(
        "gpt-5.5", metadata=None
    )
    assert next_model == "gpt-5.2"


def test_v4_9_get_next_failover_model_unknown_model(router_with_failover):
    """V4-9: get_next_failover_model 未知模型无 failover 链返回 None。"""
    next_model = router_with_failover.get_next_failover_model(
        "unknown-model", metadata=None
    )
    assert next_model is None


@pytest.mark.asyncio
async def test_v4_9_failover_disabled_does_nothing(plan_store, mock_pool):
    """V4-9: failover_enabled=False 时不执行 failover。"""
    import logging

    router = TransactionRouterCallback(
        plan_store=plan_store,
        fallback_model="deepseek-v3",
        failover_chains={"gpt-5.5": ["gpt-5.2", "claude-sonnet"]},
        failover_enabled=False,
        pool=mock_pool,
    )

    metadata = {
        "_failover_chain": ["gpt-5.2", "claude-sonnet"],
        "_failover_index": 0,
        "_original_model": "gpt-5.5",
    }
    kwargs = {"model": "gpt-5.5", "metadata": metadata}

    await router.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    # failover 被禁用，不应设置 failover 模型
    assert "_failover_model" not in metadata


# ---------------------------------------------------------------------------
# Test: Streaming + Transaction Routing (TC-TXN-ROUTE-006)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_with_transaction_routing(plan_store):
    """TC-TXN-ROUTE-006: 流式响应 + 事务路由同时工作。

    验证:
    1. async_pre_call_hook 正确完成事务路由（设置目标模型）
    2. async_post_call_streaming_iterator_hook 正确还原流式 chunk 中的 PII 占位符
    3. 事务路由 metadata 在流式处理期间保持完整
    """
    mock_pool = MagicMock(spec=ClawVaultPool)
    mock_pool.max_connections = 10

    # PII mapping: placeholder → original value
    pii_mapping = {
        "[PERSON_1]": "张三",
        "[PHONE_1]": "13800138000",
    }

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            # Simulate masking: replace PII with placeholders
            return {
                "masked_text": "你好 [PERSON_1]，你的电话是 [PHONE_1]",
                "entities_found": [
                    {"type": "PERSON_NAME", "value": "张三"},
                    {"type": "PHONE", "value": "13800138000"},
                ],
            }
        elif method == "restore":
            return {"restored_text": params["text"]}
        elif method == "get_mapping":
            return {"mapping": pii_mapping}
        return None

    mock_pool.call = AsyncMock(side_effect=mock_call)

    # Patch redis health check
    with patch(
        "aegis_router.callbacks.degradation.DegradationManager.check_redis_health",
        new_callable=AsyncMock,
        return_value=MagicMock(value="healthy"),
    ):
        router = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        # --- Step 1: Route the request via async_pre_call_hook ---
        data = {
            "messages": [
                {"role": "user", "content": "你好 张三，你的电话是 13800138000"}
            ],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "resume_parser",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        # Verify routing happened correctly
        assert data["model"] == "gemini-2.5-pro"
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["transaction_template"] == "resume_screening"
        assert data["metadata"]["transaction_agent"] == "resume_parser"

        # --- Step 2: Simulate streaming response with PII placeholders ---
        # Simulate chunks that contain PII placeholders (as the LLM would produce)
        streaming_chunks = [
            {"choices": [{"delta": {"content": "收到，"}}]},
            {"choices": [{"delta": {"content": "[PERSON_1]"}}]},
            {"choices": [{"delta": {"content": " 的电话号码是 "}}]},
            {"choices": [{"delta": {"content": "[PHONE_1]"}}]},
            {"choices": [{"delta": {"content": "，已确认。"}}]},
        ]

        async def async_chunk_generator():
            for chunk in streaming_chunks:
                yield chunk

        # --- Step 3: Process streaming chunks through the hook ---
        restored_chunks = []
        async for chunk in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=async_chunk_generator(),
            request_data=data,
        ):
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                restored_chunks.append(content)

        # --- Step 4: Verify PII placeholders were restored ---
        full_restored_text = "".join(restored_chunks)
        assert "张三" in full_restored_text
        assert "13800138000" in full_restored_text
        assert "[PERSON_1]" not in full_restored_text
        assert "[PHONE_1]" not in full_restored_text

        # --- Step 5: Verify transaction routing metadata is preserved ---
        assert data["metadata"]["target_model"] == "gemini-2.5-pro"
        assert data["metadata"]["route_reason"] == "plan"
        assert data["metadata"]["transaction_template"] == "resume_screening"
        assert data["metadata"]["transaction_agent"] == "resume_parser"
        assert data["metadata"]["routing_plugin"] == "transaction"


@pytest.mark.asyncio
async def test_streaming_with_transaction_routing_cross_chunk_placeholder(plan_store):
    """TC-TXN-ROUTE-006 补充: 占位符跨 chunk 分割时流式还原仍正确。

    模拟占位符被分割在两个 chunk 中的情况:
    - chunk1: "你好 [PER"
    - chunk2: "SON_1]，欢迎"
    验证 StreamRehydrator 的缓冲机制在事务路由场景下正常工作。
    """
    mock_pool = MagicMock(spec=ClawVaultPool)
    mock_pool.max_connections = 10

    pii_mapping = {"[PERSON_1]": "张三"}

    async def mock_call(method, params):
        if method == "check_compliance":
            return {"passed": True}
        elif method == "mask":
            return {
                "masked_text": "你好 [PERSON_1]",
                "entities_found": [{"type": "PERSON_NAME", "value": "张三"}],
            }
        elif method == "get_mapping":
            return {"mapping": pii_mapping}
        return None

    mock_pool.call = AsyncMock(side_effect=mock_call)

    with patch(
        "aegis_router.callbacks.degradation.DegradationManager.check_redis_health",
        new_callable=AsyncMock,
        return_value=MagicMock(value="healthy"),
    ):
        router = TransactionRouterCallback(
            plan_store=plan_store,
            fallback_model="deepseek-v3",
            pool=mock_pool,
        )

        data = {
            "messages": [{"role": "user", "content": "你好 张三"}],
            "metadata": {
                "transaction": {
                    "template": "resume_screening",
                    "agent": "skill_matcher",
                }
            },
        }

        await router.async_pre_call_hook({}, None, data, "completion")

        # Verify routing
        assert data["model"] == "gpt-5.5"

        # Simulate cross-chunk placeholder split
        streaming_chunks = [
            {"choices": [{"delta": {"content": "你好 [PER"}}]},
            {"choices": [{"delta": {"content": "SON_1]，欢迎"}}]},
        ]

        async def async_chunk_generator():
            for chunk in streaming_chunks:
                yield chunk

        restored_chunks = []
        async for chunk in router.async_post_call_streaming_iterator_hook(
            user_api_key_dict={},
            response=async_chunk_generator(),
            request_data=data,
        ):
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content:
                restored_chunks.append(content)

        full_restored_text = "".join(restored_chunks)
        assert "张三" in full_restored_text
        assert "[PERSON_1]" not in full_restored_text
