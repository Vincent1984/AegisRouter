"""回归测试 — RequestLoggerCallback 不影响 PII 脱敏/还原流程。

Task 8.4: PII 脱敏回归
- 发送含中文 PII 的请求，确认响应中 PII 被正确还原
- 验证 RequestLoggerCallback 不干扰 BaseRouterCallback 的 mask/restore 管道
- 测试 RequestLoggerCallback 启用和禁用两种状态

测试策略:
  1. 构建 SmartRouterCallback + RequestLoggerCallback 的回调链
  2. 发送含中文 PII (人名、手机号、身份证号) 的请求
  3. 模拟完整 E2E: pre_call_hook (脱敏) → LLM 返回占位符 → log_success_event (还原)
  4. 验证 PII 脱敏后发送给 LLM
  5. 验证 PII 在响应中被正确还原
  6. 验证 RequestLoggerCallback 不修改数据
"""

from __future__ import annotations

import copy

import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.observability.request_logger import (
    RequestLoggerCallback,
    RequestLoggingConfig,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockMessage:
    """Mock LiteLLM message object."""
    content: str
    role: str = "assistant"


@dataclass
class MockChoice:
    """Mock LiteLLM choice object."""
    message: MockMessage
    index: int = 0


@dataclass
class MockResponse:
    """Mock LiteLLM ModelResponse object."""
    choices: list


def make_response(content: str) -> MockResponse:
    """Create a mock LiteLLM response with given content."""
    return MockResponse(choices=[MockChoice(message=MockMessage(content=content))])


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock()
    pool.max_connections = 10
    return pool


@pytest.fixture
def smart_callback(mock_pool):
    """Create a SmartRouterCallback with a mocked pool."""
    return SmartRouterCallback(pool=mock_pool)


@pytest.fixture
def request_logger_enabled():
    """Create an enabled RequestLoggerCallback (stdout to avoid file IO)."""
    config = RequestLoggingConfig(
        enabled=True,
        output="stdout",
        file_path="./logs/test_pii_regression.jsonl",
        max_message_length=4096,
        retention_days=7,
    )
    return RequestLoggerCallback(config=config)


@pytest.fixture
def request_logger_disabled():
    """Create a disabled RequestLoggerCallback."""
    config = RequestLoggingConfig(enabled=False)
    return RequestLoggerCallback(config=config)


# ---------------------------------------------------------------------------
# TC-PII-REGRESSION-001: 完整 E2E — 中文人名 + 手机号 + 身份证号
# ---------------------------------------------------------------------------


class TestPIIRegressionFullE2E:
    """验证完整 E2E 流程: 含中文 PII 的请求 → 脱敏 → LLM 返回占位符 → 还原。

    同时包含 SmartRouterCallback 和 RequestLoggerCallback。
    """

    @pytest.mark.asyncio
    async def test_full_e2e_chinese_pii_with_request_logger(
        self, smart_callback, request_logger_enabled, mock_pool
    ):
        """完整 E2E: 含人名+手机号+身份证号 → 脱敏 → 占位符响应 → 还原。"""
        # === Phase 1: pre_call_hook (脱敏阶段) ===
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫张伟，手机号13912345678，身份证110101199501011234，帮我查社保",
                },
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-pii-001",
                "request_id": "req-pii-001",
            },
        }

        # ClawVault mock: compliance → mask
        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "我叫[PERSON_1]，手机号[PHONE_1]，身份证[ID_CARD_1]，帮我查社保",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 8, "end": 19, "score": 0.99},
                    {"type": "ID_CARD", "start": 23, "end": 41, "score": 0.99},
                ],
            },
        ]

        # SmartRouterCallback 脱敏
        await smart_callback.async_pre_call_hook({}, None, request_data, "completion")

        # 验证消息已被脱敏
        masked_content = request_data["messages"][0]["content"]
        assert "[PERSON_1]" in masked_content
        assert "[PHONE_1]" in masked_content
        assert "[ID_CARD_1]" in masked_content
        assert "张伟" not in masked_content
        assert "13912345678" not in masked_content
        assert "110101199501011234" not in masked_content

        # RequestLoggerCallback 观察脱敏后的请求 (不修改)
        data_before_logger = copy.deepcopy(request_data)
        await request_logger_enabled.async_pre_call_hook(
            {}, None, request_data, "completion"
        )
        # 验证 RequestLogger 未修改数据
        assert request_data["messages"][0]["content"] == data_before_logger["messages"][0]["content"]
        assert request_data["model"] == data_before_logger["model"]

        # === Phase 2: Mock LLM 返回含占位符的响应 ===
        llm_response = make_response(
            "[PERSON_1] 您好！手机 [PHONE_1]、身份证 [ID_CARD_1] 对应的社保缴纳正常。"
        )

        # === Phase 3: log_success_event (还原阶段) ===
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "restored_text": "张伟 您好！手机 13912345678、身份证 110101199501011234 对应的社保缴纳正常。"
        }

        kwargs = {
            "metadata": request_data["metadata"],
            "model": "gpt-4o",
        }

        # SmartRouterCallback 还原 PII
        await smart_callback.async_log_success_event(kwargs, llm_response, None, None)

        # 验证客户端收到还原后的原始 PII
        client_text = llm_response.choices[0].message.content
        assert client_text == "张伟 您好！手机 13912345678、身份证 110101199501011234 对应的社保缴纳正常。"
        assert "[PERSON_1]" not in client_text
        assert "[PHONE_1]" not in client_text
        assert "[ID_CARD_1]" not in client_text
        assert "张伟" in client_text
        assert "13912345678" in client_text
        assert "110101199501011234" in client_text

        # RequestLoggerCallback 也运行 (仅记录，不修改还原结果)
        await request_logger_enabled.async_log_success_event(
            kwargs, llm_response, None, None
        )

        # 验证还原后文本未被 RequestLogger 修改
        assert llm_response.choices[0].message.content == client_text

    @pytest.mark.asyncio
    async def test_full_e2e_with_disabled_request_logger(
        self, smart_callback, request_logger_disabled, mock_pool
    ):
        """禁用 RequestLoggerCallback 时，PII 脱敏/还原流程正常。"""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "员工刘德华，电话18600001234，查询工资明细",
                },
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-pii-002",
                "request_id": "req-pii-002",
            },
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "员工[PERSON_1]，电话[PHONE_1]，查询工资明细",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 5, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 8, "end": 19, "score": 0.99},
                ],
            },
        ]

        # SmartRouterCallback 脱敏
        await smart_callback.async_pre_call_hook({}, None, request_data, "completion")

        # 禁用的 RequestLogger 不做任何事
        result = await request_logger_disabled.async_pre_call_hook(
            {}, None, request_data, "completion"
        )
        assert result is request_data

        # 验证脱敏成功
        assert "[PERSON_1]" in request_data["messages"][0]["content"]
        assert "[PHONE_1]" in request_data["messages"][0]["content"]
        assert "刘德华" not in request_data["messages"][0]["content"]

        # LLM 响应
        llm_response = make_response("[PERSON_1] 的工资明细已发送到 [PHONE_1]。")

        # 还原
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "restored_text": "刘德华 的工资明细已发送到 18600001234。"
        }

        kwargs = {"metadata": request_data["metadata"], "model": "gpt-4o"}
        await smart_callback.async_log_success_event(kwargs, llm_response, None, None)

        # 禁用的 RequestLogger 在 success event 中不做任何事
        await request_logger_disabled.async_log_success_event(
            kwargs, llm_response, None, None
        )

        # 验证还原正确
        assert llm_response.choices[0].message.content == "刘德华 的工资明细已发送到 18600001234。"


# ---------------------------------------------------------------------------
# TC-PII-REGRESSION-002: RequestLogger 接收到的是脱敏后内容
# ---------------------------------------------------------------------------


class TestRequestLoggerSeesOnlyMaskedContent:
    """验证 RequestLoggerCallback 只看到脱敏后的内容 (因为路由回调先执行脱敏)。"""

    @pytest.mark.asyncio
    async def test_logger_receives_masked_content(
        self, smart_callback, request_logger_enabled, mock_pool
    ):
        """RequestLogger 在 pre_call_hook 中看到的是已脱敏的消息。"""
        request_data = {
            "messages": [
                {"role": "user", "content": "联系人赵丽颖，手机13501234567"},
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-pii-003",
                "request_id": "req-pii-003",
            },
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "联系人[PERSON_1]，手机[PHONE_1]",
                "entities_found": [
                    {"type": "PERSON", "start": 3, "end": 6, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 9, "end": 20, "score": 0.99},
                ],
            },
        ]

        # SmartRouterCallback 先执行脱敏
        await smart_callback.async_pre_call_hook({}, None, request_data, "completion")

        # 此时 request_data 已被脱敏
        assert request_data["messages"][0]["content"] == "联系人[PERSON_1]，手机[PHONE_1]"

        # RequestLogger 之后运行 — 只能看到脱敏后的内容
        # 记录 logger 接收到的数据
        data_seen_by_logger = copy.deepcopy(request_data)
        await request_logger_enabled.async_pre_call_hook(
            {}, None, request_data, "completion"
        )

        # 验证 logger 看到的是脱敏后的内容
        assert "赵丽颖" not in data_seen_by_logger["messages"][0]["content"]
        assert "13501234567" not in data_seen_by_logger["messages"][0]["content"]
        assert "[PERSON_1]" in data_seen_by_logger["messages"][0]["content"]
        assert "[PHONE_1]" in data_seen_by_logger["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_logger_does_not_alter_masked_content(
        self, smart_callback, request_logger_enabled, mock_pool
    ):
        """RequestLogger 不修改已脱敏的消息内容。"""
        request_data = {
            "messages": [
                {"role": "user", "content": "身份证440101198801012345查询信用"},
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-pii-004",
                "request_id": "req-pii-004",
            },
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "身份证[ID_CARD_1]查询信用",
                "entities_found": [
                    {"type": "ID_CARD", "start": 3, "end": 21, "score": 0.99},
                ],
            },
        ]

        await smart_callback.async_pre_call_hook({}, None, request_data, "completion")

        # 记录脱敏后状态
        masked_state = request_data["messages"][0]["content"]

        # RequestLogger 处理
        await request_logger_enabled.async_pre_call_hook(
            {}, None, request_data, "completion"
        )

        # 内容不变
        assert request_data["messages"][0]["content"] == masked_state


# ---------------------------------------------------------------------------
# TC-PII-REGRESSION-003: 两个回调共同执行 success event — 还原不受影响
# ---------------------------------------------------------------------------


class TestBothCallbacksInSuccessEvent:
    """验证 SmartRouterCallback 和 RequestLoggerCallback 同时处理 success event，
    还原流程不受影响。
    """

    @pytest.mark.asyncio
    async def test_restore_works_with_both_callbacks(
        self, smart_callback, request_logger_enabled, mock_pool
    ):
        """SmartRouterCallback 执行还原后，RequestLoggerCallback 不影响结果。"""
        response_obj = make_response(
            "[PERSON_1] 的手机 [PHONE_1] 已通过验证。"
        )

        mock_pool.call.return_value = {
            "restored_text": "王芳 的手机 13700001111 已通过验证。"
        }

        kwargs = {
            "metadata": {
                "request_id": "req-pii-005",
                "session_id": "sess-pii-005",
            },
            "model": "gpt-4o",
        }

        # SmartRouterCallback 执行还原
        await smart_callback.async_log_success_event(kwargs, response_obj, None, None)
        assert response_obj.choices[0].message.content == "王芳 的手机 13700001111 已通过验证。"

        # RequestLoggerCallback 也执行（仅记录，不修改）
        await request_logger_enabled.async_log_success_event(
            kwargs, response_obj, None, None
        )

        # 验证还原结果未被修改
        assert response_obj.choices[0].message.content == "王芳 的手机 13700001111 已通过验证。"
        assert "[PERSON_1]" not in response_obj.choices[0].message.content
        assert "[PHONE_1]" not in response_obj.choices[0].message.content

    @pytest.mark.asyncio
    async def test_restore_with_disabled_logger(
        self, smart_callback, request_logger_disabled, mock_pool
    ):
        """禁用的 RequestLoggerCallback 在 success event 中零影响。"""
        response_obj = make_response("身份证 [ID_CARD_1] 对应的公积金状态正常。")

        mock_pool.call.return_value = {
            "restored_text": "身份证 320101199001015678 对应的公积金状态正常。"
        }

        kwargs = {
            "metadata": {
                "request_id": "req-pii-006",
                "session_id": "sess-pii-006",
            },
            "model": "gpt-4o",
        }

        # SmartRouterCallback 还原
        await smart_callback.async_log_success_event(kwargs, response_obj, None, None)

        # 禁用的 RequestLogger
        await request_logger_disabled.async_log_success_event(
            kwargs, response_obj, None, None
        )

        # 验证还原正确
        result = response_obj.choices[0].message.content
        assert result == "身份证 320101199001015678 对应的公积金状态正常。"
        assert "[ID_CARD_1]" not in result


# ---------------------------------------------------------------------------
# TC-PII-REGRESSION-004: 复杂请求 — 多种中文 PII 实体
# ---------------------------------------------------------------------------


class TestMultipleChinesePIIEntities:
    """测试包含多种中文 PII 实体的复杂请求场景。"""

    @pytest.mark.asyncio
    async def test_multiple_pii_types_e2e(
        self, smart_callback, request_logger_enabled, mock_pool
    ):
        """多种 PII (2人名+2手机+1身份证) 的完整 E2E 流程。"""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "申请人张三(13800001111)和担保人李四(13800002222)，"
                        "申请人身份证110101200001011234，请审批贷款"
                    ),
                },
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-pii-007",
                "request_id": "req-pii-007",
            },
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": (
                    "申请人[PERSON_1]([PHONE_1])和担保人[PERSON_2]([PHONE_2])，"
                    "申请人身份证[ID_CARD_1]，请审批贷款"
                ),
                "entities_found": [
                    {"type": "PERSON", "start": 3, "end": 5, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 6, "end": 17, "score": 0.99},
                    {"type": "PERSON", "start": 22, "end": 24, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 25, "end": 36, "score": 0.99},
                    {"type": "ID_CARD", "start": 43, "end": 61, "score": 0.99},
                ],
            },
        ]

        # pre_call_hook: SmartRouter 脱敏
        await smart_callback.async_pre_call_hook({}, None, request_data, "completion")

        masked = request_data["messages"][0]["content"]
        assert "[PERSON_1]" in masked
        assert "[PERSON_2]" in masked
        assert "[PHONE_1]" in masked
        assert "[PHONE_2]" in masked
        assert "[ID_CARD_1]" in masked
        assert "张三" not in masked
        assert "李四" not in masked
        assert "13800001111" not in masked
        assert "13800002222" not in masked
        assert "110101200001011234" not in masked

        # RequestLogger 观察脱敏后数据
        await request_logger_enabled.async_pre_call_hook(
            {}, None, request_data, "completion"
        )
        # 确认未被修改
        assert request_data["messages"][0]["content"] == masked

        # LLM 响应含所有占位符
        llm_response = make_response(
            "贷款审批: [PERSON_1](电话[PHONE_1])信用良好，"
            "[PERSON_2](电话[PHONE_2])担保有效，"
            "身份证[ID_CARD_1]已验证。批准贷款。"
        )

        # 还原
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "restored_text": (
                "贷款审批: 张三(电话13800001111)信用良好，"
                "李四(电话13800002222)担保有效，"
                "身份证110101200001011234已验证。批准贷款。"
            )
        }

        kwargs = {"metadata": request_data["metadata"], "model": "gpt-4o"}
        await smart_callback.async_log_success_event(kwargs, llm_response, None, None)

        # RequestLogger 也处理 success event
        await request_logger_enabled.async_log_success_event(
            kwargs, llm_response, None, None
        )

        # 验证所有 PII 均已还原
        result = llm_response.choices[0].message.content
        assert "张三" in result
        assert "李四" in result
        assert "13800001111" in result
        assert "13800002222" in result
        assert "110101200001011234" in result
        assert "[PERSON_1]" not in result
        assert "[PERSON_2]" not in result
        assert "[PHONE_1]" not in result
        assert "[PHONE_2]" not in result
        assert "[ID_CARD_1]" not in result

    @pytest.mark.asyncio
    async def test_multiple_pii_with_disabled_logger(
        self, smart_callback, request_logger_disabled, mock_pool
    ):
        """禁用 RequestLogger 时多种 PII 的 E2E 流程同样正常。"""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "通知王五(18611112222)和赵六(13700003333)参加会议",
                },
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-pii-008",
                "request_id": "req-pii-008",
            },
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "通知[PERSON_1]([PHONE_1])和[PERSON_2]([PHONE_2])参加会议",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 5, "end": 16, "score": 0.99},
                    {"type": "PERSON", "start": 18, "end": 20, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 21, "end": 32, "score": 0.99},
                ],
            },
        ]

        await smart_callback.async_pre_call_hook({}, None, request_data, "completion")
        await request_logger_disabled.async_pre_call_hook(
            {}, None, request_data, "completion"
        )

        # LLM 响应
        llm_response = make_response(
            "已通知 [PERSON_1] 和 [PERSON_2]，会议时间已确认。"
        )

        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "restored_text": "已通知 王五 和 赵六，会议时间已确认。"
        }

        kwargs = {"metadata": request_data["metadata"], "model": "gpt-4o"}
        await smart_callback.async_log_success_event(kwargs, llm_response, None, None)
        await request_logger_disabled.async_log_success_event(
            kwargs, llm_response, None, None
        )

        result = llm_response.choices[0].message.content
        assert result == "已通知 王五 和 赵六，会议时间已确认。"
        assert "[PERSON_1]" not in result
        assert "[PERSON_2]" not in result
