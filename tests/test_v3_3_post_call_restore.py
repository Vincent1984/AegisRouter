"""V3-3 验证检查点: Mock LLM 返回含占位符的响应 → post_call_hook 触发 → 客户端收到还原后的原文

验证 async_log_success_event (post_call_hook) 的完整流程:
- Mock LLM 返回含多种占位符类型的响应
- async_log_success_event 正确触发并携带 metadata (request_id, session_id)
- ClawVault restore 被调用并返回还原文本
- response_obj.choices[0].message.content 被原地更新为还原后的文本
- 客户端收到还原后的原始 PII 文本

测试场景:
1. 单个占位符还原
2. 多种不同类型占位符全部还原
3. 重复出现的同一占位符还原
4. 中文 PII 还原 (中文人名、手机号、身份证号)
5. 验证 response_obj 原地变更 (in-place mutation)
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool


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
def callback(mock_pool):
    """Create a SmartRouterCallback with a mocked pool."""
    return SmartRouterCallback(pool=mock_pool)


def make_kwargs(request_id: str = "req-v3-3", session_id: str = "sess-v3-3") -> dict:
    """Create kwargs dict with metadata as LiteLLM passes to async_log_success_event."""
    return {
        "metadata": {
            "request_id": request_id,
            "session_id": session_id,
        },
        "model": "gpt-4o",
    }


# ---------------------------------------------------------------------------
# V3-3 验证: 单个占位符还原
# ---------------------------------------------------------------------------


class TestV3_3_SinglePlaceholderRestore:
    """验证单个占位符被正确还原为原始值。"""

    async def test_single_person_placeholder_restored(self, callback, mock_pool):
        """单个 [PERSON_1] 占位符还原为原始人名。"""
        response_obj = make_response("你好 [PERSON_1]，你的订单已发出。")
        kwargs = make_kwargs()

        mock_pool.call.return_value = {
            "restored_text": "你好 张三，你的订单已发出。"
        }

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        # 验证客户端收到还原后的文本
        assert response_obj.choices[0].message.content == "你好 张三，你的订单已发出。"

    async def test_single_phone_placeholder_restored(self, callback, mock_pool):
        """单个 [PHONE_1] 占位符还原为原始电话号码。"""
        response_obj = make_response("请拨打 [PHONE_1] 联系客服。")
        kwargs = make_kwargs()

        mock_pool.call.return_value = {
            "restored_text": "请拨打 13800138000 联系客服。"
        }

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        assert response_obj.choices[0].message.content == "请拨打 13800138000 联系客服。"

    async def test_single_email_placeholder_restored(self, callback, mock_pool):
        """单个 [EMAIL_1] 占位符还原为原始邮箱地址。"""
        response_obj = make_response("确认邮件已发送至 [EMAIL_1]。")
        kwargs = make_kwargs()

        mock_pool.call.return_value = {
            "restored_text": "确认邮件已发送至 zhangsan@example.com。"
        }

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        assert response_obj.choices[0].message.content == "确认邮件已发送至 zhangsan@example.com。"


# ---------------------------------------------------------------------------
# V3-3 验证: 多种不同类型占位符全部还原
# ---------------------------------------------------------------------------


class TestV3_3_MultiplePlaceholderTypes:
    """验证多种不同类型的占位符在同一响应中全部被正确还原。"""

    async def test_person_phone_email_all_restored(self, callback, mock_pool):
        """[PERSON_1], [PHONE_1], [EMAIL_1] 三种类型全部还原。"""
        response_text = (
            "尊敬的 [PERSON_1]，您的手机号 [PHONE_1] 已绑定，"
            "验证邮件已发送至 [EMAIL_1]。"
        )
        response_obj = make_response(response_text)
        kwargs = make_kwargs(request_id="req-multi-type", session_id="sess-multi")

        expected_restored = (
            "尊敬的 李明，您的手机号 13912345678 已绑定，"
            "验证邮件已发送至 liming@company.cn。"
        )
        mock_pool.call.return_value = {"restored_text": expected_restored}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        # 验证所有占位符均已还原
        result = response_obj.choices[0].message.content
        assert result == expected_restored
        assert "[PERSON_1]" not in result
        assert "[PHONE_1]" not in result
        assert "[EMAIL_1]" not in result

    async def test_multiple_persons_and_phones(self, callback, mock_pool):
        """多个不同编号的 PERSON 和 PHONE 占位符全部还原。"""
        response_text = (
            "[PERSON_1] 的电话是 [PHONE_1]，"
            "[PERSON_2] 的电话是 [PHONE_2]。"
        )
        response_obj = make_response(response_text)
        kwargs = make_kwargs()

        expected_restored = (
            "张三 的电话是 13800001111，"
            "李四 的电话是 13800002222。"
        )
        mock_pool.call.return_value = {"restored_text": expected_restored}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        result = response_obj.choices[0].message.content
        assert result == expected_restored
        assert "[PERSON_1]" not in result
        assert "[PERSON_2]" not in result
        assert "[PHONE_1]" not in result
        assert "[PHONE_2]" not in result

    async def test_restore_called_with_all_metadata(self, callback, mock_pool):
        """验证 ClawVault restore 被调用时携带正确的 request_id 和 session_id。"""
        response_obj = make_response("[PERSON_1] 和 [PHONE_1]")
        kwargs = make_kwargs(request_id="req-meta-check", session_id="sess-meta-check")

        mock_pool.call.return_value = {"restored_text": "张三 和 13800138000"}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        # 验证 pool.call 被正确调用
        mock_pool.call.assert_called_once_with(
            "restore",
            {
                "text": "[PERSON_1] 和 [PHONE_1]",
                "request_id": "req-meta-check",
                "session_id": "sess-meta-check",
            },
        )


# ---------------------------------------------------------------------------
# V3-3 验证: 重复出现的同一占位符
# ---------------------------------------------------------------------------


class TestV3_3_RepeatedPlaceholder:
    """验证同一占位符在响应中多次出现时全部被还原。"""

    async def test_same_person_placeholder_twice(self, callback, mock_pool):
        """[PERSON_1] 出现两次，均被还原为同一原始值。"""
        response_text = (
            "[PERSON_1] 您好！根据记录，[PERSON_1] 的账户余额为 500 元。"
        )
        response_obj = make_response(response_text)
        kwargs = make_kwargs()

        expected_restored = (
            "王伟 您好！根据记录，王伟 的账户余额为 500 元。"
        )
        mock_pool.call.return_value = {"restored_text": expected_restored}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        result = response_obj.choices[0].message.content
        assert result == expected_restored
        # 确认不再包含任何占位符
        assert "[PERSON_1]" not in result
        # 确认两处都被还原为相同的值
        assert result.count("王伟") == 2

    async def test_same_phone_placeholder_three_times(self, callback, mock_pool):
        """[PHONE_1] 出现三次，均被正确还原。"""
        response_text = (
            "您的号码 [PHONE_1] 已注册。如需修改请拨打 [PHONE_1]，"
            "或通过 [PHONE_1] 接收验证码。"
        )
        response_obj = make_response(response_text)
        kwargs = make_kwargs()

        expected_restored = (
            "您的号码 18611112222 已注册。如需修改请拨打 18611112222，"
            "或通过 18611112222 接收验证码。"
        )
        mock_pool.call.return_value = {"restored_text": expected_restored}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        result = response_obj.choices[0].message.content
        assert result == expected_restored
        assert result.count("18611112222") == 3
        assert "[PHONE_1]" not in result


# ---------------------------------------------------------------------------
# V3-3 验证: 中文 PII 还原 (人名、手机号、身份证号)
# ---------------------------------------------------------------------------


class TestV3_3_ChinesePIIRestore:
    """验证中文 PII (人名、手机号、身份证号) 还原的正确性。"""

    async def test_chinese_name_restored(self, callback, mock_pool):
        """中文人名从 [PERSON_1] 还原为原始中文名。"""
        response_obj = make_response("经办人: [PERSON_1]，审批通过。")
        kwargs = make_kwargs()

        mock_pool.call.return_value = {
            "restored_text": "经办人: 赵丽颖，审批通过。"
        }

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        assert response_obj.choices[0].message.content == "经办人: 赵丽颖，审批通过。"

    async def test_chinese_phone_number_restored(self, callback, mock_pool):
        """中国手机号从 [PHONE_1] 还原为 11 位手机号。"""
        response_obj = make_response("联系电话: [PHONE_1]")
        kwargs = make_kwargs()

        mock_pool.call.return_value = {
            "restored_text": "联系电话: 13501234567"
        }

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        result = response_obj.choices[0].message.content
        assert result == "联系电话: 13501234567"
        assert "[PHONE_1]" not in result

    async def test_chinese_id_card_restored(self, callback, mock_pool):
        """中国身份证号从 [ID_CARD_1] 还原为 18 位身份证号。"""
        response_obj = make_response("身份证号: [ID_CARD_1]，请确认信息。")
        kwargs = make_kwargs()

        mock_pool.call.return_value = {
            "restored_text": "身份证号: 110101199001011234，请确认信息。"
        }

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        result = response_obj.choices[0].message.content
        assert result == "身份证号: 110101199001011234，请确认信息。"
        assert "[ID_CARD_1]" not in result

    async def test_full_chinese_pii_scenario(self, callback, mock_pool):
        """完整中文场景: 人名 + 手机号 + 身份证号 全部还原。"""
        response_text = (
            "员工 [PERSON_1]，手机 [PHONE_1]，身份证 [ID_CARD_1]，"
            "已完成入职手续。"
        )
        response_obj = make_response(response_text)
        kwargs = make_kwargs(request_id="req-chinese-full", session_id="sess-cn")

        expected_restored = (
            "员工 刘德华，手机 13800001234，身份证 440101198801012345，"
            "已完成入职手续。"
        )
        mock_pool.call.return_value = {"restored_text": expected_restored}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        result = response_obj.choices[0].message.content
        assert result == expected_restored
        assert "刘德华" in result
        assert "13800001234" in result
        assert "440101198801012345" in result


# ---------------------------------------------------------------------------
# V3-3 验证: response_obj 原地变更 (in-place mutation)
# ---------------------------------------------------------------------------


class TestV3_3_InPlaceMutation:
    """验证 response_obj 是被原地修改的 (LiteLLM 的期望行为)。"""

    async def test_response_obj_mutated_in_place(self, callback, mock_pool):
        """同一个 response_obj 引用被修改，不是返回新对象。"""
        response_obj = make_response("Hello [PERSON_1]")
        kwargs = make_kwargs()

        # 保存引用以验证原地修改
        original_choices_ref = response_obj.choices
        original_choice_ref = response_obj.choices[0]
        original_message_ref = response_obj.choices[0].message

        mock_pool.call.return_value = {"restored_text": "Hello 张三"}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        # 验证同一个对象引用被修改
        assert response_obj.choices is original_choices_ref
        assert response_obj.choices[0] is original_choice_ref
        assert response_obj.choices[0].message is original_message_ref
        # 内容已更新
        assert response_obj.choices[0].message.content == "Hello 张三"

    async def test_only_content_field_changed(self, callback, mock_pool):
        """只有 message.content 字段被修改，role 等其他字段保持不变。"""
        response_obj = make_response("[PHONE_1] is calling")
        # 确认 role 初始值
        assert response_obj.choices[0].message.role == "assistant"

        kwargs = make_kwargs()
        mock_pool.call.return_value = {"restored_text": "13800138000 is calling"}

        await callback.async_log_success_event(kwargs, response_obj, None, None)

        # content 已更新
        assert response_obj.choices[0].message.content == "13800138000 is calling"
        # role 未变
        assert response_obj.choices[0].message.role == "assistant"
        # index 未变
        assert response_obj.choices[0].index == 0

    async def test_return_value_is_none(self, callback, mock_pool):
        """async_log_success_event 返回 None (原地修改，不返回新对象)。"""
        response_obj = make_response("[EMAIL_1] confirmed")
        kwargs = make_kwargs()

        mock_pool.call.return_value = {"restored_text": "test@test.com confirmed"}

        result = await callback.async_log_success_event(kwargs, response_obj, None, None)

        # LiteLLM 的 async_log_success_event 不返回值
        assert result is None
        # 但 response_obj 被原地修改
        assert response_obj.choices[0].message.content == "test@test.com confirmed"


# ---------------------------------------------------------------------------
# V3-3 验证: 完整 E2E 流程模拟 (pre_call → LLM → post_call)
# ---------------------------------------------------------------------------


class TestV3_3_EndToEndFlow:
    """模拟完整流程: pre_call_hook 脱敏 → Mock LLM 返回含占位符响应 → post_call_hook 还原。"""

    async def test_full_flow_pre_call_to_post_call(self, callback, mock_pool):
        """完整 E2E: 客户端发送 PII → 脱敏 → LLM 返回占位符 → 还原 → 客户端收到原文。"""
        # === Phase 1: pre_call_hook (模拟请求阶段) ===
        request_data = {
            "messages": [
                {"role": "user", "content": "我叫张三，手机号是13800138000，帮我查订单"},
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-e2e-001",
                "request_id": "req-e2e-001",
            },
        }

        # pre_call_hook: compliance passes, then mask
        mock_pool.call.side_effect = [
            # compliance check
            {"passed": True, "violations": [], "mode": "strict"},
            # mask result
            {
                "masked_text": "我叫[PERSON_1]，手机号是[PHONE_1]，帮我查订单",
                "entities_found": [
                    {"type": "PERSON", "start": 2, "end": 4, "score": 0.95},
                    {"type": "PHONE_NUMBER", "start": 9, "end": 20, "score": 0.99},
                ],
            },
        ]

        await callback.async_pre_call_hook({}, None, request_data, "completion")

        # 验证消息已被脱敏
        assert request_data["messages"][0]["content"] == "我叫[PERSON_1]，手机号是[PHONE_1]，帮我查订单"

        # === Phase 2: Mock LLM 返回含占位符的响应 ===
        # (LLM 在训练数据中没有真实 PII，只看到占位符，原样返回占位符)
        llm_response = make_response(
            "[PERSON_1] 您好！您的手机号 [PHONE_1] 关联的订单已发货。"
        )

        # === Phase 3: post_call_hook (模拟响应还原阶段) ===
        # Reset mock for restore call
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "restored_text": "张三 您好！您的手机号 13800138000 关联的订单已发货。"
        }

        kwargs = {
            "metadata": request_data["metadata"],
            "model": "gpt-4o",
        }

        await callback.async_log_success_event(kwargs, llm_response, None, None)

        # === 验证: 客户端收到还原后的原始 PII 文本 ===
        client_response = llm_response.choices[0].message.content
        assert client_response == "张三 您好！您的手机号 13800138000 关联的订单已发货。"
        assert "[PERSON_1]" not in client_response
        assert "[PHONE_1]" not in client_response

        # 验证 restore 被正确调用
        mock_pool.call.assert_called_once_with(
            "restore",
            {
                "text": "[PERSON_1] 您好！您的手机号 [PHONE_1] 关联的订单已发货。",
                "request_id": "req-e2e-001",
                "session_id": "sess-e2e-001",
            },
        )

    async def test_full_flow_with_chinese_id_card(self, callback, mock_pool):
        """完整 E2E 流程: 包含身份证号的场景。"""
        # pre_call_hook
        request_data = {
            "messages": [
                {"role": "user", "content": "查询身份证110101199001011234对应的社保信息"},
            ],
            "model": "gpt-4o",
            "metadata": {
                "session_id": "sess-e2e-002",
                "request_id": "req-e2e-002",
            },
        }

        mock_pool.call.side_effect = [
            {"passed": True, "violations": [], "mode": "strict"},
            {
                "masked_text": "查询身份证[ID_CARD_1]对应的社保信息",
                "entities_found": [
                    {"type": "ID_CARD", "start": 4, "end": 22, "score": 0.99},
                ],
            },
        ]

        await callback.async_pre_call_hook({}, None, request_data, "completion")
        assert "[ID_CARD_1]" in request_data["messages"][0]["content"]

        # LLM response with placeholder
        llm_response = make_response("身份证 [ID_CARD_1] 的社保缴纳状态正常，累计缴纳36个月。")

        # post_call_hook restore
        mock_pool.call.reset_mock()
        mock_pool.call.side_effect = None
        mock_pool.call.return_value = {
            "restored_text": "身份证 110101199001011234 的社保缴纳状态正常，累计缴纳36个月。"
        }

        kwargs = {"metadata": request_data["metadata"]}
        await callback.async_log_success_event(kwargs, llm_response, None, None)

        # Client receives restored text
        assert llm_response.choices[0].message.content == (
            "身份证 110101199001011234 的社保缴纳状态正常，累计缴纳36个月。"
        )
