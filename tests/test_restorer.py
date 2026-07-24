"""占位符还原模块测试

覆盖:
- 单个占位符还原
- 多个不同类型占位符同时还原
- 无占位符文本原样返回
- 占位符在文本中出现多次
- 映射为空/过期 (占位符保留原样)
- 不在映射中的占位符保持不变
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aegis_router.clawvault.restorer import PIIRestorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """创建 mock Redis 客户端。"""
    redis = AsyncMock()
    redis.get_mapping = AsyncMock(return_value={})
    return redis


@pytest.fixture
def restorer(mock_redis):
    """创建 PIIRestorer 实例。"""
    return PIIRestorer(redis_client=mock_redis)


# ---------------------------------------------------------------------------
# 单占位符还原测试
# ---------------------------------------------------------------------------


class TestSinglePlaceholderRestore:
    """单个占位符正确还原。"""

    async def test_tc_restore_001_single_placeholder(self, restorer, mock_redis):
        """TC-RESTORE-001: [PERSON_1] 还原为原始人名。"""
        mock_redis.get_mapping.return_value = {"[PERSON_1]": "张三"}

        result = await restorer.restore(
            text="你好 [PERSON_1]，欢迎回来。",
            request_id="req-001",
            session_id="sess-001",
        )

        assert result["restored_text"] == "你好 张三，欢迎回来。"

    async def test_restore_phone_placeholder(self, restorer, mock_redis):
        """[PHONE_1] 还原为原始手机号。"""
        mock_redis.get_mapping.return_value = {"[PHONE_1]": "13800138000"}

        result = await restorer.restore(
            text="请拨打 [PHONE_1] 联系客服。",
            request_id="req-002",
            session_id="sess-001",
        )

        assert result["restored_text"] == "请拨打 13800138000 联系客服。"

    async def test_restore_email_placeholder(self, restorer, mock_redis):
        """[EMAIL_1] 还原为原始邮箱。"""
        mock_redis.get_mapping.return_value = {"[EMAIL_1]": "user@example.com"}

        result = await restorer.restore(
            text="发送邮件到 [EMAIL_1] 获取详情。",
            request_id="req-003",
            session_id="sess-001",
        )

        assert result["restored_text"] == "发送邮件到 user@example.com 获取详情。"


# ---------------------------------------------------------------------------
# 多占位符同时还原测试
# ---------------------------------------------------------------------------


class TestMultiplePlaceholderRestore:
    """TC-RESTORE-002: 多个不同类型占位符同时还原。"""

    async def test_restore_multiple_types(self, restorer, mock_redis):
        """同时还原 PERSON、PHONE、EMAIL 占位符。"""
        mock_redis.get_mapping.return_value = {
            "[PERSON_1]": "张三",
            "[PHONE_1]": "13800138000",
            "[EMAIL_1]": "zhangsan@company.com",
        }

        text = "[PERSON_1] 的联系方式: 电话 [PHONE_1], 邮箱 [EMAIL_1]。"
        result = await restorer.restore(
            text=text,
            request_id="req-010",
            session_id="sess-001",
        )

        expected = "张三 的联系方式: 电话 13800138000, 邮箱 zhangsan@company.com。"
        assert result["restored_text"] == expected

    async def test_restore_multiple_same_type(self, restorer, mock_redis):
        """还原多个相同类型的不同编号占位符。"""
        mock_redis.get_mapping.return_value = {
            "[PERSON_1]": "张三",
            "[PERSON_2]": "李四",
        }

        text = "[PERSON_1] 和 [PERSON_2] 是同事。"
        result = await restorer.restore(
            text=text,
            request_id="req-011",
            session_id="sess-001",
        )

        assert result["restored_text"] == "张三 和 李四 是同事。"

    async def test_restore_all_placeholder_types(self, restorer, mock_redis):
        """还原所有支持的占位符类型。"""
        mock_redis.get_mapping.return_value = {
            "[PERSON_1]": "张三",
            "[PHONE_1]": "13800138000",
            "[EMAIL_1]": "test@test.com",
            "[IP_1]": "192.168.1.1",
            "[CREDIT_CARD_1]": "4111111111111111",
            "[ID_CARD_1]": "110101199003071234",
        }

        text = (
            "姓名: [PERSON_1], 电话: [PHONE_1], 邮箱: [EMAIL_1], "
            "IP: [IP_1], 信用卡: [CREDIT_CARD_1], 身份证: [ID_CARD_1]"
        )
        result = await restorer.restore(
            text=text,
            request_id="req-012",
            session_id="sess-001",
        )

        assert "张三" in result["restored_text"]
        assert "13800138000" in result["restored_text"]
        assert "test@test.com" in result["restored_text"]
        assert "192.168.1.1" in result["restored_text"]
        assert "4111111111111111" in result["restored_text"]
        assert "110101199003071234" in result["restored_text"]
        # 确认没有占位符残留
        assert "[PERSON_1]" not in result["restored_text"]
        assert "[PHONE_1]" not in result["restored_text"]


# ---------------------------------------------------------------------------
# 无占位符测试
# ---------------------------------------------------------------------------


class TestNoPlaceholders:
    """TC-RESTORE-003: 无占位符文本原样返回。"""

    async def test_no_placeholders_returns_unchanged(self, restorer, mock_redis):
        """文本中没有占位符时原样返回。"""
        mock_redis.get_mapping.return_value = {"[PERSON_1]": "张三"}

        text = "今天天气不错，适合出门散步。"
        result = await restorer.restore(
            text=text,
            request_id="req-020",
            session_id="sess-001",
        )

        assert result["restored_text"] == text

    async def test_empty_text_returns_empty(self, restorer, mock_redis):
        """空文本返回空。"""
        mock_redis.get_mapping.return_value = {"[PERSON_1]": "张三"}

        result = await restorer.restore(
            text="",
            request_id="req-021",
            session_id="sess-001",
        )

        assert result["restored_text"] == ""


# ---------------------------------------------------------------------------
# 占位符多次出现测试
# ---------------------------------------------------------------------------


class TestRepeatedPlaceholders:
    """TC-RESTORE-004: 占位符在文本中出现多次。"""

    async def test_placeholder_appears_multiple_times(self, restorer, mock_redis):
        """同一占位符出现多次，每次都正确还原。"""
        mock_redis.get_mapping.return_value = {"[PERSON_1]": "张三"}

        text = "[PERSON_1] 说他叫 [PERSON_1]，大家都认识 [PERSON_1]。"
        result = await restorer.restore(
            text=text,
            request_id="req-030",
            session_id="sess-001",
        )

        expected = "张三 说他叫 张三，大家都认识 张三。"
        assert result["restored_text"] == expected
        assert "[PERSON_1]" not in result["restored_text"]

    async def test_multiple_different_placeholders_repeated(self, restorer, mock_redis):
        """多个不同占位符各自出现多次。"""
        mock_redis.get_mapping.return_value = {
            "[PERSON_1]": "张三",
            "[PHONE_1]": "13800138000",
        }

        text = "[PERSON_1] 电话 [PHONE_1]，再强调一下 [PERSON_1] 的号码是 [PHONE_1]。"
        result = await restorer.restore(
            text=text,
            request_id="req-031",
            session_id="sess-001",
        )

        expected = "张三 电话 13800138000，再强调一下 张三 的号码是 13800138000。"
        assert result["restored_text"] == expected


# ---------------------------------------------------------------------------
# 映射为空/过期测试
# ---------------------------------------------------------------------------


class TestEmptyOrExpiredMapping:
    """TC-RESTORE-005: 映射为空或过期时的降级行为。"""

    async def test_empty_mapping_leaves_placeholders(self, restorer, mock_redis):
        """映射为空时，占位符保持原样。"""
        mock_redis.get_mapping.return_value = {}

        text = "联系 [PERSON_1] 获取帮助。"
        result = await restorer.restore(
            text=text,
            request_id="req-040",
            session_id="sess-001",
        )

        assert result["restored_text"] == "联系 [PERSON_1] 获取帮助。"

    async def test_none_session_id(self, restorer, mock_redis):
        """session_id 为 None 时正常工作。"""
        mock_redis.get_mapping.return_value = {"[PHONE_1]": "13900139000"}

        result = await restorer.restore(
            text="拨打 [PHONE_1]",
            request_id="req-041",
            session_id=None,
        )

        assert result["restored_text"] == "拨打 13900139000"

    async def test_redis_returns_empty_dict(self, restorer, mock_redis):
        """Redis 返回空字典 (TTL 过期后的场景)。"""
        mock_redis.get_mapping.return_value = {}

        text = "[PERSON_1] 的身份证号是 [ID_CARD_1]。"
        result = await restorer.restore(
            text=text,
            request_id="req-042",
            session_id="sess-expired",
        )

        # 占位符保持原样
        assert "[PERSON_1]" in result["restored_text"]
        assert "[ID_CARD_1]" in result["restored_text"]


# ---------------------------------------------------------------------------
# 部分占位符不在映射中测试
# ---------------------------------------------------------------------------


class TestPartialMapping:
    """占位符部分不在映射中的情况。"""

    async def test_unknown_placeholder_left_unchanged(self, restorer, mock_redis):
        """不在映射中的占位符保持原样。"""
        mock_redis.get_mapping.return_value = {"[PERSON_1]": "张三"}

        text = "[PERSON_1] 的电话是 [PHONE_1]。"
        result = await restorer.restore(
            text=text,
            request_id="req-050",
            session_id="sess-001",
        )

        # PERSON_1 被还原, PHONE_1 保留
        assert "张三" in result["restored_text"]
        assert "[PHONE_1]" in result["restored_text"]
        assert "[PERSON_1]" not in result["restored_text"]

    async def test_mixed_known_and_unknown_placeholders(self, restorer, mock_redis):
        """部分占位符在映射中，部分不在。"""
        mock_redis.get_mapping.return_value = {
            "[PERSON_1]": "张三",
            "[EMAIL_1]": "test@example.com",
        }

        text = "[PERSON_1] 邮箱 [EMAIL_1], IP [IP_1], 卡号 [CREDIT_CARD_1]。"
        result = await restorer.restore(
            text=text,
            request_id="req-051",
            session_id="sess-001",
        )

        assert "张三" in result["restored_text"]
        assert "test@example.com" in result["restored_text"]
        assert "[IP_1]" in result["restored_text"]
        assert "[CREDIT_CARD_1]" in result["restored_text"]


# ---------------------------------------------------------------------------
# RedisClient 调用验证
# ---------------------------------------------------------------------------


class TestRedisInteraction:
    """验证 RedisClient 的调用方式正确。"""

    async def test_get_mapping_called_with_correct_args(self, restorer, mock_redis):
        """确认 get_mapping 被正确调用。"""
        mock_redis.get_mapping.return_value = {}

        await restorer.restore(
            text="Hello",
            request_id="req-100",
            session_id="sess-200",
        )

        mock_redis.get_mapping.assert_called_once_with(
            request_id="req-100",
            session_id="sess-200",
        )

    async def test_get_mapping_called_without_session(self, restorer, mock_redis):
        """session_id=None 时正确传递。"""
        mock_redis.get_mapping.return_value = {}

        await restorer.restore(
            text="Hello",
            request_id="req-101",
            session_id=None,
        )

        mock_redis.get_mapping.assert_called_once_with(
            request_id="req-101",
            session_id=None,
        )


# ---------------------------------------------------------------------------
# 返回结构测试
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """返回值结构完整性测试。"""

    async def test_return_contains_restored_text_key(self, restorer, mock_redis):
        """返回值包含 restored_text 字段。"""
        mock_redis.get_mapping.return_value = {}

        result = await restorer.restore(
            text="Test",
            request_id="req-200",
            session_id="sess-001",
        )

        assert "restored_text" in result
        assert isinstance(result["restored_text"], str)

    async def test_return_type_is_dict(self, restorer, mock_redis):
        """返回值类型为 dict。"""
        mock_redis.get_mapping.return_value = {}

        result = await restorer.restore(
            text="Test",
            request_id="req-201",
            session_id="sess-001",
        )

        assert isinstance(result, dict)
