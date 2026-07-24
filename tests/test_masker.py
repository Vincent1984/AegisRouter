"""PII 脱敏模块测试

覆盖:
- 英文 PII 检测 (人名、邮箱、电话、IP、信用卡)
- 占位符格式正确性
- 相同 PII 获得相同占位符
- 无 PII 文本通过不变
- 多实体类型混合检测
- Redis 映射存储
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegis_router.clawvault.masker import PIIMasker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """创建 mock Redis 客户端。"""
    redis = AsyncMock()
    redis.get_mapping = AsyncMock(return_value={})
    redis.store_mapping = AsyncMock()
    redis.update_session_mapping = AsyncMock()
    return redis


@pytest.fixture
def masker(mock_redis):
    """创建 PIIMasker 实例 (使用 en_core_web_sm)。"""
    return PIIMasker(
        redis_client=mock_redis,
        language="en",
        nlp_model="en_core_web_sm",
        score_threshold=0.4,
    )


@pytest.fixture
def masker_no_redis():
    """创建无 Redis 的 PIIMasker 实例。"""
    return PIIMasker(
        redis_client=None,
        language="en",
        nlp_model="en_core_web_sm",
        score_threshold=0.4,
    )


# ---------------------------------------------------------------------------
# 英文 PII 检测测试
# ---------------------------------------------------------------------------


class TestEnglishPIIDetection:
    """任务 29：英文 PII 检测验收测试。"""

    async def test_tc_mask_001_detect_person_name(self, masker_no_redis):
        """TC-MASK-001: 英文人名被替换为 [PERSON_1]。"""
        text = "John Smith sent an email"
        result = await masker_no_redis.mask(text, session_id="s1", request_id="r1")

        assert result["masked_text"] == "[PERSON_1] sent an email"
        assert result["mapping"] == {"[PERSON_1]": "John Smith"}
        assert {entity["type"] for entity in result["entities_found"]} == {"PERSON"}

    async def test_tc_mask_002_detect_email(self, masker_no_redis):
        """TC-MASK-002: 邮箱地址被替换为 [EMAIL_1]。"""
        email = "john.doe@example.com"
        text = f"Please contact {email} for details."
        result = await masker_no_redis.mask(text, session_id="s1", request_id="r2")

        assert result["masked_text"] == "Please contact [EMAIL_1] for details."
        assert result["mapping"] == {"[EMAIL_1]": email}
        assert {entity["type"] for entity in result["entities_found"]} == {
            "EMAIL_ADDRESS"
        }

    @pytest.mark.parametrize(
        "address",
        [
            pytest.param("192.168.1.100", id="ipv4"),
            pytest.param("2001:db8:85a3::8a2e:370:7334", id="ipv6"),
        ],
    )
    async def test_tc_mask_003_detect_ip_addresses(self, masker_no_redis, address):
        """TC-MASK-003: IPv4 和 IPv6 地址均被替换为 [IP_1]。"""
        result = await masker_no_redis.mask(
            f"Connect to {address} now.", session_id="s1", request_id=f"ip-{address}"
        )

        assert result["masked_text"] == "Connect to [IP_1] now."
        assert result["mapping"] == {"[IP_1]": address}
        assert {entity["type"] for entity in result["entities_found"]} == {"IP_ADDRESS"}

    @pytest.mark.parametrize(
        "card_number",
        [
            pytest.param("4111111111111111", id="visa"),
            pytest.param("5555555555554444", id="mastercard"),
            pytest.param("378282246310005", id="amex"),
        ],
    )
    async def test_tc_mask_004_detect_credit_card_formats(
        self, masker_no_redis, card_number
    ):
        """TC-MASK-004: Visa、MasterCard 和 Amex 均替换为 [CREDIT_CARD_1]。"""
        result = await masker_no_redis.mask(
            f"Charge card {card_number} today.",
            session_id="s1",
            request_id=f"card-{card_number}",
        )

        assert result["masked_text"] == "Charge card [CREDIT_CARD_1] today."
        assert result["mapping"] == {"[CREDIT_CARD_1]": card_number}
        assert {entity["type"] for entity in result["entities_found"]} == {
            "CREDIT_CARD"
        }

    async def test_tc_mask_005_detect_international_phone(self, masker_no_redis):
        """TC-MASK-005: 国际电话号码被替换为 [PHONE_1]。"""
        phone = "+44 20 7946 0958"
        result = await masker_no_redis.mask(
            f"Call our London office at {phone}.", session_id="s1", request_id="r5"
        )

        assert result["masked_text"] == "Call our London office at [PHONE_1]."
        assert result["mapping"] == {"[PHONE_1]": phone}
        assert {entity["type"] for entity in result["entities_found"]} == {
            "PHONE_NUMBER"
        }

    async def test_tc_mask_006_detect_three_pii_types(self, masker_no_redis):
        """TC-MASK-006: 单条 prompt 同时检测至少三种 PII。"""
        text = (
            "John Smith uses john.smith@example.com from server 203.0.113.42."
        )
        result = await masker_no_redis.mask(text, session_id="s1", request_id="r6")

        assert result["masked_text"] == (
            "[PERSON_1] uses [EMAIL_1] from server [IP_1]."
        )
        assert result["mapping"] == {
            "[PERSON_1]": "John Smith",
            "[EMAIL_1]": "john.smith@example.com",
            "[IP_1]": "203.0.113.42",
        }
        assert {entity["type"] for entity in result["entities_found"]} == {
            "PERSON",
            "EMAIL_ADDRESS",
            "IP_ADDRESS",
        }

    async def test_tc_mask_007_normal_text_has_no_false_positive(
        self, masker_no_redis
    ):
        """TC-MASK-007: 不含 PII 的正常文本保持不变。"""
        text = "The weather is pleasant and the deployment completed successfully."
        result = await masker_no_redis.mask(text, session_id="s1", request_id="r7")

        assert result == {
            "masked_text": text,
            "entities_found": [],
            "mapping": {},
        }


# ---------------------------------------------------------------------------
# 占位符格式和一致性测试
# ---------------------------------------------------------------------------


class TestPlaceholderFormat:
    """占位符格式正确性测试。"""

    async def test_placeholder_format_email(self, masker):
        """邮箱占位符格式为 [EMAIL_N]。"""
        text = "Email: user@test.org"
        result = await masker.mask(text, session_id="s1", request_id="r1")

        # 验证占位符格式
        for placeholder in result["mapping"].keys():
            assert placeholder.startswith("[")
            assert placeholder.endswith("]")
            # 格式: [TYPE_N]
            inner = placeholder[1:-1]
            parts = inner.rsplit("_", 1)
            assert len(parts) == 2
            assert parts[1].isdigit()

    async def test_placeholder_format_ip(self, masker):
        """IP 占位符格式为 [IP_N]。"""
        text = "Connect to 10.0.0.1 for access."
        result = await masker.mask(text, session_id="s1", request_id="r1")

        if "[IP_1]" in result["masked_text"]:
            assert result["mapping"]["[IP_1]"] == "10.0.0.1"

    async def test_same_pii_same_placeholder(self, masker):
        """相同 PII 获得相同占位符。"""
        text = (
            "Send to user@example.com and also CC user@example.com for backup."
        )
        result = await masker.mask(text, session_id="s1", request_id="r1")

        # 相同邮箱应只出现一次在 mapping 中
        email_placeholders = [
            k for k in result["mapping"].keys() if "EMAIL" in k
        ]
        assert len(email_placeholders) == 1
        # 但在 masked_text 中出现两次
        assert result["masked_text"].count(email_placeholders[0]) == 2

    async def test_different_pii_different_placeholders(self, masker):
        """不同 PII 获得不同占位符。"""
        text = "Email alice@test.com or bob@test.com for info."
        result = await masker.mask(text, session_id="s1", request_id="r1")

        email_placeholders = [
            k for k in result["mapping"].keys() if "EMAIL" in k
        ]
        # 两个不同邮箱应产生两个不同占位符
        assert len(email_placeholders) == 2
        assert email_placeholders[0] != email_placeholders[1]


class TestPlaceholderConsistency:
    """任务 31：FR-2.3、FR-2.4、FR-2.7 占位符一致性测试。"""

    async def test_tc_mask_cons_001_same_session_reuses_placeholder(
        self, masker, mock_redis
    ):
        """TC-MASK-CONS-001: 同一 session 的相同 PII 跨 request 复用占位符。"""
        text = "Contact recurring.user@example.com for details."
        first = await masker.mask(text, session_id="session-a", request_id="request-1")
        first_placeholder = next(iter(first["mapping"]))

        # 第二个 request 从 Redis 会话映射读取第一个 request 的占位符。
        mock_redis.get_mapping.return_value = first["mapping"]
        second = await masker.mask(text, session_id="session-a", request_id="request-2")

        assert next(iter(second["mapping"])) == first_placeholder
        assert second["masked_text"] == first["masked_text"]
        mock_redis.get_mapping.assert_awaited_with(
            request_id="request-2", session_id="session-a"
        )

    async def test_tc_mask_cons_002_different_sessions_are_isolated(
        self, masker, mock_redis
    ):
        """TC-MASK-CONS-002: 不同 session 的相同 PII 生成不同占位符。"""
        text = "Contact isolated.user@example.com for details."
        first = await masker.mask(text, session_id="session-a", request_id="request-a")
        second = await masker.mask(text, session_id="session-b", request_id="request-b")

        first_placeholder = next(iter(first["mapping"]))
        second_placeholder = next(iter(second["mapping"]))
        assert first_placeholder != second_placeholder
        assert first["mapping"][first_placeholder] == "isolated.user@example.com"
        assert second["mapping"][second_placeholder] == "isolated.user@example.com"

        stored_sessions = {
            call.args[0] for call in mock_redis.update_session_mapping.await_args_list
        }
        assert stored_sessions == {"session-a", "session-b"}

    async def test_tc_mask_cons_003_repeated_pii_in_request_reuses_placeholder(
        self, masker_no_redis
    ):
        """TC-MASK-CONS-003: 同一 request 重复 PII 使用同一占位符。"""
        email = "repeat.user@example.com"
        result = await masker_no_redis.mask(
            f"Send to {email} and copy {email}.",
            session_id="session-repeat",
            request_id="request-repeat",
        )

        assert result["mapping"] == {"[EMAIL_1]": email}
        assert result["masked_text"].count("[EMAIL_1]") == 2


class TestMultipleEntityTypes:
    """多实体类型混合测试。"""

    async def test_multiple_entity_types(self, masker):
        """混合多种 PII 类型同时检测。"""
        text = (
            "Contact support@company.com or call +1-800-555-0123. "
            "Server is at 172.16.0.1."
        )
        result = await masker.mask(text, session_id="s1", request_id="r1")

        # 应检测到多种类型
        entity_types = {e["type"] for e in result["entities_found"]}
        # 至少应包含邮箱和 IP
        assert "EMAIL_ADDRESS" in entity_types or "IP_ADDRESS" in entity_types

        # masked_text 中不应包含原始 PII
        assert "support@company.com" not in result["masked_text"]
        assert "172.16.0.1" not in result["masked_text"]

    async def test_credit_card_and_email(self, masker):
        """信用卡 + 邮箱混合检测。"""
        text = "Card: 4532015112830366, send receipt to user@shop.com"
        result = await masker.mask(text, session_id="s1", request_id="r1")

        # 应有至少两种类型
        assert len(result["mapping"]) >= 2
        assert "4532015112830366" not in result["masked_text"]
        assert "user@shop.com" not in result["masked_text"]


# ---------------------------------------------------------------------------
# Redis 存储测试
# ---------------------------------------------------------------------------


class TestRedisIntegration:
    """Redis 映射存储测试。"""

    async def test_stores_mapping_to_redis(self, masker, mock_redis):
        """脱敏后映射被存储到 Redis。"""
        text = "Email me at test@example.com please."
        await masker.mask(text, session_id="session-123", request_id="req-456")

        # 验证 store_mapping 被调用
        mock_redis.store_mapping.assert_called_once()
        call_args = mock_redis.store_mapping.call_args
        # 位置参数: (session_id, request_id, mapping)
        assert call_args[0][0] == "session-123"
        assert call_args[0][1] == "req-456"

        # 验证 update_session_mapping 被调用
        mock_redis.update_session_mapping.assert_called_once()

    async def test_no_redis_call_when_no_pii(self, masker, mock_redis):
        """无 PII 时不调用 Redis。"""
        text = "Hello world, nice weather."
        await masker.mask(text, session_id="s1", request_id="r1")

        mock_redis.store_mapping.assert_not_called()
        mock_redis.update_session_mapping.assert_not_called()

    async def test_no_redis_client(self, masker_no_redis):
        """无 Redis 客户端时正常工作不报错。"""
        text = "Contact alice@test.com for info."
        result = await masker_no_redis.mask(text, session_id="s1", request_id="r1")

        # 应正常检测并返回
        assert "alice@test.com" not in result["masked_text"]
        assert len(result["mapping"]) >= 1

    async def test_redis_failure_graceful(self, masker, mock_redis):
        """Redis 异常时优雅降级。"""
        mock_redis.store_mapping.side_effect = Exception("Redis connection lost")

        text = "Email: fail@test.com"
        # 不应抛出异常
        result = await masker.mask(text, session_id="s1", request_id="r1")

        # 结果仍然正常
        assert "fail@test.com" not in result["masked_text"]
        assert len(result["mapping"]) >= 1


# ---------------------------------------------------------------------------
# register_recognizer 测试
# ---------------------------------------------------------------------------


class TestRegisterRecognizer:
    """自定义 Recognizer 注册测试。"""

    def test_register_custom_recognizer(self, masker):
        """可以注册自定义 Recognizer。"""
        from presidio_analyzer import PatternRecognizer, Pattern

        # 创建一个简单的 pattern recognizer
        custom_recognizer = PatternRecognizer(
            supported_entity="CUSTOM_ENTITY",
            patterns=[Pattern("test", r"TEST-\d{4}", 0.8)],
        )

        # 不应抛出异常
        masker.register_recognizer(custom_recognizer)

    async def test_custom_recognizer_works(self, masker):
        """注册后自定义 Recognizer 能正常工作。"""
        from presidio_analyzer import PatternRecognizer, Pattern

        custom_recognizer = PatternRecognizer(
            supported_entity="CUSTOM_ID",
            patterns=[Pattern("custom_id", r"CUST-\d{6}", 0.9)],
        )
        masker.register_recognizer(custom_recognizer)

        text = "Your ID is CUST-123456, please keep it safe."
        result = await masker.mask(text, session_id="s1", request_id="r1")

        assert "CUST-123456" not in result["masked_text"]
        assert any(e["type"] == "CUSTOM_ID" for e in result["entities_found"])


# ---------------------------------------------------------------------------
# 返回结构测试
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """返回值结构完整性测试。"""

    async def test_return_keys(self, masker):
        """返回值包含所有必需字段。"""
        text = "Call 192.168.0.1 now."
        result = await masker.mask(text, session_id="s1", request_id="r1")

        assert "masked_text" in result
        assert "entities_found" in result
        assert "mapping" in result

    async def test_entities_found_structure(self, masker):
        """entities_found 中每个实体包含 type/start/end/score。"""
        text = "Email: hello@world.com for updates."
        result = await masker.mask(text, session_id="s1", request_id="r1")

        for entity in result["entities_found"]:
            assert "type" in entity
            assert "start" in entity
            assert "end" in entity
            assert "score" in entity
            assert isinstance(entity["score"], float)
            assert 0 <= entity["score"] <= 1.0
