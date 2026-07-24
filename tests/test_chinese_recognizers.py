"""中文 PII 识别器测试

覆盖:
- ChinesePhoneRecognizer: 中国手机号识别
- ChineseIdCardRecognizer: 中国身份证号识别 + 校验位验证
- 与 PIIMasker 集成测试
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aegis_router.clawvault.masker import PIIMasker
from aegis_router.clawvault.recognizers import (
    ChineseIdCardRecognizer,
    ChinesePhoneRecognizer,
    validate_id_card_checksum,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def phone_recognizer():
    return ChinesePhoneRecognizer()


@pytest.fixture
def id_card_recognizer():
    return ChineseIdCardRecognizer()


@pytest.fixture
def masker_with_cn_recognizers():
    """创建集成了中文识别器的 PIIMasker。"""
    masker = PIIMasker(
        redis_client=None,
        language="en",
        nlp_model="en_core_web_sm",
        score_threshold=0.4,
    )
    masker.register_recognizer(ChinesePhoneRecognizer())
    masker.register_recognizer(ChineseIdCardRecognizer())
    return masker


# ---------------------------------------------------------------------------
# ChinesePhoneRecognizer 测试
# ---------------------------------------------------------------------------


class TestChinesePhoneRecognizer:
    """中国手机号识别器测试。"""

    def test_supported_entity(self, phone_recognizer):
        """识别器支持 CN_PHONE 实体类型。"""
        assert "CN_PHONE" in phone_recognizer.supported_entities

    def test_detect_130_segment(self, phone_recognizer):
        """检测 130 号段手机号。"""
        results = phone_recognizer.analyze(
            text="请拨打 13012345678 联系我",
            entities=["CN_PHONE"],
        )
        assert len(results) == 1
        assert results[0].entity_type == "CN_PHONE"
        assert results[0].start == 4
        assert results[0].end == 15

    def test_detect_138_segment(self, phone_recognizer):
        """检测 138 号段手机号 (经典号段)。"""
        results = phone_recognizer.analyze(
            text="联系电话: 13800138000",
            entities=["CN_PHONE"],
        )
        assert len(results) == 1
        assert results[0].entity_type == "CN_PHONE"

    def test_detect_191_new_segment(self, phone_recognizer):
        """检测 191 新号段。"""
        results = phone_recognizer.analyze(
            text="我的手机号是 19100001234",
            entities=["CN_PHONE"],
        )
        assert len(results) == 1

    def test_detect_199_new_segment(self, phone_recognizer):
        """检测 199 新号段。"""
        results = phone_recognizer.analyze(
            text="请用 19912345678 联系",
            entities=["CN_PHONE"],
        )
        assert len(results) == 1

    def test_no_match_12_prefix(self, phone_recognizer):
        """12 开头的号码不应匹配。"""
        results = phone_recognizer.analyze(
            text="电话号码 12345678901",
            entities=["CN_PHONE"],
        )
        assert len(results) == 0

    def test_no_match_10_digits(self, phone_recognizer):
        """不足 11 位不应匹配。"""
        results = phone_recognizer.analyze(
            text="号码 1380013800",
            entities=["CN_PHONE"],
        )
        assert len(results) == 0

    def test_no_match_12_digits(self, phone_recognizer):
        """超过 11 位的长数字不应匹配为手机号。"""
        results = phone_recognizer.analyze(
            text="编号 138001380001",
            entities=["CN_PHONE"],
        )
        # 12 位数字不应被匹配 (由 lookahead/lookbehind 限定)
        assert len(results) == 0

    def test_multiple_phones(self, phone_recognizer):
        """同一文本中多个手机号。"""
        results = phone_recognizer.analyze(
            text="张三 13800138000，李四 15900001111",
            entities=["CN_PHONE"],
        )
        assert len(results) == 2

    def test_phone_in_sentence(self, phone_recognizer):
        """手机号嵌在句子中。"""
        results = phone_recognizer.analyze(
            text="客服热线13912345678欢迎拨打",
            entities=["CN_PHONE"],
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# ChineseIdCardRecognizer 测试
# ---------------------------------------------------------------------------


class TestIdCardChecksum:
    """身份证校验位验证函数测试。"""

    def test_valid_checksum(self):
        """合法身份证号通过校验。"""
        # 110101199003070011 — 手动计算校验位正确
        assert validate_id_card_checksum("110101199003070011") is True

    def test_valid_with_x_upper(self):
        """校验码为大写 X 的合法身份证号。"""
        # 11010119900307002X — 校验位为 X
        assert validate_id_card_checksum("11010119900307002X") is True

    def test_valid_with_x_lower(self):
        """校验码为小写 x 应通过。"""
        assert validate_id_card_checksum("11010119900307002x") is True

    def test_invalid_checksum(self):
        """校验位错误的身份证号不通过。"""
        # 将正确的尾号 1 改为 5 (110101199003070011 → 110101199003070015)
        assert validate_id_card_checksum("110101199003070015") is False

    def test_too_short(self):
        """长度不足 18 位不通过。"""
        assert validate_id_card_checksum("11010119900307001") is False

    def test_too_long(self):
        """长度超过 18 位不通过。"""
        assert validate_id_card_checksum("1101011990030700141") is False

    def test_non_digit_in_first_17(self):
        """前 17 位含非数字不通过。"""
        assert validate_id_card_checksum("11010119900307A014") is False


class TestChineseIdCardRecognizer:
    """中国身份证号识别器测试。"""

    def test_supported_entity(self, id_card_recognizer):
        """识别器支持 CN_ID_CARD 实体类型。"""
        assert "CN_ID_CARD" in id_card_recognizer.supported_entities

    def test_detect_valid_id_card(self, id_card_recognizer):
        """检测合法身份证号。"""
        results = id_card_recognizer.analyze(
            text="身份证号码: 110101199003070011",
            entities=["CN_ID_CARD"],
        )
        assert len(results) == 1
        assert results[0].entity_type == "CN_ID_CARD"
        assert results[0].score == 1.0  # checksum 通过应提升到 MAX_SCORE

    def test_detect_id_card_with_x(self, id_card_recognizer):
        """检测校验位为 X 的身份证号。"""
        results = id_card_recognizer.analyze(
            text="证件 11010119900307002X 已验证",
            entities=["CN_ID_CARD"],
        )
        assert len(results) == 1
        assert results[0].score == 1.0

    def test_reject_invalid_checksum(self, id_card_recognizer):
        """拒绝校验位错误的号码 (不应有匹配结果)。"""
        results = id_card_recognizer.analyze(
            text="身份证号码: 110101199003070015",
            entities=["CN_ID_CARD"],
        )
        # 校验失败，结果被丢弃
        assert len(results) == 0

    def test_no_match_random_digits(self, id_card_recognizer):
        """随机 18 位数字不应匹配 (日期格式不合法)。"""
        results = id_card_recognizer.analyze(
            text="编号 123456789012345678",
            entities=["CN_ID_CARD"],
        )
        assert len(results) == 0

    def test_no_match_invalid_month(self, id_card_recognizer):
        """月份 > 12 不应匹配。"""
        results = id_card_recognizer.analyze(
            text="号码 110101199013070014",
            entities=["CN_ID_CARD"],
        )
        assert len(results) == 0

    def test_no_match_invalid_day(self, id_card_recognizer):
        """日期 > 31 不应匹配。"""
        results = id_card_recognizer.analyze(
            text="号码 110101199003320014",
            entities=["CN_ID_CARD"],
        )
        assert len(results) == 0

    def test_id_card_in_longer_number(self, id_card_recognizer):
        """身份证号嵌入更长的数字串时不应匹配。"""
        results = id_card_recognizer.analyze(
            text="流水号 91101011990030700149",
            entities=["CN_ID_CARD"],
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# 与 PIIMasker 集成测试
# ---------------------------------------------------------------------------


class TestMaskerIntegration:
    """识别器与 PIIMasker 集成测试。"""

    async def test_phone_masked_as_phone_placeholder(self, masker_with_cn_recognizers):
        """中国手机号应替换为 [PHONE_N]。"""
        text = "请联系 13800138000 获取更多信息"
        result = await masker_with_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )

        assert "13800138000" not in result["masked_text"]
        assert "[PHONE_1]" in result["masked_text"]
        assert result["mapping"]["[PHONE_1]"] == "13800138000"

    async def test_id_card_masked_as_id_card_placeholder(
        self, masker_with_cn_recognizers
    ):
        """中国身份证号应替换为 [ID_CARD_N]。"""
        text = "员工身份证号为 110101199003070011"
        result = await masker_with_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )

        assert "110101199003070011" not in result["masked_text"]
        assert "[ID_CARD_1]" in result["masked_text"]
        assert result["mapping"]["[ID_CARD_1]"] == "110101199003070011"

    async def test_mixed_cn_pii(self, masker_with_cn_recognizers):
        """同时包含手机号和身份证号。"""
        text = "张三手机 13800138000 身份证 110101199003070011"
        result = await masker_with_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )

        assert "13800138000" not in result["masked_text"]
        assert "110101199003070011" not in result["masked_text"]
        assert "[PHONE_1]" in result["masked_text"]
        assert "[ID_CARD_1]" in result["masked_text"]

    async def test_invalid_id_card_not_masked(self, masker_with_cn_recognizers):
        """校验位错误的身份证号不应被脱敏。"""
        text = "无效号码 110101199003070015 不应被检测"
        result = await masker_with_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )

        # 无效的身份证号不应被替换
        assert "110101199003070015" in result["masked_text"]

    async def test_v2_4_id_card_detected_and_replaced(self, masker_with_cn_recognizers):
        """V2-4 验证: 中国身份证号检测并替换为 [ID_CARD_1]。

        注: requirements 中的示例 110101199003071234 校验位不合法 (期望为 3),
        使用校验位正确的 110101199003071233 验证完整流程。
        同时验证 110101199003071234 因校验失败不被检测。
        """
        # 合法身份证号 (校验位正确) 应被检测并替换
        valid_id = "110101199003071233"
        text = f"用户身份证号为 {valid_id} 请妥善保管"
        result = await masker_with_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )
        assert valid_id not in result["masked_text"]
        assert "[ID_CARD_1]" in result["masked_text"]
        assert result["mapping"]["[ID_CARD_1]"] == valid_id

        # 非法身份证号 (校验位错误) 不应被检测
        invalid_id = "110101199003071234"
        text_invalid = f"用户身份证号为 {invalid_id} 请妥善保管"
        result_invalid = await masker_with_cn_recognizers.mask(
            text_invalid, session_id="s2", request_id="r2"
        )
        assert invalid_id in result_invalid["masked_text"]
        assert "[ID_CARD_1]" not in result_invalid["masked_text"]

    async def test_new_phone_segments(self, masker_with_cn_recognizers):
        """新号段 (191/199) 应被正确识别。"""
        text = "新号码 19100001234 和 19912345678"
        result = await masker_with_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )

        assert "19100001234" not in result["masked_text"]
        assert "19912345678" not in result["masked_text"]
        assert "[PHONE_1]" in result["masked_text"]
        assert "[PHONE_2]" in result["masked_text"]


# ---------------------------------------------------------------------------
# ChineseNameRecognizer 测试
# ---------------------------------------------------------------------------

from aegis_router.clawvault.recognizers import ChineseNameRecognizer


@pytest.fixture
def name_recognizer():
    """创建使用启发式回退的中文人名识别器。"""
    return ChineseNameRecognizer(nlp=None)


@pytest.fixture
def masker_with_all_cn_recognizers():
    """创建集成了所有中文识别器 (含 ChineseNameRecognizer) 的 PIIMasker。"""
    masker = PIIMasker(
        redis_client=None,
        language="en",
        nlp_model="en_core_web_sm",
        score_threshold=0.4,
    )
    masker.register_recognizer(ChinesePhoneRecognizer())
    masker.register_recognizer(ChineseIdCardRecognizer())
    masker.register_recognizer(ChineseNameRecognizer())
    return masker


class TestChineseNameRecognizer:
    """中文人名识别器测试。"""

    def test_supported_entity(self, name_recognizer):
        """识别器支持 CN_NAME 实体类型。"""
        assert "CN_NAME" in name_recognizer.supported_entities

    def test_detect_two_char_name(self, name_recognizer):
        """检测常见 2 字中文人名 (姓 + 1 字名)。"""
        results = name_recognizer.analyze(
            text="张三是我的朋友",
            entities=["CN_NAME"],
        )
        assert len(results) >= 1
        # 验证检测到 "张三"
        detected_names = [
            "张三是我的朋友"[r.start: r.end] for r in results
        ]
        assert "张三" in detected_names

    def test_detect_three_char_name(self, name_recognizer):
        """检测 3 字中文人名 (姓 + 2 字名)。"""
        results = name_recognizer.analyze(
            text="张三丰是武当派创始人",
            entities=["CN_NAME"],
        )
        assert len(results) >= 1
        detected_names = [
            "张三丰是武当派创始人"[r.start: r.end] for r in results
        ]
        assert "张三丰" in detected_names

    def test_detect_name_in_context(self, name_recognizer):
        """在上下文句子中检测人名。"""
        text = "请联系张三获取报告"
        results = name_recognizer.analyze(
            text=text,
            entities=["CN_NAME"],
        )
        assert len(results) >= 1
        detected_names = [text[r.start: r.end] for r in results]
        assert "张三" in detected_names

    def test_context_words_boost_score(self, name_recognizer):
        """上下文关键词应提升置信度。"""
        # 有上下文词
        results_with_context = name_recognizer.analyze(
            text="姓名：张三",
            entities=["CN_NAME"],
        )
        # 无上下文词
        results_without_context = name_recognizer.analyze(
            text="张三",
            entities=["CN_NAME"],
        )
        assert len(results_with_context) >= 1
        assert len(results_without_context) >= 1
        # 有上下文的分数应更高
        assert results_with_context[0].score > results_without_context[0].score

    def test_no_false_positive_common_words(self, name_recognizer):
        """常见非人名词语不应被误识别。"""
        text = "周末去高中参观"
        results = name_recognizer.analyze(
            text=text,
            entities=["CN_NAME"],
        )
        detected_names = [text[r.start: r.end] for r in results]
        assert "周末" not in detected_names
        assert "高中" not in detected_names

    def test_no_detection_for_irrelevant_entities(self, name_recognizer):
        """当请求的实体类型不含 CN_NAME 时返回空列表。"""
        results = name_recognizer.analyze(
            text="张三是我的朋友",
            entities=["CN_PHONE"],
        )
        assert len(results) == 0

    def test_multiple_names(self, name_recognizer):
        """检测同一文本中的多个人名。"""
        text = "张三和李四是好朋友"
        results = name_recognizer.analyze(
            text=text,
            entities=["CN_NAME"],
        )
        detected_names = [text[r.start: r.end] for r in results]
        assert "张三" in detected_names
        assert "李四" in detected_names


class TestChineseNameMaskerIntegration:
    """ChineseNameRecognizer 与 PIIMasker 集成测试。"""

    async def test_name_masked_as_person_placeholder(
        self, masker_with_all_cn_recognizers
    ):
        """中文人名应替换为 [PERSON_N]。"""
        text = "请联系张三获取报告"
        result = await masker_with_all_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )

        assert "张三" not in result["masked_text"]
        assert "[PERSON_1]" in result["masked_text"]
        assert result["mapping"]["[PERSON_1]"] == "张三"

    async def test_name_with_phone_masked(self, masker_with_all_cn_recognizers):
        """同时包含人名和手机号时，各自被正确脱敏。"""
        text = "张三的手机号是13800138000"
        result = await masker_with_all_cn_recognizers.mask(
            text, session_id="s1", request_id="r1"
        )

        assert "张三" not in result["masked_text"]
        assert "13800138000" not in result["masked_text"]
        # 人名映射到 PERSON，手机号映射到 PHONE
        has_person = any("PERSON" in k for k in result["mapping"])
        has_phone = any("PHONE" in k for k in result["mapping"])
        assert has_person
        assert has_phone


# ---------------------------------------------------------------------------
# 任务 30：中文 PII 检测验收测试
# ---------------------------------------------------------------------------


class TestChinesePIIAcceptance:
    """任务 30：中文 PII 检测端到端验收测试。"""

    @pytest.mark.parametrize(
        "phone",
        [
            pytest.param("13800138000", id="classic-138"),
            pytest.param("19100001234", id="new-191"),
            pytest.param("19912345678", id="new-199"),
        ],
    )
    async def test_tc_mask_cn_001_chinese_mobile_segments(
        self, masker_with_all_cn_recognizers, phone
    ):
        """TC-MASK-CN-001: 传统及 191/199 新号段均替换为 [PHONE_1]。"""
        result = await masker_with_all_cn_recognizers.mask(
            f"联系电话：{phone}", session_id="cn-001", request_id=phone
        )

        assert result["masked_text"] == "联系电话：[PHONE_1]"
        assert result["mapping"] == {"[PHONE_1]": phone}
        assert {entity["type"] for entity in result["entities_found"]} == {
            "CN_PHONE"
        }

    async def test_tc_mask_cn_002_id_card_with_x_checksum(
        self, masker_with_all_cn_recognizers
    ):
        """TC-MASK-CN-002: 末位 X 且校验合法的 18 位身份证替换为 [ID_CARD_1]。"""
        id_card = "11010119900307002X"
        result = await masker_with_all_cn_recognizers.mask(
            f"身份证号：{id_card}", session_id="cn-002", request_id="id-x"
        )

        assert result["masked_text"] == "身份证号：[ID_CARD_1]"
        assert result["mapping"] == {"[ID_CARD_1]": id_card}
        assert {entity["type"] for entity in result["entities_found"]} == {
            "CN_ID_CARD"
        }

    async def test_tc_mask_cn_003_common_chinese_name(
        self, masker_with_all_cn_recognizers
    ):
        """TC-MASK-CN-003: 常见姓氏开头的三字中文人名替换为 [PERSON_1]。"""
        result = await masker_with_all_cn_recognizers.mask(
            "姓名：张三丰", session_id="cn-003", request_id="name"
        )

        assert result["masked_text"] == "姓名：[PERSON_1]"
        assert result["mapping"] == {"[PERSON_1]": "张三丰"}
        assert {entity["type"] for entity in result["entities_found"]} == {
            "CN_NAME"
        }

    async def test_tc_mask_cn_004_name_in_long_sentence_context(
        self, masker_with_all_cn_recognizers
    ):
        """TC-MASK-CN-004: 长句上下文中的中文人名能够准确识别。"""
        text = "在昨天举行的项目复盘会上，研发负责人张三丰详细介绍了下一阶段计划。"
        result = await masker_with_all_cn_recognizers.mask(
            text, session_id="cn-004", request_id="long-context"
        )

        assert result["masked_text"] == (
            "在昨天举行的项目复盘会上，研发负责人[PERSON_1]详细介绍了下一阶段计划。"
        )
        assert result["mapping"] == {"[PERSON_1]": "张三丰"}

    async def test_tc_mask_cn_005_invalid_id_checksum_not_detected(
        self, masker_with_all_cn_recognizers
    ):
        """TC-MASK-CN-005: 校验位非法的身份证号不产生 ID_CARD 占位符。"""
        invalid_id = "110101199003070015"
        text = f"待核验身份证号：{invalid_id}"
        result = await masker_with_all_cn_recognizers.mask(
            text, session_id="cn-005", request_id="invalid-id"
        )

        assert result["masked_text"] == text
        assert not any(key.startswith("[ID_CARD_") for key in result["mapping"])
        assert not any(
            entity["type"] == "CN_ID_CARD" for entity in result["entities_found"]
        )

    async def test_tc_mask_cn_006_mixed_chinese_name_and_english_email(
        self, masker_with_all_cn_recognizers
    ):
        """TC-MASK-CN-006: 中英文混合 prompt 同时脱敏中文人名和英文邮箱。"""
        email = "zhang.sanfeng@example.com"
        text = f"请将张三丰的资料发送到 {email}。"
        result = await masker_with_all_cn_recognizers.mask(
            text, session_id="cn-006", request_id="mixed-language"
        )

        assert result["masked_text"] == "请将[PERSON_1]的资料发送到 [EMAIL_1]。"
        assert result["mapping"] == {
            "[PERSON_1]": "张三丰",
            "[EMAIL_1]": email,
        }
        assert {entity["type"] for entity in result["entities_found"]} == {
            "CN_NAME",
            "EMAIL_ADDRESS",
        }

    async def test_tc_mask_cn_007_phone_in_sms_dialogue(
        self, masker_with_all_cn_recognizers
    ):
        """TC-MASK-CN-007: 短信/对话格式文本中的手机号被正确提取。"""
        phone = "19912345678"
        text = f"【快递短信】客服：包裹已到站，请回复或致电 {phone}。"
        result = await masker_with_all_cn_recognizers.mask(
            text, session_id="cn-007", request_id="sms"
        )

        assert phone not in result["masked_text"]
        assert "[PHONE_1]" in result["masked_text"]
        assert result["mapping"]["[PHONE_1]"] == phone
        assert any(
            entity["type"] == "CN_PHONE" for entity in result["entities_found"]
        )