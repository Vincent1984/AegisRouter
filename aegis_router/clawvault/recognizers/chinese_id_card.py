"""中国身份证号识别器

支持 18 位二代身份证号识别，包含正则匹配 + 校验位验证。
"""

from __future__ import annotations

from typing import Optional

from presidio_analyzer import Pattern, PatternRecognizer

# 身份证校验位权重因子
_WEIGHT_FACTORS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]

# 校验码映射表 (mod 11 余数 → 校验码)
_CHECK_CODE_MAP = "10X98765432"


def validate_id_card_checksum(id_number: str) -> bool:
    """验证 18 位身份证号的校验位。

    算法:
    1. 将前 17 位数字分别乘以权重因子
    2. 对加权和取模 11
    3. 查表得到期望校验码
    4. 比较期望校验码与实际第 18 位

    Parameters
    ----------
    id_number : str
        18 位身份证号字符串。

    Returns
    -------
    bool
        校验位是否合法。
    """
    if len(id_number) != 18:
        return False

    # 前 17 位必须全部是数字
    digits_17 = id_number[:17]
    if not digits_17.isdigit():
        return False

    # 计算加权和
    weighted_sum = sum(
        int(digits_17[i]) * _WEIGHT_FACTORS[i] for i in range(17)
    )

    # 取模 11 得到校验码
    remainder = weighted_sum % 11
    expected_check = _CHECK_CODE_MAP[remainder]

    # 比较 (不区分大小写)
    actual_check = id_number[17].upper()
    return actual_check == expected_check


class ChineseIdCardRecognizer(PatternRecognizer):
    """中国身份证号 Recognizer。

    使用正则匹配 18 位身份证号格式，并通过校验位验证排除误报。

    Entity type: CN_ID_CARD
    Pattern: [1-9]\\d{5}(19|20)\\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])\\d{3}[\\dXx]
    """

    PATTERNS = [
        Pattern(
            name="chinese_id_card_18",
            regex=(
                r"(?<!\d)"
                r"[1-9]\d{5}"                      # 6 位地区码
                r"(19|20)\d{2}"                     # 4 位年份 (19xx/20xx)
                r"(0[1-9]|1[0-2])"                  # 2 位月份
                r"(0[1-9]|[12]\d|3[01])"            # 2 位日期
                r"\d{3}"                            # 3 位顺序码
                r"[\dXx]"                           # 1 位校验码
                r"(?!\d)"
            ),
            score=0.5,  # 初始分数较低，通过校验位验证后提升
        ),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="CN_ID_CARD",
            patterns=self.PATTERNS,
            name="ChineseIdCardRecognizer",
            supported_language="en",
            context=["身份证", "身份", "证件", "id card", "identity"],
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        """对正则匹配结果进行校验位验证。

        Parameters
        ----------
        pattern_text : str
            正则匹配到的文本。

        Returns
        -------
        Optional[bool]
            True — 校验通过，提高置信度;
            False — 校验失败，丢弃结果;
            None — 无法判断。
        """
        return validate_id_card_checksum(pattern_text)


