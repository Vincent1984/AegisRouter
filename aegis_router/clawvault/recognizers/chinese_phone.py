"""中国手机号识别器

支持 11 位中国大陆手机号识别 (1[3-9] 开头)，覆盖所有运营商号段。
"""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer


class ChinesePhoneRecognizer(PatternRecognizer):
    """中国手机号 Recognizer。

    识别 11 位中国大陆手机号码，支持 13x-19x 所有号段。

    Entity type: CN_PHONE
    Pattern: 1[3-9]\\d{9}
    """

    PATTERNS = [
        Pattern(
            name="chinese_mobile",
            regex=r"(?<!\d)1[3-9]\d{9}(?!\d)",
            score=0.7,
        ),
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="CN_PHONE",
            patterns=self.PATTERNS,
            name="ChinesePhoneRecognizer",
            supported_language="en",  # Presidio uses "en" as default language code
            context=["手机", "电话", "联系", "phone", "mobile", "tel"],
        )
