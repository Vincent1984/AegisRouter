"""自定义 Presidio Recognizer

提供中文 PII 识别器:
- ChinesePhoneRecognizer: 中国手机号
- ChineseIdCardRecognizer: 中国身份证号
- ChineseNameRecognizer: 中文人名 (spaCy NER + 百家姓增强)
"""

from aegis_router.clawvault.recognizers.chinese_id_card import (
    ChineseIdCardRecognizer,
    validate_id_card_checksum,
)
from aegis_router.clawvault.recognizers.chinese_name import ChineseNameRecognizer
from aegis_router.clawvault.recognizers.chinese_phone import ChinesePhoneRecognizer

__all__ = [
    "ChinesePhoneRecognizer",
    "ChineseIdCardRecognizer",
    "ChineseNameRecognizer",
    "validate_id_card_checksum",
]
