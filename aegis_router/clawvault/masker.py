"""PII 脱敏模块 (集成 Presidio Analyzer + Anonymizer)

使用 Microsoft Presidio 进行 PII 实体检测，并以顺序占位符 [TYPE_N] 替换。
支持 session 级映射一致性，并通过 RedisClient 持久化映射表。
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from aegis_router.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认支持的实体类型
# ---------------------------------------------------------------------------

DEFAULT_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "CREDIT_CARD",
]

# 实体类型到占位符前缀的映射
_ENTITY_TO_PLACEHOLDER: dict[str, str] = {
    "PERSON": "PERSON",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "IP_ADDRESS": "IP",
    "CREDIT_CARD": "CREDIT_CARD",
    "CN_PHONE": "PHONE",
    "CN_ID_CARD": "ID_CARD",
    "CN_NAME": "PERSON",
}


def _get_placeholder_prefix(entity_type: str) -> str:
    """将 Presidio 实体类型映射为占位符前缀。"""
    return _ENTITY_TO_PLACEHOLDER.get(entity_type, entity_type)


# ---------------------------------------------------------------------------
# PIIMasker 主类
# ---------------------------------------------------------------------------


class PIIMasker:
    """PII 脱敏器，集成 Presidio Analyzer 和 Anonymizer。

    Parameters
    ----------
    redis_client : RedisClient | None
        Redis 客户端实例，用于持久化映射表。为 None 时跳过存储。
    language : str
        主要分析语言，默认 ``en``。
    nlp_model : str
        spaCy NLP 模型名称，默认 ``en_core_web_sm``。
    score_threshold : float
        实体检测置信度阈值，低于此值的结果将被过滤。
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        language: str = "en",
        nlp_model: str = "en_core_web_sm",
        score_threshold: float = 0.4,
        entities: list[str] | None = None,
    ) -> None:
        self._redis = redis_client
        self._language = language
        self._score_threshold = score_threshold
        self._entities = entities or list(DEFAULT_ENTITIES)

        # 构建 NLP 引擎
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": nlp_model}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

        # 创建 Analyzer 引擎
        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[language],
        )

        # 创建 Anonymizer 引擎
        self._anonymizer = AnonymizerEngine()

        logger.info(
            "PIIMasker 初始化完成 (language=%s, model=%s, threshold=%.2f)",
            language,
            nlp_model,
            score_threshold,
        )

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def register_recognizer(self, recognizer: Any) -> None:
        """注册自定义 Recognizer 到 Analyzer 引擎。

        Parameters
        ----------
        recognizer : EntityRecognizer
            Presidio EntityRecognizer 子类实例。
        """
        self._analyzer.registry.add_recognizer(recognizer)
        # 将新实体类型添加到扫描列表
        if hasattr(recognizer, "supported_entities"):
            for entity_type in recognizer.supported_entities:
                if entity_type not in self._entities:
                    self._entities.append(entity_type)
        logger.info("已注册自定义 Recognizer: %s", type(recognizer).__name__)

    async def mask(self, text: str, session_id: str, request_id: str) -> dict:
        """对文本进行 PII 脱敏。

        Parameters
        ----------
        text : str
            待脱敏的原始文本。
        session_id : str
            会话 ID，用于 session 级映射一致性。
        request_id : str
            请求 ID，用于请求级映射存储。

        Returns
        -------
        dict
            包含以下字段:
            - ``masked_text``: 脱敏后文本
            - ``entities_found``: 检测到的实体列表
            - ``mapping``: 占位符到原始值的映射
        """
        # 1. 使用 Presidio Analyzer 检测 PII 实体
        results = self._analyzer.analyze(
            text=text,
            language=self._language,
            entities=self._entities,
            score_threshold=self._score_threshold,
        )

        # 如果没有检测到 PII，直接返回
        if not results:
            return {
                "masked_text": text,
                "entities_found": [],
                "mapping": {},
            }

        # 2. 去重 + 按位置排序 (从后往前替换避免位置偏移)
        results = self._deduplicate_results(results)
        results.sort(key=lambda r: r.start)

        # 3. 读取会话映射后生成占位符，确保跨 request 复用。
        # Redis 不可用时仍完成当前请求的脱敏，但无法保证跨请求一致性。
        session_mapping: dict[str, str] = {}
        if self._redis:
            try:
                session_mapping = await self._redis.get_mapping(
                    request_id=request_id,
                    session_id=session_id,
                )
            except Exception as e:
                logger.error("Redis 会话映射读取失败: %s", e)

        masked_text, mapping, entities_found = self._apply_placeholders(
            text,
            results,
            existing_mapping=session_mapping,
            session_id=session_id if self._redis else None,
        )

        # 4. 存储当前请求实际使用的映射，并合并到会话映射。
        if self._redis and mapping:
            try:
                await self._redis.store_mapping(session_id, request_id, mapping)
                await self._redis.update_session_mapping(session_id, mapping)
            except Exception as e:
                logger.error("Redis 映射存储失败: %s", e)

        return {
            "masked_text": masked_text,
            "entities_found": entities_found,
            "mapping": mapping,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _deduplicate_results(
        self, results: list[RecognizerResult]
    ) -> list[RecognizerResult]:
        """去除重叠的检测结果，保留置信度最高的。"""
        if not results:
            return []

        # 按 score 降序排列，优先保留高置信度结果
        sorted_results = sorted(results, key=lambda r: -r.score)
        kept: list[RecognizerResult] = []

        for result in sorted_results:
            # 检查是否与已保留的结果重叠
            overlaps = False
            for kept_result in kept:
                if result.start < kept_result.end and result.end > kept_result.start:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(result)

        return kept

    @staticmethod
    def _session_counter_base(session_id: str | None) -> int:
        """返回会话专属的数字命名空间起点。

        占位符格式仍保持 ``[TYPE_N]``，但使用 session_id 的稳定摘要为
        每个会话分配独立数字区间，避免不同会话对相同 PII 产生可关联的
        相同占位符。无 Redis 的独立调用继续从 1 开始编号。
        """
        if session_id is None:
            return 0
        digest = hashlib.sha256(session_id.encode("utf-8")).digest()
        return int.from_bytes(digest, byteorder="big") * 1_000_000

    def _apply_placeholders(
        self,
        text: str,
        results: list[RecognizerResult],
        existing_mapping: dict[str, str] | None = None,
        session_id: str | None = None,
    ) -> tuple[str, dict, list[dict]]:
        """为检测到的实体生成顺序占位符并替换。

        当前请求中的相同值复用同一占位符；已有会话映射中的值也会复用
        原占位符。新占位符从会话专属数字区间分配，以保持 session 隔离。

        Returns
        -------
        tuple
            (masked_text, mapping, entities_found)
        """
        counter_base = self._session_counter_base(session_id)
        type_counters: dict[str, int] = defaultdict(lambda: counter_base)
        existing_mapping = existing_mapping or {}

        # 原始值 → 占位符，用于会话内及当前请求内复用。
        value_to_placeholder: dict[str, str] = {
            original: placeholder
            for placeholder, original in existing_mapping.items()
        }
        # 仅包含当前请求实际使用的占位符，供 request 级还原使用。
        mapping: dict[str, str] = {}
        entities_found: list[dict] = []

        # 避免新分配的编号与会话中的已有编号冲突。
        for placeholder in existing_mapping:
            if not (placeholder.startswith("[") and placeholder.endswith("]")):
                continue
            inner = placeholder[1:-1]
            prefix, separator, raw_counter = inner.rpartition("_")
            if separator and raw_counter.isdigit():
                type_counters[prefix] = max(type_counters[prefix], int(raw_counter))

        sorted_results = sorted(results, key=lambda r: r.start)
        replacements: list[tuple[int, int, str]] = []

        for result in sorted_results:
            original_value = text[result.start: result.end]
            placeholder_prefix = _get_placeholder_prefix(result.entity_type)

            if original_value in value_to_placeholder:
                placeholder = value_to_placeholder[original_value]
            else:
                type_counters[placeholder_prefix] += 1
                counter = type_counters[placeholder_prefix]
                placeholder = f"[{placeholder_prefix}_{counter}]"
                value_to_placeholder[original_value] = placeholder

            mapping[placeholder] = original_value
            replacements.append((result.start, result.end, placeholder))
            entities_found.append({
                "type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
            })

        # 从后往前替换，避免位置偏移。
        masked_text = text
        for start, end, placeholder in reversed(replacements):
            masked_text = masked_text[:start] + placeholder + masked_text[end:]

        return masked_text, mapping, entities_found
