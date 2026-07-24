"""占位符还原模块

从 Redis 读取 PII 映射表，将响应文本中的占位符 [TYPE_N] 还原为原始值。
支持非流式一次性还原和流式 chunk 逐块还原两种模式。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from aegis_router.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 完整占位符正则: [TYPE_N] 其中 TYPE 由大写字母和下划线组成, N 为数字
_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z][A-Z_]*_\d+\]")


# ---------------------------------------------------------------------------
# PIIRestorer 主类
# ---------------------------------------------------------------------------


class PIIRestorer:
    """PII 占位符还原器。

    从 Redis 获取映射表，将文本中的占位符替换回原始值。

    Parameters
    ----------
    redis_client : RedisClient
        Redis 客户端实例，用于读取映射表。
    """

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client

    async def restore(
        self,
        text: str,
        request_id: str,
        session_id: str | None = None,
    ) -> dict:
        """将文本中的占位符还原为真实值。

        Parameters
        ----------
        text : str
            含有占位符的文本（如 LLM 响应）。
        request_id : str
            请求 ID，用于从 Redis 获取映射。
        session_id : str | None
            会话 ID。如果提供，在请求级映射不存在时回退到会话级。

        Returns
        -------
        dict
            包含 ``restored_text`` 字段的结果字典。
        """
        # 获取映射表
        mapping = await self._redis.get_mapping(
            request_id=request_id,
            session_id=session_id,
        )

        # 执行占位符替换
        restored_text = self._replace_placeholders(text, mapping)

        logger.debug(
            "占位符还原完成: request_id=%s, 替换数=%d",
            request_id,
            self._count_replacements(text, restored_text),
        )

        return {"restored_text": restored_text}

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_placeholders(text: str, mapping: dict) -> str:
        """在文本中替换所有已知占位符。

        对映射中的每个占位符，执行全文替换。不在映射中的占位符保持原样。

        Parameters
        ----------
        text : str
            待还原文本。
        mapping : dict
            占位符 → 原始值映射表。

        Returns
        -------
        str
            还原后的文本。
        """
        if not mapping:
            return text

        result = text
        for placeholder, original_value in mapping.items():
            result = result.replace(placeholder, original_value)

        return result

    @staticmethod
    def _count_replacements(original: str, restored: str) -> int:
        """计算替换次数（简单估算，用于日志）。"""
        if original == restored:
            return 0
        # 统计原文中占位符数量作为近似值
        return len(_PLACEHOLDER_PATTERN.findall(original))
