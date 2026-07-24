"""流式还原引擎

解决 SSE 流式传输中占位符跨 chunk 分割的问题。
使用缓冲机制确保占位符在被完整接收后才进行替换还原。

示例场景:
- chunk1: "你好 [PER"
- chunk2: "SON_1]，欢迎"
→ 输出: "你好 张三，欢迎"
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 完整占位符正则: [TYPE_N] 其中 TYPE 由大写字母和下划线组成, N 为数字
# 与 aegis_router/clawvault/restorer.py 中的 _PLACEHOLDER_PATTERN 保持一致
_PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z][A-Z_]*_\d+\]")

# 尾部不完整占位符正则: 检测 buffer 末尾可能的不完整占位符
# 匹配以 [ 开头、后跟大写字母/下划线/数字但缺少 ] 的尾部片段
_PARTIAL_PATTERN = re.compile(r"\[[A-Z_\d]*$")


# ---------------------------------------------------------------------------
# StreamRehydrator
# ---------------------------------------------------------------------------


class StreamRehydrator:
    """流式占位符还原器。

    在 SSE 流式传输中，LLM 响应以小 chunk 逐步到达。
    占位符 (如 [PERSON_1]) 可能被切分在多个 chunk 中。
    本类通过缓冲机制确保正确还原。

    Parameters
    ----------
    mapping : dict
        占位符 → 原始值映射表，如 {"[PERSON_1]": "张三", "[PHONE_1]": "13800138000"}

    使用示例
    --------
    >>> rehydrator = StreamRehydrator({"[PERSON_1]": "张三"})
    >>> rehydrator.process_chunk("你好 [PERSON_1]")
    '你好 张三'
    >>> rehydrator.flush_remaining()
    ''
    """

    def __init__(self, mapping: dict) -> None:
        self.mapping = mapping
        self.buffer = ""

    def process_chunk(self, chunk_text: str) -> str:
        """处理单个流式 chunk，返回可安全输出的文本。

        将 chunk 追加到内部缓冲区，检测尾部是否存在不完整占位符。
        完整的部分执行占位符替换后输出，不完整的部分保留在缓冲区中等待后续 chunk。

        Parameters
        ----------
        chunk_text : str
            当前接收到的文本 chunk。

        Returns
        -------
        str
            可安全输出的已还原文本（可能为空字符串，表示所有内容仍在缓冲中）。
        """
        self.buffer += chunk_text

        # 检测尾部是否有不完整占位符
        partial_match = _PARTIAL_PATTERN.search(self.buffer)

        if partial_match:
            # 尾部存在不完整占位符 → 保留在缓冲区
            safe_part = self.buffer[:partial_match.start()]
            self.buffer = self.buffer[partial_match.start():]
        else:
            # 无不完整占位符 → 全部安全输出
            safe_part = self.buffer
            self.buffer = ""

        # 对安全部分执行完整占位符替换
        restored = _PLACEHOLDER_PATTERN.sub(
            lambda m: self.mapping.get(m.group(), m.group()),
            safe_part,
        )

        return restored

    def flush_remaining(self) -> str:
        """刷新缓冲区中剩余的内容。

        在流结束时调用，将缓冲区中所有内容（包括可能的不完整占位符文本）
        进行占位符替换后输出。

        Returns
        -------
        str
            缓冲区中剩余文本的还原结果。
        """
        restored = _PLACEHOLDER_PATTERN.sub(
            lambda m: self.mapping.get(m.group(), m.group()),
            self.buffer,
        )
        self.buffer = ""
        return restored
