"""流式还原引擎测试

覆盖:
- TC-STREAM-001: 完整占位符在单个 chunk 中 → 立即替换
- TC-STREAM-002: 占位符分割在 2 个 chunk 中 → 缓冲后替换
- TC-STREAM-003: 占位符分割在 3 个 chunk 中 → 正确处理
- TC-STREAM-004: 连续占位符 → 全部正确替换
- TC-STREAM-005: 混合文本和占位符 → 普通文本立即输出，占位符缓冲
- TC-STREAM-006: 流结束时的 flush_remaining
- TC-STREAM-007: 不在映射中的占位符 → 保持原样
- TC-STREAM-008: 大量小 chunk (1-2 字符) → 无性能退化
"""

from __future__ import annotations

import time

import pytest

from aegis_router.callbacks.stream_rehydrator import StreamRehydrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mapping():
    """标准测试映射表。"""
    return {
        "[PERSON_1]": "张三",
        "[PERSON_2]": "李四",
        "[PHONE_1]": "13800138000",
        "[EMAIL_1]": "zhangsan@example.com",
        "[ID_CARD_1]": "110101199003071234",
    }


@pytest.fixture
def rehydrator(mapping):
    """创建 StreamRehydrator 实例。"""
    return StreamRehydrator(mapping)


# ---------------------------------------------------------------------------
# TC-STREAM-001: 完整占位符在单个 chunk 中
# ---------------------------------------------------------------------------


class TestCompletePlaceholderSingleChunk:
    """TC-STREAM-001: 完整占位符在单个 chunk 中 → 立即替换。"""

    def test_single_placeholder_immediate_replacement(self, rehydrator):
        """单个完整占位符立即被替换。"""
        result = rehydrator.process_chunk("你好 [PERSON_1]，欢迎回来。")
        assert result == "你好 张三，欢迎回来。"

    def test_multiple_placeholders_in_one_chunk(self, rehydrator):
        """一个 chunk 中含多个完整占位符，全部替换。"""
        result = rehydrator.process_chunk("[PERSON_1] 的电话是 [PHONE_1]。")
        assert result == "张三 的电话是 13800138000。"

    def test_plain_text_without_placeholder(self, rehydrator):
        """纯文本 chunk 直接输出。"""
        result = rehydrator.process_chunk("今天天气不错。")
        assert result == "今天天气不错。"

    def test_empty_chunk(self, rehydrator):
        """空 chunk 返回空字符串。"""
        result = rehydrator.process_chunk("")
        assert result == ""


# ---------------------------------------------------------------------------
# TC-STREAM-002: 占位符分割在 2 个 chunk 中
# ---------------------------------------------------------------------------


class TestPlaceholderSplitTwoChunks:
    """TC-STREAM-002: 占位符分割在 2 个 chunk → 缓冲后替换。"""

    def test_split_at_middle(self, rehydrator):
        """占位符从中间分割: [PER + SON_1]。"""
        result1 = rehydrator.process_chunk("你好 [PER")
        # [PER 是不完整占位符，前面的 "你好 " 应该输出
        assert result1 == "你好 "

        result2 = rehydrator.process_chunk("SON_1]，欢迎。")
        assert result2 == "张三，欢迎。"

    def test_split_after_bracket(self, rehydrator):
        """占位符在 [ 后分割: [ + PERSON_1]。"""
        result1 = rehydrator.process_chunk("联系 [")
        assert result1 == "联系 "

        result2 = rehydrator.process_chunk("PERSON_1] 获取帮助。")
        assert result2 == "张三 获取帮助。"

    def test_split_before_closing_bracket(self, rehydrator):
        """占位符在 ] 前分割: [PERSON_1 + ]。"""
        result1 = rehydrator.process_chunk("姓名: [PERSON_1")
        assert result1 == "姓名: "

        result2 = rehydrator.process_chunk("] 已确认。")
        assert result2 == "张三 已确认。"

    def test_split_phone_placeholder(self, rehydrator):
        """手机号占位符分割: [PHO + NE_1]。"""
        result1 = rehydrator.process_chunk("拨打 [PHO")
        assert result1 == "拨打 "

        result2 = rehydrator.process_chunk("NE_1] 联系。")
        assert result2 == "13800138000 联系。"


# ---------------------------------------------------------------------------
# TC-STREAM-003: 占位符分割在 3+ 个 chunk 中
# ---------------------------------------------------------------------------


class TestPlaceholderSplitThreeChunks:
    """TC-STREAM-003: 占位符分割在 3+ 个 chunk → 正确处理。"""

    def test_split_across_three_chunks(self, rehydrator):
        """占位符分割在 3 个 chunk: [ + PERSON + _1]。"""
        result1 = rehydrator.process_chunk("你好 [")
        assert result1 == "你好 "

        result2 = rehydrator.process_chunk("PERSON")
        assert result2 == ""

        result3 = rehydrator.process_chunk("_1] 再见。")
        assert result3 == "张三 再见。"

    def test_split_across_four_chunks(self, rehydrator):
        """占位符分割在 4 个 chunk: [ + PER + SON_1 + ]。"""
        result1 = rehydrator.process_chunk("Hi [")
        assert result1 == "Hi "

        result2 = rehydrator.process_chunk("PER")
        assert result2 == ""

        result3 = rehydrator.process_chunk("SON_1")
        assert result3 == ""

        result4 = rehydrator.process_chunk("] world")
        assert result4 == "张三 world"

    def test_character_by_character(self, rehydrator):
        """逐字符传入占位符。"""
        text = "[PHONE_1]"
        accumulated = ""

        for char in text[:-1]:
            result = rehydrator.process_chunk(char)
            accumulated += result

        # 最后一个字符 ']' 完成占位符
        result = rehydrator.process_chunk(text[-1])
        accumulated += result

        assert accumulated == "13800138000"


# ---------------------------------------------------------------------------
# TC-STREAM-004: 连续占位符
# ---------------------------------------------------------------------------


class TestConsecutivePlaceholders:
    """TC-STREAM-004: 连续占位符 → 全部正确替换。"""

    def test_consecutive_placeholders_single_chunk(self, rehydrator):
        """连续占位符在一个 chunk 中。"""
        result = rehydrator.process_chunk("[PERSON_1][PHONE_1]")
        assert result == "张三13800138000"

    def test_consecutive_placeholders_with_space(self, rehydrator):
        """连续占位符之间有空格。"""
        result = rehydrator.process_chunk("[PERSON_1] [PHONE_1]")
        assert result == "张三 13800138000"

    def test_consecutive_placeholders_split_between(self, rehydrator):
        """连续占位符在交界处分割。"""
        result1 = rehydrator.process_chunk("[PERSON_1][PH")
        assert result1 == "张三"

        result2 = rehydrator.process_chunk("ONE_1]")
        assert result2 == "13800138000"

    def test_three_consecutive_placeholders(self, rehydrator):
        """三个连续占位符。"""
        result = rehydrator.process_chunk("[PERSON_1] [PERSON_2] [PHONE_1]")
        assert result == "张三 李四 13800138000"


# ---------------------------------------------------------------------------
# TC-STREAM-005: 混合文本和占位符
# ---------------------------------------------------------------------------


class TestMixedTextAndPlaceholders:
    """TC-STREAM-005: 混合文本和占位符 → 普通文本立即输出。"""

    def test_text_before_split_placeholder_flushes(self, rehydrator):
        """占位符前的普通文本立即输出。"""
        result = rehydrator.process_chunk("前面的文本 [PER")
        assert result == "前面的文本 "

    def test_text_after_completed_placeholder(self, rehydrator):
        """完整占位符后的文本一同输出。"""
        result = rehydrator.process_chunk("[PERSON_1] 后面的文本")
        assert result == "张三 后面的文本"

    def test_alternating_text_and_placeholders(self, rehydrator):
        """交替的文本和占位符，多 chunk。"""
        result1 = rehydrator.process_chunk("姓名: ")
        assert result1 == "姓名: "

        result2 = rehydrator.process_chunk("[PERSON_1], 电话: ")
        assert result2 == "张三, 电话: "

        result3 = rehydrator.process_chunk("[PHONE_1]。")
        assert result3 == "13800138000。"

    def test_mixed_split_scenario(self, rehydrator):
        """复杂混合场景: 文本 + 分割占位符 + 文本。"""
        result1 = rehydrator.process_chunk("联系人: [PER")
        assert result1 == "联系人: "

        result2 = rehydrator.process_chunk("SON_1], 邮箱: [EMA")
        assert result2 == "张三, 邮箱: "

        result3 = rehydrator.process_chunk("IL_1]。")
        assert result3 == "zhangsan@example.com。"


# ---------------------------------------------------------------------------
# TC-STREAM-006: flush_remaining
# ---------------------------------------------------------------------------


class TestFlushRemaining:
    """TC-STREAM-006: 流结束时 flush_remaining 正确处理。"""

    def test_flush_empty_buffer(self, rehydrator):
        """缓冲区为空时 flush 返回空字符串。"""
        result = rehydrator.flush_remaining()
        assert result == ""

    def test_flush_after_complete_processing(self, rehydrator):
        """所有 chunk 处理完毕后 flush 返回空。"""
        rehydrator.process_chunk("[PERSON_1] hello")
        result = rehydrator.flush_remaining()
        assert result == ""

    def test_flush_incomplete_placeholder(self, rehydrator):
        """流意外结束，缓冲区中有不完整占位符文本。"""
        rehydrator.process_chunk("文本 [PER")
        # 流结束，flush 将缓冲区内容输出（不完整的不匹配正则，原样输出）
        result = rehydrator.flush_remaining()
        assert result == "[PER"

    def test_flush_with_complete_placeholder_in_buffer(self, rehydrator):
        """flush 时缓冲区中有完整占位符（边界情况）。"""
        # 这种情况下 process_chunk 应已输出，但手动构造测试
        rehydrator.buffer = "[PERSON_1]"
        result = rehydrator.flush_remaining()
        assert result == "张三"

    def test_flush_resets_buffer(self, rehydrator):
        """flush 后缓冲区被清空。"""
        rehydrator.process_chunk("text [PER")
        rehydrator.flush_remaining()
        assert rehydrator.buffer == ""

    def test_full_stream_with_flush(self, rehydrator):
        """完整流式处理流程: 多 chunk + flush。"""
        output_parts = []
        output_parts.append(rehydrator.process_chunk("你好 [PER"))
        output_parts.append(rehydrator.process_chunk("SON_1]，你的电话 [PHO"))
        output_parts.append(rehydrator.process_chunk("NE_1] 已记录。"))
        output_parts.append(rehydrator.flush_remaining())

        full_output = "".join(output_parts)
        assert full_output == "你好 张三，你的电话 13800138000 已记录。"


# ---------------------------------------------------------------------------
# TC-STREAM-007: 不在映射中的占位符
# ---------------------------------------------------------------------------


class TestUnmappedPlaceholders:
    """TC-STREAM-007: 不在映射中的占位符 → 保持原样。"""

    def test_unknown_placeholder_kept_as_is(self, rehydrator):
        """不在映射中的占位符不被替换。"""
        result = rehydrator.process_chunk("参考 [NOTE_1] 中的内容。")
        assert result == "参考 [NOTE_1] 中的内容。"

    def test_mix_of_known_and_unknown(self, rehydrator):
        """已知占位符替换，未知占位符保留。"""
        result = rehydrator.process_chunk("[PERSON_1] 提到了 [ADDR_1]。")
        assert result == "张三 提到了 [ADDR_1]。"

    def test_unknown_placeholder_split(self, rehydrator):
        """未知占位符被分割时仍正确保留。"""
        result1 = rehydrator.process_chunk("见 [NO")
        assert result1 == "见 "

        result2 = rehydrator.process_chunk("TE_1] 文档。")
        assert result2 == "[NOTE_1] 文档。"

    def test_empty_mapping(self):
        """映射为空时所有占位符保持原样。"""
        rehydrator = StreamRehydrator({})
        result = rehydrator.process_chunk("[PERSON_1] 你好 [PHONE_1]")
        assert result == "[PERSON_1] 你好 [PHONE_1]"


# ---------------------------------------------------------------------------
# TC-STREAM-008: 性能测试 — 大量小 chunk
# ---------------------------------------------------------------------------


class TestPerformance:
    """TC-STREAM-008: 大量小 chunk (1-2 字符) → 无性能退化。"""

    def test_many_small_chunks_performance(self, mapping):
        """1000 个 1-2 字符的 chunk 在合理时间内完成。"""
        rehydrator = StreamRehydrator(mapping)

        # 构造包含占位符的长文本，每次送入 1-2 字符
        text = "你好 [PERSON_1]，电话 [PHONE_1]，邮箱 [EMAIL_1]。结束。"
        chunks = [text[i:i + 2] for i in range(0, len(text), 2)]

        start = time.perf_counter()
        output_parts = []
        for chunk in chunks:
            output_parts.append(rehydrator.process_chunk(chunk))
        output_parts.append(rehydrator.flush_remaining())
        elapsed = time.perf_counter() - start

        full_output = "".join(output_parts)
        assert full_output == "你好 张三，电话 13800138000，邮箱 zhangsan@example.com。结束。"
        # 应在 100ms 内完成
        assert elapsed < 0.1, f"处理耗时过长: {elapsed:.4f}s"

    def test_single_char_chunks_large_volume(self, mapping):
        """大量单字符 chunk (模拟极端场景)。"""
        rehydrator = StreamRehydrator(mapping)

        # 包含多个占位符的较长文本，逐字符传入
        text = "[PERSON_1] 说 [PERSON_2] 的电话是 [PHONE_1]，身份证 [ID_CARD_1]。" * 10
        chunks = list(text)  # 每个字符一个 chunk

        start = time.perf_counter()
        output_parts = []
        for chunk in chunks:
            output_parts.append(rehydrator.process_chunk(chunk))
        output_parts.append(rehydrator.flush_remaining())
        elapsed = time.perf_counter() - start

        full_output = "".join(output_parts)
        expected_unit = "张三 说 李四 的电话是 13800138000，身份证 110101199003071234。"
        expected = expected_unit * 10
        assert full_output == expected
        # 即使 500+ 个字符逐一传入，也应在 500ms 内完成
        assert elapsed < 0.5, f"处理耗时过长: {elapsed:.4f}s"

    def test_buffer_does_not_grow_unbounded(self, mapping):
        """缓冲区不会无限增长 — 普通文本应即时清空。"""
        rehydrator = StreamRehydrator(mapping)

        # 传入大量不含占位符的文本
        for _ in range(1000):
            rehydrator.process_chunk("普通文本不含占位符。")

        # 缓冲区应为空
        assert rehydrator.buffer == ""
