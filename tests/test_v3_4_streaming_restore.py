"""V3-4 验证检查点: stream=true 模式，Mock LLM 流式返回被切割的占位符 → 客户端收到正确还原的完整文本

验证 async_post_call_streaming_iterator_hook 的完整流程:
- Mock LLM 以流式 chunk 返回含占位符的响应
- 占位符可能被切割在多个 chunk 边界上
- StreamRehydrator 正确缓冲和还原分割的占位符
- 客户端最终收到完整还原的文本

测试场景:
1. 占位符被切割为 2 个 chunk
2. 占位符被切割为 3 个 chunk
3. 混合完整和分割的占位符
4. 无占位符的纯文本流
5. Bypass 模式 (ClawVault 不可用)
6. 空映射表 (无 PII 检测)
7. 流结束时 flush_remaining 正确执行
"""

from __future__ import annotations

import copy
import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aegis_router.callbacks.smart_router import SmartRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool


# ---------------------------------------------------------------------------
# Mock Streaming Chunk 数据类
# ---------------------------------------------------------------------------


@dataclass
class MockDelta:
    """Mock LiteLLM streaming delta object."""
    content: Optional[str] = None
    role: Optional[str] = None


@dataclass
class MockStreamChoice:
    """Mock LiteLLM streaming choice object."""
    delta: MockDelta
    index: int = 0


@dataclass
class MockStreamChunk:
    """Mock LiteLLM streaming chunk object."""
    choices: list = field(default_factory=list)


def make_stream_chunk(content: Optional[str] = None, role: Optional[str] = None) -> MockStreamChunk:
    """Create a mock streaming chunk with given content."""
    delta = MockDelta(content=content, role=role)
    choice = MockStreamChoice(delta=delta)
    return MockStreamChunk(choices=[choice])


async def async_iter_chunks(chunks: list[MockStreamChunk]):
    """Create an async iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """Create a mock ClawVaultPool."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock()
    pool.max_connections = 10
    return pool


@pytest.fixture
def callback(mock_pool):
    """Create a SmartRouterCallback with a mocked pool."""
    return SmartRouterCallback(pool=mock_pool)


def make_request_data(
    request_id: str = "req-v3-4",
    session_id: str = "sess-v3-4",
) -> dict:
    """Create request_data dict with metadata."""
    return {
        "metadata": {
            "request_id": request_id,
            "session_id": session_id,
        }
    }


async def collect_stream_content(async_gen) -> str:
    """Collect all text content from a streaming async generator."""
    texts = []
    async for chunk in async_gen:
        content = None
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                content = choice.delta.content
        if content:
            texts.append(content)
    return "".join(texts)


async def collect_stream_chunks(async_gen) -> list[MockStreamChunk]:
    """Collect all chunks from a streaming async generator."""
    chunks = []
    async for chunk in async_gen:
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# V3-4 验证: 占位符被切割为 2 个 chunk
# ---------------------------------------------------------------------------


class TestV3_4_SplitAcrossTwoChunks:
    """验证占位符被切割为 2 个 chunk 时能正确还原。"""

    async def test_person_placeholder_split_two_chunks(self, callback, mock_pool):
        """[PERSON_1] 被切割为 '[PER' + 'SON_1]' → 还原为 '张三'。"""
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "张三"}
        }

        chunks = [
            make_stream_chunk(content="你好 [PER"),
            make_stream_chunk(content="SON_1]，欢迎回来"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "你好 张三，欢迎回来"

    async def test_phone_placeholder_split_two_chunks(self, callback, mock_pool):
        """[PHONE_1] 被切割为 '[PHONE' + '_1]' → 还原为 '13800138000'。"""
        mock_pool.call.return_value = {
            "mapping": {"[PHONE_1]": "13800138000"}
        }

        chunks = [
            make_stream_chunk(content="请拨打 [PHONE"),
            make_stream_chunk(content="_1] 联系客服"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "请拨打 13800138000 联系客服"

    async def test_email_placeholder_split_at_bracket(self, callback, mock_pool):
        """[EMAIL_1] 被切割为 '[' + 'EMAIL_1]' → 还原为 'test@example.com'。"""
        mock_pool.call.return_value = {
            "mapping": {"[EMAIL_1]": "test@example.com"}
        }

        chunks = [
            make_stream_chunk(content="邮箱: ["),
            make_stream_chunk(content="EMAIL_1] 已确认"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "邮箱: test@example.com 已确认"


# ---------------------------------------------------------------------------
# V3-4 验证: 占位符被切割为 3 个 chunk
# ---------------------------------------------------------------------------


class TestV3_4_SplitAcrossThreeChunks:
    """验证占位符被切割为 3 个 chunk 时能正确还原。"""

    async def test_phone_placeholder_split_three_chunks(self, callback, mock_pool):
        """[PHONE_1] 被切割为 '[' + 'PHONE' + '_1] 已记录' → 还原为 '13800138000 已记录'。"""
        mock_pool.call.return_value = {
            "mapping": {"[PHONE_1]": "13800138000"}
        }

        chunks = [
            make_stream_chunk(content="["),
            make_stream_chunk(content="PHONE"),
            make_stream_chunk(content="_1] 已记录"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "13800138000 已记录"

    async def test_person_placeholder_split_three_chunks(self, callback, mock_pool):
        """[PERSON_1] 被切割为 '[PER' + 'SON' + '_1]你好' → 还原为 '张三你好'。"""
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "张三"}
        }

        chunks = [
            make_stream_chunk(content="[PER"),
            make_stream_chunk(content="SON"),
            make_stream_chunk(content="_1]你好"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "张三你好"

    async def test_id_card_placeholder_split_three_chunks(self, callback, mock_pool):
        """[ID_CARD_1] 被切割为 '[ID' + '_CARD' + '_1]' → 还原为身份证号。"""
        mock_pool.call.return_value = {
            "mapping": {"[ID_CARD_1]": "110101199003071234"}
        }

        chunks = [
            make_stream_chunk(content="身份证: [ID"),
            make_stream_chunk(content="_CARD"),
            make_stream_chunk(content="_1] 已验证"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "身份证: 110101199003071234 已验证"


# ---------------------------------------------------------------------------
# V3-4 验证: 混合完整和分割的占位符
# ---------------------------------------------------------------------------


class TestV3_4_MixedCompleteAndSplit:
    """验证混合完整和分割占位符在同一流中全部正确还原。"""

    async def test_complete_then_split_placeholder(self, callback, mock_pool):
        """一个完整占位符后跟一个分割占位符 → 全部还原。"""
        mock_pool.call.return_value = {
            "mapping": {
                "[PERSON_1]": "张三",
                "[PHONE_1]": "13800138000",
            }
        }

        chunks = [
            make_stream_chunk(content="[PERSON_1] 的电话是 [PHO"),
            make_stream_chunk(content="NE_1]，请联系。"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "张三 的电话是 13800138000，请联系。"

    async def test_split_then_complete_placeholder(self, callback, mock_pool):
        """一个分割占位符后跟一个完整占位符 → 全部还原。"""
        mock_pool.call.return_value = {
            "mapping": {
                "[PERSON_1]": "李四",
                "[EMAIL_1]": "lisi@test.com",
            }
        }

        chunks = [
            make_stream_chunk(content="[PER"),
            make_stream_chunk(content="SON_1] 的邮箱是 [EMAIL_1]"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "李四 的邮箱是 lisi@test.com"

    async def test_multiple_placeholders_mixed_splitting(self, callback, mock_pool):
        """三个占位符: 一个完整、一个 2-split、一个 3-split → 全部还原。"""
        mock_pool.call.return_value = {
            "mapping": {
                "[PERSON_1]": "王五",
                "[PHONE_1]": "13900139000",
                "[ID_CARD_1]": "320101199512120001",
            }
        }

        chunks = [
            make_stream_chunk(content="用户 [PERSON_1]，手机 [PHO"),
            make_stream_chunk(content="NE_1]，身份证 [ID"),
            make_stream_chunk(content="_CARD"),
            make_stream_chunk(content="_1] 已录入系统"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "用户 王五，手机 13900139000，身份证 320101199512120001 已录入系统"


# ---------------------------------------------------------------------------
# V3-4 验证: 无占位符的纯文本流
# ---------------------------------------------------------------------------


class TestV3_4_NoPlaceholders:
    """验证无占位符的纯文本流正确透传。"""

    async def test_plain_text_passes_through(self, callback, mock_pool):
        """纯文本 chunk 无占位符 → 内容原样通过。"""
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "张三"}
        }

        chunks = [
            make_stream_chunk(content="今天天气很好，"),
            make_stream_chunk(content="适合出门散步。"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "今天天气很好，适合出门散步。"

    async def test_text_with_brackets_but_no_placeholder(self, callback, mock_pool):
        """含方括号但非占位符格式的文本 → 原样通过。"""
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "张三"}
        }

        chunks = [
            make_stream_chunk(content="请参考 [参考文档] 中的"),
            make_stream_chunk(content="内容，第 [3] 章。"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "请参考 [参考文档] 中的内容，第 [3] 章。"


# ---------------------------------------------------------------------------
# V3-4 验证: Bypass 模式 (ClawVault 不可用)
# ---------------------------------------------------------------------------


class TestV3_4_BypassMode:
    """验证 ClawVault 不可用时 chunk 原样透传。"""

    async def test_mapping_returns_none_bypass(self, callback, mock_pool):
        """get_mapping 返回 None (ClawVault 不可用) → chunk 原样透传。"""
        mock_pool.call.return_value = None

        chunks = [
            make_stream_chunk(content="你好 [PER"),
            make_stream_chunk(content="SON_1]，欢迎"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        # Bypass mode: 占位符不会被还原，原样输出
        assert result == "你好 [PERSON_1]，欢迎"

    async def test_no_request_id_bypass(self, callback, mock_pool):
        """无 request_id → 直接透传所有 chunk。"""
        chunks = [
            make_stream_chunk(content="Hello "),
            make_stream_chunk(content="[PERSON_1]"),
        ]

        request_data = {"metadata": {}}  # No request_id
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "Hello [PERSON_1]"
        # get_mapping should NOT be called
        mock_pool.call.assert_not_called()


# ---------------------------------------------------------------------------
# V3-4 验证: 空映射表 (无 PII 检测)
# ---------------------------------------------------------------------------


class TestV3_4_EmptyMapping:
    """验证映射表为空时 chunk 原样透传。"""

    async def test_empty_mapping_passes_through(self, callback, mock_pool):
        """映射表为空 (无 PII 检测) → chunk 原样透传。"""
        mock_pool.call.return_value = {"mapping": {}}

        chunks = [
            make_stream_chunk(content="这是一条"),
            make_stream_chunk(content="普通消息。"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "这是一条普通消息。"

    async def test_empty_mapping_with_bracket_text(self, callback, mock_pool):
        """映射表为空但文本含占位符格式 → 原样透传（不还原）。"""
        mock_pool.call.return_value = {"mapping": {}}

        chunks = [
            make_stream_chunk(content="[PERSON_1] 的信息"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "[PERSON_1] 的信息"


# ---------------------------------------------------------------------------
# V3-4 验证: 流结束时 flush_remaining 正确执行
# ---------------------------------------------------------------------------


class TestV3_4_FlushRemaining:
    """验证流结束时 flush_remaining 被调用并输出剩余缓冲内容。"""

    async def test_flush_partial_placeholder_at_end(self, callback, mock_pool):
        """流结束时缓冲区仍有不完整占位符文本 → flush 输出。"""
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "张三"}
        }

        # 流在占位符中间结束（不完整的占位符，非法占位符文本原样输出）
        chunks = [
            make_stream_chunk(content="你好 [PER"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        # 不完整占位符在 flush 时原样输出（因为不匹配完整占位符模式）
        assert result == "你好 [PER"

    async def test_flush_complete_placeholder_buffered_at_end(self, callback, mock_pool):
        """最后一个 chunk 使缓冲区形成完整占位符 → flush 还原后输出。"""
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "张三"}
        }

        # 最后一个 chunk 仅包含占位符的结尾部分
        # chunk1: "你好 [PER" → 缓冲 "[PER"，输出 "你好 "
        # chunk2: "SON_1]" → 缓冲变为 "[PERSON_1]"，无尾部 partial → 全部输出
        chunks = [
            make_stream_chunk(content="你好 [PER"),
            make_stream_chunk(content="SON_1]"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        result = await collect_stream_content(stream)
        assert result == "你好 张三"

    async def test_role_only_chunks_pass_through(self, callback, mock_pool):
        """Role-only chunk (content=None) 原样通过不影响还原。"""
        mock_pool.call.return_value = {
            "mapping": {"[PERSON_1]": "张三"}
        }

        chunks = [
            make_stream_chunk(content=None, role="assistant"),  # role-only chunk
            make_stream_chunk(content="你好 [PERSON_1]"),
        ]

        request_data = make_request_data()
        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )

        collected = await collect_stream_chunks(stream)
        # First chunk (role-only) should pass through
        assert collected[0].choices[0].delta.role == "assistant"
        assert collected[0].choices[0].delta.content is None

        # Collect text content from remaining chunks
        texts = []
        for c in collected[1:]:
            content = c.choices[0].delta.content
            if content:
                texts.append(content)
        assert "".join(texts) == "你好 张三"

    async def test_get_mapping_called_with_correct_params(self, callback, mock_pool):
        """验证 get_mapping 被调用时携带正确的 request_id 和 session_id。"""
        mock_pool.call.return_value = {"mapping": {}}

        chunks = [make_stream_chunk(content="hello")]
        request_data = make_request_data(
            request_id="req-check-params",
            session_id="sess-check-params",
        )

        stream = callback.async_post_call_streaming_iterator_hook(
            {}, async_iter_chunks(chunks), request_data
        )
        await collect_stream_content(stream)

        mock_pool.call.assert_called_once_with(
            "get_mapping",
            {"request_id": "req-check-params", "session_id": "sess-check-params"},
        )
