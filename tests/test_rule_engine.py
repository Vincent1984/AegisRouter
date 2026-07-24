"""规则前置引擎测试 — 寒暄词库匹配与路由决策"""

import pytest

from aegis_router.config import TrivialConfig
from aegis_router.router.rule_engine import RuleEngine, RuleEngineResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patterns_file(tmp_path):
    """创建临时寒暄词库文件。"""
    content = (
        "# 寒暄词库\n"
        "你好\n"
        "hello\n"
        "hi\n"
        "hey\n"
        "谢谢\n"
        "再见\n"
        "早上好\n"
        "晚上好\n"
        "good morning\n"
        "good evening\n"
        "good night\n"
        "thanks\n"
        "thank you\n"
        "bye\n"
        "goodbye\n"
        "嗨\n"
        "哈喽\n"
        "晚安\n"
        "早安\n"
        "你好呀\n"
    )
    f = tmp_path / "trivial_chat.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)


@pytest.fixture
def rule_engine(patterns_file):
    """创建启用的规则前置引擎实例。"""
    config = TrivialConfig(
        enabled=True,
        max_length=30,
        target_model="local-7b",
        patterns_file=patterns_file,
    )
    return RuleEngine(config)


# ---------------------------------------------------------------------------
# TC-ROUTE-RULE-001: 短寒暄 ("你好") → 路由到 local-7b
# ---------------------------------------------------------------------------

class TestTrivialGreeting:
    """TC-ROUTE-RULE-001: 短寒暄直接路由到本地小模型。"""

    def test_chinese_greeting_matches(self, rule_engine):
        """'你好' 应命中规则前置，路由到 local-7b。"""
        result = rule_engine.check("你好")
        assert result.matched is True
        assert result.target_model == "local-7b"
        assert result.matched_pattern == "你好"


# ---------------------------------------------------------------------------
# TC-ROUTE-RULE-002: 长文本 (>30字) 即使含寒暄词 → 不走规则前置
# ---------------------------------------------------------------------------

class TestLongTextBypass:
    """TC-ROUTE-RULE-002: 超过 max_length 的文本不走规则前置。"""

    def test_long_text_with_greeting_not_matched(self, rule_engine):
        """超过 30 字符的文本即使包含 '你好' 也不命中规则前置。"""
        # 构造一个超过 30 字符且包含 "你好" 的文本
        long_prompt = "你好，我想请你帮我详细解释一下深度学习中反向传播算法的工作原理和数学推导过程"
        assert len(long_prompt.strip()) > 30

        result = rule_engine.check(long_prompt)
        assert result.matched is False
        assert result.target_model is None


# ---------------------------------------------------------------------------
# TC-ROUTE-RULE-003: 英文寒暄 ("hello", "hi", "thanks") → 路由到 local-7b
# ---------------------------------------------------------------------------

class TestEnglishGreeting:
    """TC-ROUTE-RULE-003: 英文寒暄词也应命中规则前置。"""

    @pytest.mark.parametrize("prompt,expected_pattern", [
        ("hello", "hello"),
        ("hi", "hi"),
        ("thanks", "thanks"),
    ])
    def test_english_greetings_match(self, rule_engine, prompt, expected_pattern):
        """英文寒暄词应命中规则前置，路由到 local-7b。"""
        result = rule_engine.check(prompt)
        assert result.matched is True
        assert result.target_model == "local-7b"
        assert result.matched_pattern == expected_pattern


# ---------------------------------------------------------------------------
# TC-ROUTE-RULE-004: 非寒暄短文本 ("解释量子计算") → 不走规则前置，进入分类器
# ---------------------------------------------------------------------------

class TestNonGreetingShortText:
    """TC-ROUTE-RULE-004: 短文本但非寒暄词不命中规则前置。"""

    def test_short_non_greeting_not_matched(self, rule_engine):
        """'解释量子计算' 虽然是短文本，但不是寒暄词，不命中规则前置。"""
        result = rule_engine.check("解释量子计算")
        assert result.matched is False
        assert result.target_model is None
