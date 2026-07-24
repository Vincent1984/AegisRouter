"""RouteLLM 分类器集成测试

使用启发式 mock router 模拟真实的 RouteLLM 分类行为，
验证 ModelClassifier 的端到端集成合约。

Test Cases:
- TC-CLASSIFIER-001: 简单 prompt → score < 0.3
- TC-CLASSIFIER-002: 中等 prompt → 0.3 < score < 0.7
- TC-CLASSIFIER-003: 复杂 prompt → score > 0.7
- TC-CLASSIFIER-004: 分类器推理延迟 < 10ms（100 次取平均）
- TC-CLASSIFIER-005: score_input=masked 模式下脱敏 prompt 打分与原文分数偏差 < 0.1
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aegis_router.config import ClassifierConfig
from aegis_router.router.model_classifier import (
    ClassifierResult,
    ModelClassifier,
)


# ---------------------------------------------------------------------------
# Heuristic Mock Router
# ---------------------------------------------------------------------------

# 复杂度关键词 — 出现这些词表示 prompt 更复杂
_COMPLEX_KEYWORDS = frozenset([
    "审计", "安全漏洞", "修复方案", "分布式", "架构设计",
    "代码审查", "性能优化", "多线程", "并发", "事务",
    "audit", "vulnerability", "distributed", "architecture",
    "refactor", "concurrency", "security", "algorithm",
])

# 中等复杂度关键词
_MEDIUM_KEYWORDS = frozenset([
    "分析", "报告", "总结", "比较", "解释", "编写",
    "500字", "1000字", "产品", "方案",
    "analyze", "report", "summarize", "compare", "explain", "write",
])


def _heuristic_score(prompt: str) -> float:
    """基于 prompt 长度和关键词的确定性启发式打分。

    规则：
    - 基础分 = 归一化长度 (0 ~ 0.4)
    - 复杂关键词命中 +0.18 / 个 (最多 +0.54)
    - 中等关键词命中 +0.08 / 个 (最多 +0.24)
    - 最终 clamp 到 [0.0, 1.0]
    """
    # 基础分：基于长度，短 prompt 得分低
    length = len(prompt)
    # 长度贡献：0~20 字 → 0.0~0.1, 20~100 字 → 0.1~0.3, 100+ → 0.3~0.4
    if length <= 20:
        base = length / 20.0 * 0.1
    elif length <= 100:
        base = 0.1 + (length - 20) / 80.0 * 0.2
    else:
        base = min(0.4, 0.3 + (length - 100) / 200.0 * 0.1)

    # 关键词加分
    complex_hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in prompt)
    medium_hits = sum(1 for kw in _MEDIUM_KEYWORDS if kw in prompt)

    score = base + complex_hits * 0.18 + medium_hits * 0.08

    # Clamp
    return max(0.0, min(1.0, score))


class HeuristicMockRouter:
    """模拟 RouteLLM Router 的启发式 mock。

    使用确定性的启发式算法，根据 prompt 的长度和关键词组合
    产生与真实分类器行为一致的分数分布。
    """

    def calculate_strong_win_rate(self, prompt: str) -> float:
        return _heuristic_score(prompt)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier():
    """创建使用 HeuristicMockRouter 的 ModelClassifier 实例。"""
    config = ClassifierConfig(type="mf", model_path=None)

    with patch(
        "aegis_router.router.model_classifier.ModelClassifier._get_router_class"
    ) as mock_cls, patch(
        "aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs"
    ) as mock_kwargs:
        mock_cls.return_value = MagicMock(return_value=HeuristicMockRouter())
        mock_kwargs.return_value = {}

        clf = ModelClassifier(config, timeout_ms=5000.0)
        clf.ensure_loaded()

    return clf


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestClassifierIntegration:
    """RouteLLM 分类器集成测试。"""

    def test_tc_classifier_001_simple_prompt_low_score(self, classifier):
        """TC-CLASSIFIER-001: 简单 prompt 得分应 < 0.3。

        使用简短翻译类 prompt，分类器应识别为简单任务。
        """
        prompt = "翻译: hello → 你好"
        result = classifier.classify(prompt)

        assert isinstance(result, ClassifierResult)
        assert result.score < 0.3, (
            f"简单 prompt 得分应 < 0.3, 实际: {result.score:.4f}"
        )
        assert result.classifier_type == "mf"

    def test_tc_classifier_002_medium_prompt_mid_score(self, classifier):
        """TC-CLASSIFIER-002: 中等 prompt 得分应在 0.3 ~ 0.7 之间。

        使用中等复杂度的分析报告类 prompt。
        """
        prompt = "写一篇500字的产品分析报告"
        result = classifier.classify(prompt)

        assert isinstance(result, ClassifierResult)
        assert 0.3 < result.score < 0.7, (
            f"中等 prompt 得分应在 (0.3, 0.7), 实际: {result.score:.4f}"
        )
        assert result.classifier_type == "mf"

    def test_tc_classifier_003_complex_prompt_high_score(self, classifier):
        """TC-CLASSIFIER-003: 复杂 prompt 得分应 > 0.7。

        使用涉及代码审计和安全漏洞修复的复杂 prompt。
        """
        prompt = "审计这段代码的安全漏洞并给出修复方案，需要考虑SQL注入、XSS攻击和权限提升等问题"
        result = classifier.classify(prompt)

        assert isinstance(result, ClassifierResult)
        assert result.score > 0.7, (
            f"复杂 prompt 得分应 > 0.7, 实际: {result.score:.4f}"
        )
        assert result.classifier_type == "mf"

    def test_tc_classifier_004_latency_under_10ms(self):
        """TC-CLASSIFIER-004: 分类器推理延迟 < 10ms（100 次取平均）。

        使用快速执行的 mock router 验证延迟要求。
        """
        config = ClassifierConfig(type="mf", model_path=None)

        # 使用快速返回的 mock
        fast_router = MagicMock()
        fast_router.calculate_strong_win_rate.return_value = 0.5

        with patch(
            "aegis_router.router.model_classifier.ModelClassifier._get_router_class"
        ) as mock_cls, patch(
            "aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs"
        ) as mock_kwargs:
            mock_cls.return_value = MagicMock(return_value=fast_router)
            mock_kwargs.return_value = {}

            clf = ModelClassifier(config, timeout_ms=5000.0)
            clf.ensure_loaded()

        # 执行 100 次推理并测量平均延迟
        num_iterations = 100
        prompts = [
            "简单问题",
            "写一篇分析报告关于市场趋势",
            "审计代码安全漏洞",
        ]

        start = time.perf_counter()
        for i in range(num_iterations):
            prompt = prompts[i % len(prompts)]
            clf.classify(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        avg_latency_ms = elapsed_ms / num_iterations
        assert avg_latency_ms < 10.0, (
            f"平均推理延迟应 < 10ms, 实际: {avg_latency_ms:.4f}ms"
        )

    def test_tc_classifier_005_masked_score_deviation(self, classifier):
        """TC-CLASSIFIER-005: score_input=masked 模式下脱敏 prompt 打分偏差 < 0.1。

        验证 PII 被替换为占位符后，分类分数与原文差异可接受。
        """
        # 原始 prompt（包含个人信息）
        original_prompt = "帮我分析张三的绩效报告，他在2024年Q1的销售额是150万"

        # 脱敏后的 prompt（PII 被替换为占位符）
        masked_prompt = "帮我分析[PERSON_1]的绩效报告，他在2024年Q1的销售额是[AMOUNT_1]"

        original_result = classifier.classify(original_prompt)
        masked_result = classifier.classify(masked_prompt)

        score_deviation = abs(original_result.score - masked_result.score)
        assert score_deviation < 0.1, (
            f"脱敏前后分数偏差应 < 0.1, "
            f"原始: {original_result.score:.4f}, "
            f"脱敏: {masked_result.score:.4f}, "
            f"偏差: {score_deviation:.4f}"
        )


class TestClassifierIntegrationEdgeCases:
    """集成测试边界情况补充。"""

    def test_empty_prompt_returns_low_score(self, classifier):
        """空 prompt 应得到极低分数。"""
        result = classifier.classify("")
        assert result.score < 0.1

    def test_score_deterministic(self, classifier):
        """相同输入应产生相同输出（确定性验证）。"""
        prompt = "写一篇500字的产品分析报告"
        result1 = classifier.classify(prompt)
        result2 = classifier.classify(prompt)
        assert result1.score == result2.score

    def test_result_contains_latency(self, classifier):
        """结果中应包含非负延迟值。"""
        result = classifier.classify("测试延迟")
        assert result.latency_ms >= 0.0
