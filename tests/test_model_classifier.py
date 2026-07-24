"""ModelClassifier 单元测试

使用 mock 避免加载实际的 RouteLLM 模型权重。
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aegis_router.config import ClassifierConfig
from aegis_router.router.model_classifier import (
    ClassifierResult,
    ModelClassifier,
    DEFAULT_TIMEOUT_MS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mf_config():
    """MF 类型分类器配置。"""
    return ClassifierConfig(type="mf", model_path=None)


@pytest.fixture
def bert_config():
    """BERT 类型分类器配置。"""
    return ClassifierConfig(type="bert", model_path="/path/to/bert")


@pytest.fixture
def mock_router():
    """创建模拟的 RouteLLM Router 实例。"""
    router = MagicMock()
    router.calculate_strong_win_rate.return_value = 0.75
    return router


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestModelClassifierInit:
    """初始化与配置测试。"""

    def test_init_default_timeout(self, mf_config):
        classifier = ModelClassifier(mf_config)
        assert classifier._timeout_ms == DEFAULT_TIMEOUT_MS
        assert classifier.classifier_type == "mf"
        assert not classifier.is_available

    def test_init_custom_timeout(self, mf_config):
        classifier = ModelClassifier(mf_config, timeout_ms=50.0)
        assert classifier._timeout_ms == 50.0

    def test_classifier_type_property(self, bert_config):
        classifier = ModelClassifier(bert_config)
        assert classifier.classifier_type == "bert"


# ---------------------------------------------------------------------------
# Lazy Loading Tests
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """懒加载行为测试。"""

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_model_not_loaded_until_first_use(
        self, mock_kwargs, mock_cls, mf_config
    ):
        """模型在初始化时不加载。"""
        classifier = ModelClassifier(mf_config)
        assert not classifier._loaded
        assert classifier._router is None
        mock_cls.assert_not_called()

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_model_loaded_on_ensure_loaded(
        self, mock_kwargs, mock_cls, mf_config
    ):
        """ensure_loaded 触发模型加载。"""
        mock_router_instance = MagicMock()
        mock_cls.return_value = MagicMock(return_value=mock_router_instance)
        mock_kwargs.return_value = {"checkpoint_path": "routellm/mf_gpt4_augmented"}

        classifier = ModelClassifier(mf_config)
        classifier.ensure_loaded()

        assert classifier._loaded
        assert classifier.is_available
        assert classifier._router is mock_router_instance

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_model_loaded_only_once(self, mock_kwargs, mock_cls, mf_config):
        """多次调用 ensure_loaded 只加载一次。"""
        mock_cls.return_value = MagicMock(return_value=MagicMock())
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config)
        classifier.ensure_loaded()
        classifier.ensure_loaded()
        classifier.ensure_loaded()

        # _get_router_class called only once
        mock_cls.assert_called_once()

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_thread_safe_loading(self, mock_kwargs, mock_cls, mf_config):
        """多线程并发调用 ensure_loaded 只加载一次。"""
        load_count = {"value": 0}

        def side_effect(*args, **kwargs):
            load_count["value"] += 1
            time.sleep(0.01)  # Simulate loading delay
            return MagicMock()

        mock_cls.return_value = side_effect
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config)

        threads = [
            threading.Thread(target=classifier.ensure_loaded) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert load_count["value"] == 1


# ---------------------------------------------------------------------------
# Classification Tests
# ---------------------------------------------------------------------------


class TestClassify:
    """推理功能测试。"""

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_classify_returns_result(self, mock_kwargs, mock_cls, mf_config):
        """classify 返回正确的 ClassifierResult。"""
        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.return_value = 0.82
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config, timeout_ms=5000.0)
        result = classifier.classify("Explain quantum computing in detail")

        assert isinstance(result, ClassifierResult)
        assert result.score == 0.82
        assert result.classifier_type == "mf"
        assert result.latency_ms >= 0.0
        mock_router.calculate_strong_win_rate.assert_called_once_with(
            "Explain quantum computing in detail"
        )

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_classify_clamps_score_above_one(self, mock_kwargs, mock_cls, mf_config):
        """分数超过 1.0 时被 clamp 到 1.0。"""
        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.return_value = 1.5
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config, timeout_ms=5000.0)
        result = classifier.classify("test")
        assert result.score == 1.0

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_classify_clamps_score_below_zero(self, mock_kwargs, mock_cls, mf_config):
        """分数低于 0.0 时被 clamp 到 0.0。"""
        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.return_value = -0.3
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config, timeout_ms=5000.0)
        result = classifier.classify("test")
        assert result.score == 0.0

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_classify_with_bert_type(self, mock_kwargs, mock_cls, bert_config):
        """BERT 分类器正常工作。"""
        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.return_value = 0.45
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(bert_config, timeout_ms=5000.0)
        result = classifier.classify("Hello!")
        assert result.score == 0.45
        assert result.classifier_type == "bert"


# ---------------------------------------------------------------------------
# Timeout Tests
# ---------------------------------------------------------------------------


class TestTimeout:
    """超时机制测试。"""

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_timeout_raises_on_slow_inference(self, mock_kwargs, mock_cls, mf_config):
        """推理超时时抛出 TimeoutError。"""

        def slow_inference(prompt):
            time.sleep(0.05)  # 50ms - well over any reasonable timeout
            return 0.5

        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.side_effect = slow_inference
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        # Set very low timeout to guarantee timeout
        classifier = ModelClassifier(mf_config, timeout_ms=1.0)

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            classifier.classify("some complex prompt")

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_no_timeout_for_fast_inference(self, mock_kwargs, mock_cls, mf_config):
        """快速推理不触发超时。"""
        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.return_value = 0.6
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config, timeout_ms=5000.0)
        result = classifier.classify("hi")
        assert result.score == 0.6


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """错误处理测试。"""

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_load_failure_marks_unavailable(self, mock_kwargs, mock_cls, mf_config):
        """模型加载失败时标记为不可用。"""
        mock_cls.return_value = MagicMock(
            side_effect=RuntimeError("Model file not found")
        )
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config)

        with pytest.raises(RuntimeError, match="failed to load"):
            classifier.ensure_loaded()

        assert not classifier.is_available

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_classify_raises_after_load_failure(self, mock_kwargs, mock_cls, mf_config):
        """模型加载失败后调用 classify 抛出 RuntimeError。"""
        mock_cls.return_value = MagicMock(
            side_effect=ValueError("Bad checkpoint")
        )
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config)

        with pytest.raises(RuntimeError):
            classifier.ensure_loaded()

        with pytest.raises(RuntimeError, match="failed to load"):
            classifier.classify("test")

    def test_unsupported_classifier_type(self):
        """不支持的分类器类型抛出错误。"""
        config = ClassifierConfig(type="nonexistent", model_path=None)
        classifier = ModelClassifier(config)

        with pytest.raises(RuntimeError, match="failed to load"):
            classifier.ensure_loaded()


# ---------------------------------------------------------------------------
# Reload Tests
# ---------------------------------------------------------------------------


class TestReload:
    """热更新测试。"""

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_reload_reloads_model(self, mock_kwargs, mock_cls, mf_config):
        """reload 重新加载模型。"""
        mock_router_1 = MagicMock()
        mock_router_2 = MagicMock()
        mock_cls.return_value = MagicMock(side_effect=[mock_router_1, mock_router_2])
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config, timeout_ms=5000.0)
        classifier.ensure_loaded()
        assert classifier._router is mock_router_1

        classifier.reload()
        assert classifier._router is mock_router_2

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    def test_reload_with_new_config(self, mock_kwargs, mock_cls, mf_config):
        """reload 可以使用新配置。"""
        mock_cls.return_value = MagicMock(return_value=MagicMock())
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config)
        classifier.ensure_loaded()
        assert classifier.classifier_type == "mf"

        new_config = ClassifierConfig(type="bert", model_path="/new/path")
        classifier.reload(config=new_config)
        assert classifier.classifier_type == "bert"


# ---------------------------------------------------------------------------
# Async Tests
# ---------------------------------------------------------------------------


class TestAsync:
    """异步接口测试。"""

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    async def test_aclassify_returns_result(self, mock_kwargs, mock_cls, mf_config):
        """aclassify 异步返回正确结果。"""
        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.return_value = 0.9
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config, timeout_ms=5000.0)
        result = await classifier.aclassify("Write a distributed system")

        assert isinstance(result, ClassifierResult)
        assert result.score == 0.9
        assert result.classifier_type == "mf"

    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_class")
    @patch("aegis_router.router.model_classifier.ModelClassifier._get_router_kwargs")
    async def test_aclassify_propagates_timeout(self, mock_kwargs, mock_cls, mf_config):
        """aclassify 传播超时异常。"""

        def slow_inference(prompt):
            time.sleep(0.05)
            return 0.5

        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate.side_effect = slow_inference
        mock_cls.return_value = MagicMock(return_value=mock_router)
        mock_kwargs.return_value = {}

        classifier = ModelClassifier(mf_config, timeout_ms=1.0)

        with pytest.raises(TimeoutError):
            await classifier.aclassify("test")


# ---------------------------------------------------------------------------
# Router Config Resolution Tests
# ---------------------------------------------------------------------------


class TestRouterConfig:
    """Router 配置解析测试。"""

    def test_default_config_used_when_model_path_none(self, mf_config):
        """model_path 为 None 时使用默认配置。"""
        import sys

        fake_controller = MagicMock()
        fake_controller.GPT_4_AUGMENTED_CONFIG = {
            "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
            "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
        }

        with patch.dict(sys.modules, {"routellm.controller": fake_controller}):
            classifier = ModelClassifier(mf_config)
            kwargs = classifier._get_router_kwargs()
            assert kwargs == {"checkpoint_path": "routellm/mf_gpt4_augmented"}

    def test_custom_model_path_used(self, bert_config):
        """指定 model_path 时直接使用。

        当 model_path 不为 None 时，不需要导入 routellm.controller。
        """
        classifier = ModelClassifier(bert_config)
        kwargs = classifier._get_router_kwargs()
        assert kwargs == {"checkpoint_path": "/path/to/bert"}
