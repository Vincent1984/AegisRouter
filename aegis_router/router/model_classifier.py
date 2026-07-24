"""RouteLLM 推理封装

加载 RouteLLM 分类器模型（mf/bert），对 prompt 进行本地推理，
输出 [0, 1] 的难度分数。分数越高表示 prompt 越复杂。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import anyio

from aegis_router.config import ClassifierConfig

logger = logging.getLogger(__name__)

# 默认推理超时（毫秒）
DEFAULT_TIMEOUT_MS: float = 15.0


@dataclass
class ClassifierResult:
    """分类器推理结果。"""

    score: float
    """prompt 难度分数 [0, 1]，越高越复杂。"""

    classifier_type: str
    """分类器类型，如 'mf' 或 'bert'。"""

    latency_ms: float
    """推理耗时（毫秒）。"""


class ModelClassifier:
    """RouteLLM 模型分类器封装。

    特性：
    - 懒加载：首次调用时才加载模型
    - 线程安全：使用锁保护模型加载
    - 异步兼容：提供 async classify 方法
    - 超时机制：推理超时返回 TimeoutError

    Parameters
    ----------
    config : ClassifierConfig
        分类器配置，包含 type 和 model_path。
    timeout_ms : float
        推理超时阈值（毫秒），默认 15ms。
    """

    def __init__(
        self,
        config: ClassifierConfig,
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self._config = config
        self._timeout_ms = timeout_ms
        self._router: Optional[object] = None
        self._load_lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """分类器是否可用（已加载且无错误）。"""
        return self._loaded and self._load_error is None

    @property
    def classifier_type(self) -> str:
        """当前配置的分类器类型。"""
        return self._config.type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """确保模型已加载（懒加载入口）。

        线程安全，多线程同时调用时只会加载一次。

        Raises
        ------
        RuntimeError
            如果模型加载失败。
        """
        if self._loaded:
            if self._load_error is not None:
                raise RuntimeError(
                    f"Classifier '{self._config.type}' failed to load: {self._load_error}"
                )
            return

        with self._load_lock:
            # Double-check pattern
            if self._loaded:
                if self._load_error is not None:
                    raise RuntimeError(
                        f"Classifier '{self._config.type}' failed to load: {self._load_error}"
                    )
                return
            self._do_load()

    def classify(self, prompt: str) -> ClassifierResult:
        """对 prompt 进行同步推理，返回难度分数。

        Parameters
        ----------
        prompt : str
            待评估的用户 prompt。

        Returns
        -------
        ClassifierResult
            包含分数、分类器类型、延迟的结果。

        Raises
        ------
        RuntimeError
            模型未加载或加载失败。
        TimeoutError
            推理超时。
        """
        self.ensure_loaded()

        start = time.perf_counter()
        score = self._router.calculate_strong_win_rate(prompt)  # type: ignore[union-attr]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if elapsed_ms > self._timeout_ms:
            raise TimeoutError(
                f"Classifier inference exceeded timeout: "
                f"{elapsed_ms:.2f}ms > {self._timeout_ms:.2f}ms"
            )

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, float(score)))

        return ClassifierResult(
            score=score,
            classifier_type=self._config.type,
            latency_ms=elapsed_ms,
        )

    async def aclassify(self, prompt: str) -> ClassifierResult:
        """对 prompt 进行异步推理，返回难度分数。

        内部使用 anyio.to_thread.run_sync 将同步推理包装为异步调用。

        Parameters
        ----------
        prompt : str
            待评估的用户 prompt。

        Returns
        -------
        ClassifierResult
            包含分数、分类器类型、延迟的结果。

        Raises
        ------
        RuntimeError
            模型未加载或加载失败。
        TimeoutError
            推理超时。
        """
        return await anyio.to_thread.run_sync(self.classify, prompt)

    def reload(self, config: Optional[ClassifierConfig] = None) -> None:
        """重新加载模型（用于热更新）。

        Parameters
        ----------
        config : ClassifierConfig, optional
            新配置。如果为 None，则使用当前配置重新加载。
        """
        with self._load_lock:
            if config is not None:
                self._config = config
            self._router = None
            self._loaded = False
            self._load_error = None
            self._do_load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_load(self) -> None:
        """执行实际的模型加载逻辑。"""
        try:
            router_cls = self._get_router_class(self._config.type)
            config_kwargs = self._get_router_kwargs()

            logger.info(
                "Loading RouteLLM classifier: type=%s, model_path=%s",
                self._config.type,
                self._config.model_path,
            )

            self._router = router_cls(**config_kwargs)
            self._loaded = True
            self._load_error = None

            logger.info("RouteLLM classifier '%s' loaded successfully.", self._config.type)

        except Exception as exc:
            self._loaded = True  # Mark as attempted
            self._load_error = exc
            self._router = None
            logger.warning(
                "Failed to load RouteLLM classifier '%s': %s",
                self._config.type,
                exc,
            )
            raise RuntimeError(
                f"Classifier '{self._config.type}' failed to load: {exc}"
            ) from exc

    def _get_router_class(self, classifier_type: str):
        """获取 RouteLLM Router 类。"""
        from routellm.routers.routers import ROUTER_CLS

        if classifier_type not in ROUTER_CLS:
            raise ValueError(
                f"Unsupported classifier type: '{classifier_type}'. "
                f"Supported types: {list(ROUTER_CLS.keys())}"
            )
        return ROUTER_CLS[classifier_type]

    def _get_router_kwargs(self) -> dict:
        """构建 Router 构造参数。

        当 model_path 为 None 时，使用 RouteLLM 的默认 GPT-4 增强配置。
        """
        classifier_type = self._config.type

        if self._config.model_path is not None:
            return {"checkpoint_path": self._config.model_path}

        # Use default config from RouteLLM
        from routellm.controller import GPT_4_AUGMENTED_CONFIG

        default_config = GPT_4_AUGMENTED_CONFIG.get(classifier_type, {})
        return dict(default_config)
