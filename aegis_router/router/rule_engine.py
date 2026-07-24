"""规则前置引擎

在路由管道的第一阶段检测寒暄/问候类 prompt，
命中时直接路由到本地小模型，跳过 RouteLLM 分类器。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aegis_router.config import TrivialConfig


@dataclass
class RuleEngineResult:
    """规则引擎路由决策结果。"""

    matched: bool
    target_model: Optional[str] = None
    matched_pattern: Optional[str] = None


class RuleEngine:
    """规则前置引擎 — 寒暄词库匹配。

    Parameters
    ----------
    config : TrivialConfig
        规则前置配置，包括 enabled、max_length、patterns_file、target_model。
    """

    def __init__(self, config: TrivialConfig) -> None:
        self._config = config
        self._patterns: list[str] = []
        if self._config.enabled:
            self._load_patterns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trivial_chat(self, prompt: str) -> bool:
        """判断 prompt 是否为寒暄/问候类文本。

        逻辑：
        1. 如果引擎未启用，直接返回 False
        2. 如果 prompt 长度超过 max_length，返回 False
        3. 对 prompt 进行 lower + strip 后，检查是否包含任一模式串

        Returns
        -------
        bool
            True 表示是寒暄文本，应路由到本地小模型。
        """
        if not self._config.enabled:
            return False

        if len(prompt.strip()) > self._config.max_length:
            return False

        prompt_lower = prompt.lower().strip()
        return any(p in prompt_lower for p in self._patterns)

    def check(self, prompt: str) -> RuleEngineResult:
        """检查 prompt 并返回完整路由决策。

        Returns
        -------
        RuleEngineResult
            包含是否命中、目标模型、命中的模式串。
        """
        if not self._config.enabled:
            return RuleEngineResult(matched=False)

        if len(prompt.strip()) > self._config.max_length:
            return RuleEngineResult(matched=False)

        prompt_lower = prompt.lower().strip()
        for pattern in self._patterns:
            if pattern in prompt_lower:
                return RuleEngineResult(
                    matched=True,
                    target_model=self._config.target_model,
                    matched_pattern=pattern,
                )
        return RuleEngineResult(matched=False)

    def get_target_model(self) -> str:
        """返回寒暄路由的目标模型名称。"""
        return self._config.target_model

    def reload_patterns(self) -> None:
        """重新加载模式文件（用于热更新）。"""
        self._patterns = []
        if self._config.enabled:
            self._load_patterns()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_patterns(self) -> None:
        """从文件加载模式列表。

        文件格式：每行一个模式，# 开头为注释，空行跳过。
        所有模式统一转为 lowercase 存储。
        """
        patterns_file = self._resolve_patterns_path()
        if patterns_file is None or not patterns_file.exists():
            return

        with open(patterns_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                self._patterns.append(line.lower())

    def _resolve_patterns_path(self) -> Optional[Path]:
        """解析模式文件路径。"""
        if self._config.patterns_file:
            return Path(self._config.patterns_file)
        # 默认路径
        return Path("./patterns/trivial_chat.txt")
