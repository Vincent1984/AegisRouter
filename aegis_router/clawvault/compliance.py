"""合规检测引擎

提供 Prompt Injection 检测、敏感词过滤等合规拦截能力。
支持三种拦截模式: strict / interactive / permissive。
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger("clawvault.compliance")

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

ComplianceMode = Literal["strict", "interactive", "permissive"]


@dataclass
class Violation:
    """单条违规记录。"""

    id: str
    pattern: str
    severity: str
    description: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
class ComplianceResult:
    """合规检测结果。"""

    passed: bool
    violations: list[Violation] = field(default_factory=list)
    mode: ComplianceMode = "strict"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "mode": self.mode,
        }


# ---------------------------------------------------------------------------
# Injection Pattern
# ---------------------------------------------------------------------------


@dataclass
class InjectionPattern:
    """单条注入攻击模式。"""

    id: str
    pattern: str
    severity: str
    description: str


# ---------------------------------------------------------------------------
# ComplianceEngine
# ---------------------------------------------------------------------------


class ComplianceEngine:
    """合规检测引擎 — 支持 Prompt Injection 检测 + 敏感词过滤。

    Parameters
    ----------
    patterns_file : str | Path
        注入攻击模式 YAML 文件路径。
    sensitive_words_file : str | Path
        敏感词库文本文件路径（每行一个词，# 开头为注释）。
    mode : ComplianceMode
        拦截模式: strict（严格拒绝）、interactive（提示确认）、permissive（记录但放行）。
    role_switch_keywords_file : str | Path | None
        角色切换关键词文件路径（每行一个，# 为注释）。为 None 时使用内置默认列表。
    feature_scoring_config : dict | None
        特征评分配置，格式: {"min_hits": 3, "density_threshold": 0.15, "min_words": 5}
    """

    # 用于检测 Base64 编码内容的正则 (至少 20 个字符的连续 base64)
    _BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

    # 默认角色切换关键词（仅在未指定外部文件时使用）
    _DEFAULT_ROLE_SWITCH_KEYWORDS = [
        "ignore",
        "forget",
        "disregard",
        "override",
        "bypass",
        "pretend",
        "act as",
        "role play",
        "忽略",
        "忘记",
        "无视",
        "覆盖",
        "绕过",
        "假装",
        "扮演",
        "角色扮演",
    ]

    # 默认特征评分阈值
    _DEFAULT_FEATURE_SCORING = {
        "min_hits": 3,
        "density_threshold": 0.15,
        "min_words": 5,
    }

    def __init__(
        self,
        patterns_file: str | Path | None = None,
        sensitive_words_file: str | Path | None = None,
        mode: ComplianceMode = "strict",
        role_switch_keywords_file: str | Path | None = None,
        feature_scoring_config: dict | None = None,
    ):
        self.mode: ComplianceMode = mode
        self._patterns: list[InjectionPattern] = []
        self._sensitive_words: list[str] = []
        self._role_switch_keywords: list[str] = []
        self._feature_scoring: dict = feature_scoring_config or self._DEFAULT_FEATURE_SCORING.copy()

        # File paths for reload
        self._patterns_file = Path(patterns_file) if patterns_file else None
        self._sensitive_words_file = Path(sensitive_words_file) if sensitive_words_file else None
        self._role_switch_keywords_file = Path(role_switch_keywords_file) if role_switch_keywords_file else None

        # Load rules
        if self._patterns_file:
            self._load_patterns()
        if self._sensitive_words_file:
            self._load_sensitive_words()
        self._load_role_switch_keywords()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_compliance(
        self, text: str, direction: str = "inbound"
    ) -> ComplianceResult:
        """执行合规检测。

        Parameters
        ----------
        text : str
            待检测文本。
        direction : str
            检测方向: "inbound"（请求前置）或 "outbound"（响应后置）。

        Returns
        -------
        ComplianceResult
            检测结果，包含是否通过和违规列表。
        """
        violations: list[Violation] = []

        if direction == "inbound":
            # 1. 规则匹配 (fast path) — Prompt Injection 检测
            violations.extend(self._check_injection_patterns(text))

            # 2. Base64 编码检测 — 解码后再检查
            violations.extend(self._check_base64_injection(text))

            # 3. 特征评分 — 关键词密度
            violations.extend(self._check_feature_scoring(text))

            # 4. 敏感词过滤
            violations.extend(self._check_sensitive_words(text))

        elif direction == "outbound":
            # 响应后置检测: 敏感词 + 有害内容
            violations.extend(self._check_sensitive_words(text))

        # 根据模式决定是否通过
        passed = self._determine_passed(violations)

        return ComplianceResult(
            passed=passed,
            violations=violations,
            mode=self.mode,
        )

    def reload(self) -> None:
        """重新加载规则文件（热更新）。"""
        if self._patterns_file:
            self._load_patterns()
            logger.info("Reloaded injection patterns from %s", self._patterns_file)
        if self._sensitive_words_file:
            self._load_sensitive_words()
            logger.info("Reloaded sensitive words from %s", self._sensitive_words_file)
        self._load_role_switch_keywords()
        logger.info("Reloaded role switch keywords")

    @property
    def patterns(self) -> list[InjectionPattern]:
        """当前加载的注入攻击模式列表。"""
        return self._patterns

    @property
    def sensitive_words(self) -> list[str]:
        """当前加载的敏感词列表。"""
        return self._sensitive_words

    # ------------------------------------------------------------------
    # Private: Loading
    # ------------------------------------------------------------------

    def _load_patterns(self) -> None:
        """从 YAML 文件加载注入攻击模式。"""
        if self._patterns_file is None or not self._patterns_file.exists():
            self._patterns = []
            return

        try:
            with open(self._patterns_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict) or "patterns" not in data:
                self._patterns = []
                return

            self._patterns = [
                InjectionPattern(
                    id=p.get("id", "UNKNOWN"),
                    pattern=p.get("pattern", ""),
                    severity=p.get("severity", "medium"),
                    description=p.get("description", ""),
                )
                for p in data["patterns"]
                if p.get("pattern")
            ]
        except Exception as exc:
            logger.error("Failed to load injection patterns: %s", exc)
            self._patterns = []

    def _load_sensitive_words(self) -> None:
        """从文本文件加载敏感词（每行一个，# 为注释）。"""
        if self._sensitive_words_file is None or not self._sensitive_words_file.exists():
            self._sensitive_words = []
            return

        try:
            with open(self._sensitive_words_file, "r", encoding="utf-8") as f:
                words = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        words.append(line)
                self._sensitive_words = words
        except Exception as exc:
            logger.error("Failed to load sensitive words: %s", exc)
            self._sensitive_words = []

    def _load_role_switch_keywords(self) -> None:
        """从文本文件加载角色切换关键词，若文件不存在则使用内置默认列表。"""
        if self._role_switch_keywords_file and self._role_switch_keywords_file.exists():
            try:
                with open(self._role_switch_keywords_file, "r", encoding="utf-8") as f:
                    keywords = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            keywords.append(line)
                    self._role_switch_keywords = keywords
                    logger.info("Loaded %d role switch keywords from %s", len(keywords), self._role_switch_keywords_file)
                    return
            except Exception as exc:
                logger.error("Failed to load role switch keywords: %s", exc)

        # Fallback to defaults
        self._role_switch_keywords = self._DEFAULT_ROLE_SWITCH_KEYWORDS.copy()

    # ------------------------------------------------------------------
    # Private: Detection
    # ------------------------------------------------------------------

    def _check_injection_patterns(self, text: str) -> list[Violation]:
        """规则匹配 (fast path): 大小写不敏感的模式匹配。"""
        violations = []
        text_lower = text.lower()

        for pattern in self._patterns:
            if pattern.pattern.lower() in text_lower:
                violations.append(
                    Violation(
                        id=pattern.id,
                        pattern=pattern.pattern,
                        severity=pattern.severity,
                        description=pattern.description,
                    )
                )

        return violations

    def _check_base64_injection(self, text: str) -> list[Violation]:
        """检测 Base64 编码的注入攻击: 解码后重新检查模式。"""
        violations = []

        for match in self._BASE64_PATTERN.finditer(text):
            candidate = match.group()
            try:
                decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
            except Exception:
                continue

            if not decoded or len(decoded) < 5:
                continue

            # 对解码后的文本执行模式检查
            decoded_lower = decoded.lower()
            for pattern in self._patterns:
                if pattern.pattern.lower() in decoded_lower:
                    violations.append(
                        Violation(
                            id=f"{pattern.id}_B64",
                            pattern=f"base64({pattern.pattern})",
                            severity=pattern.severity,
                            description=f"Base64 编码的注入攻击: {pattern.description}",
                        )
                    )

        return violations

    def _check_feature_scoring(self, text: str) -> list[Violation]:
        """特征评分: 检测角色切换关键词密度。

        如果关键词密度超过阈值，认为存在高风险的角色切换攻击。
        阈值通过 feature_scoring_config 配置。
        """
        violations = []
        text_lower = text.lower()

        min_words = self._feature_scoring.get("min_words", 5)
        min_hits = self._feature_scoring.get("min_hits", 3)
        density_threshold = self._feature_scoring.get("density_threshold", 0.15)

        # 短文本不做密度检测（避免误报）
        word_count = max(len(text.split()), 1)
        if word_count < min_words:
            return violations

        hit_count = sum(1 for kw in self._role_switch_keywords if kw in text_lower)

        # 密度阈值检查
        density = hit_count / word_count
        if hit_count >= min_hits and density > density_threshold:
            violations.append(
                Violation(
                    id="FEAT_ROLE_SWITCH",
                    pattern=f"role_switch_density={density:.2f}",
                    severity="medium",
                    description=f"角色切换关键词密度过高 ({hit_count} hits, density={density:.2%})",
                )
            )

        return violations

    def _check_sensitive_words(self, text: str) -> list[Violation]:
        """敏感词过滤: 大小写不敏感匹配。"""
        violations = []
        text_lower = text.lower()

        for idx, word in enumerate(self._sensitive_words, start=1):
            if word.lower() in text_lower:
                violations.append(
                    Violation(
                        id=f"SENS_{idx:03d}",
                        pattern=word,
                        severity="high",
                        description=f"命中敏感词: {word}",
                    )
                )

        return violations

    # ------------------------------------------------------------------
    # Private: Mode logic
    # ------------------------------------------------------------------

    def _determine_passed(self, violations: list[Violation]) -> bool:
        """根据拦截模式决定是否通过。

        - strict: 有任何违规即拒绝
        - interactive: 有违规时标记不通过（由上层决定是否提示用户确认）
        - permissive: 始终放行（仅记录）
        """
        if not violations:
            return True

        if self.mode == "permissive":
            return True
        else:
            # strict 和 interactive 都标记为不通过
            return False


# ---------------------------------------------------------------------------
# Module-level singleton (lazy initialization)
# ---------------------------------------------------------------------------

_engine: ComplianceEngine | None = None


def get_compliance_engine(
    patterns_file: str | Path | None = None,
    sensitive_words_file: str | Path | None = None,
    mode: ComplianceMode = "strict",
    role_switch_keywords_file: str | Path | None = None,
    feature_scoring_config: dict | None = None,
) -> ComplianceEngine:
    """获取或创建合规检测引擎单例。"""
    global _engine
    if _engine is None:
        _engine = ComplianceEngine(
            patterns_file=patterns_file,
            sensitive_words_file=sensitive_words_file,
            mode=mode,
            role_switch_keywords_file=role_switch_keywords_file,
            feature_scoring_config=feature_scoring_config,
        )
    return _engine


def reset_compliance_engine() -> None:
    """重置单例（用于测试）。"""
    global _engine
    _engine = None
