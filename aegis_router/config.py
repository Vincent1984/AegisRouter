"""全局配置管理模块

加载并管理 AegisRouter 系统的 YAML 配置文件:
- config/config.yaml       — LiteLLM 模型池配置
- config/models.yaml       — 模型参数（用于评分/路由）
- config/route_config.yaml — 路由阈值、权重、策略
- config/route_overrides.yaml — 人工覆盖的模型区间
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Environment variable resolution
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"^os\.environ/(.+)$")


def resolve_env_vars(obj: Any) -> Any:
    """递归解析配置值中的 os.environ/VARIABLE_NAME 模式。

    如果环境变量未设置，保留原始字符串（避免启动崩溃）。
    """
    if isinstance(obj, str):
        match = _ENV_PATTERN.match(obj)
        if match:
            var_name = match.group(1)
            return os.environ.get(var_name, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class LiteLLMParams(BaseModel):
    """单个模型的 LiteLLM 参数。"""
    model_config = {"protected_namespaces": ()}

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class ModelListEntry(BaseModel):
    """model_list 中的单个模型条目。"""
    model_config = {"protected_namespaces": ()}

    model_name: str
    litellm_params: LiteLLMParams


class LiteLLMSettings(BaseModel):
    """litellm_settings 段。"""
    callbacks: Optional[str] = None


class GeneralSettings(BaseModel):
    """general_settings 段。"""
    master_key: Optional[str] = None


class RouterSettings(BaseModel):
    """router_settings 段 — LiteLLM Failover 路由配置。"""
    routing_strategy: str = "simple-shuffle"
    num_retries: int = 2
    timeout: int = 30
    retry_after: int = 1
    allowed_fails: int = 1
    fallbacks: list[dict[str, list[str]]] = Field(default_factory=list)


class LiteLLMConfig(BaseModel):
    """config/config.yaml 的完整结构。"""
    model_config = {"protected_namespaces": ()}

    model_list: list[ModelListEntry] = Field(default_factory=list)
    litellm_settings: Optional[LiteLLMSettings] = None
    general_settings: Optional[GeneralSettings] = None
    router_settings: Optional[RouterSettings] = None


# --- models.yaml ---


class ModelParams(BaseModel):
    """单个模型的能力参数。"""
    parameter_size_b: Optional[float] = None
    context_window: int = 4096
    benchmark_mmlu: Optional[float] = None
    benchmark_humaneval: Optional[float] = None
    benchmark_math: Optional[float] = None
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    latency_avg_ms: Optional[float] = None
    supports_streaming: bool = True
    supports_function_call: bool = False
    available: bool = True  # 标记模型是否可用，False 时评分跳过


class ModelEntry(BaseModel):
    """models.yaml 中的单个模型条目。"""
    name: str
    litellm_model: str
    params: ModelParams = Field(default_factory=ModelParams)


class ModelsConfig(BaseModel):
    """config/models.yaml 的完整结构。"""
    models: list[ModelEntry] = Field(default_factory=list)


# --- route_config.yaml ---


class TrivialConfig(BaseModel):
    """规则前置（寒暄检测）配置。"""
    enabled: bool = True
    max_length: int = 30
    patterns_file: Optional[str] = None
    target_model: str = "local-7b"


class ClassifierConfig(BaseModel):
    """RouteLLM 分类器配置。"""
    model_config = {"protected_namespaces": ()}

    type: str = "mf"
    model_path: Optional[str] = None


class ScoringWeights(BaseModel):
    """评分权重。"""
    benchmark_mmlu: float = 0.25
    benchmark_humaneval: float = 0.20
    benchmark_math: float = 0.20
    context_window: float = 0.10
    cost_efficiency: float = 0.25


class ScoringNormalization(BaseModel):
    """归一化边界值。"""
    benchmark_mmlu: list[float] = Field(default_factory=lambda: [50.0, 95.0])
    benchmark_humaneval: list[float] = Field(default_factory=lambda: [30.0, 95.0])
    benchmark_math: list[float] = Field(default_factory=lambda: [20.0, 95.0])
    context_window: list[float] = Field(default_factory=lambda: [4096.0, 2000000.0])
    cost_per_1m_input: list[float] = Field(default_factory=lambda: [0.0, 20.0])


class ScoringConfig(BaseModel):
    """评分配置。"""
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    normalization: ScoringNormalization = Field(default_factory=ScoringNormalization)
    range_tolerance: float = 0.15


class RoutingConfig(BaseModel):
    """config/route_config.yaml 中 routing 段的完整结构。"""
    score_input: str = "masked"
    trivial: TrivialConfig = Field(default_factory=TrivialConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    overlap_strategy: str = "lowest_cost"
    fallback_model: str = "deepseek-v3"
    session_policy: Literal["sticky", "per_turn", "escalate_only"] = "sticky"
    session_lock_ttl_minutes: float = Field(default=60, gt=0)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


class FailoverConfig(BaseModel):
    """config/route_config.yaml 中 failover 段的结构。"""
    enabled: bool = True
    timeout_ms: int = 50
    chains: dict[str, list[str]] = Field(default_factory=dict)


class RouteConfigFile(BaseModel):
    """config/route_config.yaml 的顶层结构。"""
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    failover: FailoverConfig = Field(default_factory=FailoverConfig)


# --- route_overrides.yaml ---


class ScoreOverride(BaseModel):
    """单个模型的人工覆盖配置。"""
    score_range: list[float]
    reason: Optional[str] = None


class RouteOverridesConfig(BaseModel):
    """config/route_overrides.yaml 的完整结构。"""
    overrides: dict[str, ScoreOverride] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregated config object
# ---------------------------------------------------------------------------


class AegisConfig(BaseModel):
    """AegisRouter 聚合配置对象，包含所有配置文件的内容。"""
    litellm: LiteLLMConfig = Field(default_factory=LiteLLMConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    failover: FailoverConfig = Field(default_factory=FailoverConfig)
    overrides: RouteOverridesConfig = Field(default_factory=RouteOverridesConfig)
    config_dir: str = "./config"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_yaml_file(filepath: Path) -> dict[str, Any]:
    """加载单个 YAML 文件，如果文件不存在则返回空字典。"""
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(config_dir: str | Path = "./config") -> AegisConfig:
    """加载所有 YAML 配置文件并返回类型化的 AegisConfig 对象。

    Parameters
    ----------
    config_dir : str | Path
        配置目录路径，默认 ``./config``。

    Returns
    -------
    AegisConfig
        包含所有配置的聚合对象。
    """
    config_path = Path(config_dir)

    # 加载并解析环境变量
    raw_config = resolve_env_vars(_load_yaml_file(config_path / "config.yaml"))
    raw_models = resolve_env_vars(_load_yaml_file(config_path / "models.yaml"))
    raw_route = resolve_env_vars(_load_yaml_file(config_path / "route_config.yaml"))
    raw_overrides = resolve_env_vars(_load_yaml_file(config_path / "route_overrides.yaml"))

    # 构建 Pydantic 模型
    litellm_cfg = LiteLLMConfig(**raw_config) if raw_config else LiteLLMConfig()
    models_cfg = ModelsConfig(**raw_models) if raw_models else ModelsConfig()

    if raw_route:
        route_file = RouteConfigFile(**raw_route)
        routing_cfg = route_file.routing
        failover_cfg = route_file.failover
    else:
        routing_cfg = RoutingConfig()
        failover_cfg = FailoverConfig()

    # Handle case where route_overrides.yaml has `overrides: null` (all entries commented out)
    if raw_overrides:
        # Filter out None values from parsed YAML (e.g., `overrides:` with no value)
        cleaned_overrides = {k: v for k, v in raw_overrides.items() if v is not None}
        overrides_cfg = RouteOverridesConfig(**cleaned_overrides) if cleaned_overrides else RouteOverridesConfig()
    else:
        overrides_cfg = RouteOverridesConfig()

    return AegisConfig(
        litellm=litellm_cfg,
        models=models_cfg,
        routing=routing_cfg,
        failover=failover_cfg,
        overrides=overrides_cfg,
        config_dir=str(config_path),
    )


# ---------------------------------------------------------------------------
# Global config singleton
# ---------------------------------------------------------------------------

_global_config: Optional[AegisConfig] = None


def get_config() -> AegisConfig:
    """获取全局配置单例。如果尚未加载，则使用默认路径加载。"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def reload_config(config_dir: str | Path | None = None) -> AegisConfig:
    """重新加载配置（用于热更新场景）。

    Parameters
    ----------
    config_dir : str | Path | None
        配置目录路径。为 None 时使用当前配置中记录的路径。

    Returns
    -------
    AegisConfig
        重新加载后的配置对象。
    """
    global _global_config
    if config_dir is None and _global_config is not None:
        config_dir = _global_config.config_dir
    elif config_dir is None:
        config_dir = "./config"
    _global_config = load_config(config_dir)
    return _global_config


def reset_config() -> None:
    """重置全局配置单例（主要用于测试）。"""
    global _global_config
    _global_config = None
