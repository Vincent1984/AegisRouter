"""Tests for LiteLLM Failover chain configuration (Task 21).

验证 config/config.yaml 中的 router_settings / fallbacks 配置:
- YAML 格式合法
- 所有 fallback 引用的模型均在 model_list 中声明
- 配置结构符合 LiteLLM 文档要求
"""

from pathlib import Path

import pytest
import yaml

from aegis_router.config import (
    AegisConfig,
    LiteLLMConfig,
    RouterSettings,
    load_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


@pytest.fixture
def project_config_dir() -> Path:
    """Return path to the actual project config/ directory."""
    path = Path(__file__).parent.parent / "config"
    if not path.exists():
        pytest.skip("Project config/ directory not found")
    return path


@pytest.fixture
def config_yaml_path(project_config_dir: Path) -> Path:
    return project_config_dir / "config.yaml"


@pytest.fixture
def raw_config(config_yaml_path: Path) -> dict:
    """Load raw YAML from config.yaml."""
    with open(config_yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests: YAML validity and structure
# ---------------------------------------------------------------------------


class TestFailoverConfigYAML:
    """验证 config.yaml 是合法 YAML 且包含 router_settings。"""

    def test_config_yaml_is_valid(self, config_yaml_path: Path):
        """config.yaml 可被正确解析为 YAML。"""
        with open(config_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_router_settings_present(self, raw_config: dict):
        """config.yaml 包含 router_settings 顶层键。"""
        assert "router_settings" in raw_config

    def test_router_settings_has_fallbacks(self, raw_config: dict):
        """router_settings 中包含 fallbacks 列表。"""
        rs = raw_config["router_settings"]
        assert "fallbacks" in rs
        assert isinstance(rs["fallbacks"], list)
        assert len(rs["fallbacks"]) > 0

    def test_router_settings_has_required_fields(self, raw_config: dict):
        """router_settings 包含关键配置字段。"""
        rs = raw_config["router_settings"]
        assert "routing_strategy" in rs
        assert "num_retries" in rs
        assert "timeout" in rs
        assert "allowed_fails" in rs


# ---------------------------------------------------------------------------
# Tests: Failover chain consistency
# ---------------------------------------------------------------------------


class TestFailoverChainConsistency:
    """验证 fallback 链中引用的模型都在 model_list 中声明。"""

    def test_all_fallback_models_exist_in_model_list(self, raw_config: dict):
        """fallbacks 中出现的所有模型名称必须在 model_list 中声明。"""
        model_names = {m["model_name"] for m in raw_config["model_list"]}
        fallbacks = raw_config["router_settings"]["fallbacks"]

        for entry in fallbacks:
            for primary_model, candidates in entry.items():
                # 主模型在 model_list 中
                assert primary_model in model_names, (
                    f"主模型 '{primary_model}' 未在 model_list 中声明"
                )
                # 每个候选模型也在 model_list 中
                for candidate in candidates:
                    assert candidate in model_names, (
                        f"候选模型 '{candidate}' (来自 {primary_model} 的 fallback 链) "
                        f"未在 model_list 中声明"
                    )

    def test_fallback_entries_are_dicts(self, raw_config: dict):
        """fallbacks 列表中每个条目是 dict 格式。"""
        fallbacks = raw_config["router_settings"]["fallbacks"]
        for entry in fallbacks:
            assert isinstance(entry, dict)
            # 每个 entry 应有且仅有一个键
            assert len(entry) == 1

    def test_no_self_referencing_fallback(self, raw_config: dict):
        """模型不应将自身列为候选 fallback。"""
        fallbacks = raw_config["router_settings"]["fallbacks"]
        for entry in fallbacks:
            for primary, candidates in entry.items():
                assert primary not in candidates, (
                    f"模型 '{primary}' 不应将自身列为 fallback 候选"
                )


# ---------------------------------------------------------------------------
# Tests: Pydantic model parsing
# ---------------------------------------------------------------------------


class TestRouterSettingsPydantic:
    """验证 Pydantic 模型可以正确解析 router_settings。"""

    def test_load_config_parses_router_settings(self, project_config_dir: Path):
        """load_config 能正确加载 router_settings 到 AegisConfig。"""
        cfg = load_config(project_config_dir)
        assert cfg.litellm.router_settings is not None

    def test_router_settings_type(self, project_config_dir: Path):
        """router_settings 被解析为 RouterSettings 实例。"""
        cfg = load_config(project_config_dir)
        assert isinstance(cfg.litellm.router_settings, RouterSettings)

    def test_router_settings_fallbacks_not_empty(self, project_config_dir: Path):
        """解析后的 fallbacks 列表非空。"""
        cfg = load_config(project_config_dir)
        rs = cfg.litellm.router_settings
        assert len(rs.fallbacks) > 0

    def test_router_settings_values(self, project_config_dir: Path):
        """router_settings 的配置值与 YAML 中的一致。"""
        cfg = load_config(project_config_dir)
        rs = cfg.litellm.router_settings
        assert rs.routing_strategy == "simple-shuffle"
        assert rs.num_retries == 2
        assert rs.timeout == 30
        assert rs.retry_after == 1
        assert rs.allowed_fails == 1

    def test_config_without_router_settings_defaults_to_none(self, tmp_path: Path):
        """不包含 router_settings 的配置文件不会报错，默认为 None。"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        minimal_config = {
            "model_list": [
                {
                    "model_name": "test",
                    "litellm_params": {"model": "openai/test"},
                }
            ]
        }
        (config_dir / "config.yaml").write_text(
            yaml.dump(minimal_config), encoding="utf-8"
        )
        cfg = load_config(config_dir)
        assert cfg.litellm.router_settings is None
