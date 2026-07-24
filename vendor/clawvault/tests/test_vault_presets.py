"""Tests for built-in vault presets."""
# ruff: noqa: S101

from __future__ import annotations

import yaml

from claw_vault.config import (
    DetectionConfig,
    FileMonitorConfig,
    GuardConfig,
    VaultPreset,
    get_builtin_presets,
    load_settings,
    save_settings,
)

EXPECTED_PRESET_IDS = [
    "file-protection",
    "photo-protection",
    "account-secrets",
    "privacy-shield",
    "full-lockdown",
    "developer-workflow",
    "cloud-infra",
    "crypto-wallet",
    "database-protection",
    "healthcare-hipaa",
    "financial-strict",
    "audit-only",
    "communication-logs",
    "source-code-repo",
    "ci-cd-pipelines",
    "mobile-dev",
    "backup-archive",
    "legal-contracts",
    "enterprise-internal",
    "gdpr-compliance",
    "hr-recruiting",
]

ALLOWED_RULE_ACTIONS = {"allow", "block", "sanitize", "ask_user"}


def test_builtin_vault_presets_restore_all_historical_ids():
    presets = get_builtin_presets()

    assert len(presets) == 21
    assert [preset.id for preset in presets] == EXPECTED_PRESET_IDS
    assert len({preset.id for preset in presets}) == len(presets)


def test_builtin_vault_presets_have_required_metadata():
    for preset in get_builtin_presets():
        assert preset.builtin is True
        assert preset.id
        assert preset.name
        assert preset.description
        assert preset.icon
        assert preset.created_at


def test_builtin_vault_presets_match_current_config_models():
    for preset in get_builtin_presets():
        DetectionConfig(**preset.detection)
        GuardConfig(**preset.guard)
        FileMonitorConfig(**preset.file_monitor)


def test_builtin_vault_preset_rule_actions_are_supported():
    for preset in get_builtin_presets():
        for rule in preset.rules:
            assert rule["action"] in ALLOWED_RULE_ACTIONS


def test_audit_only_preset_allows_without_blocking():
    audit_only = next(preset for preset in get_builtin_presets() if preset.id == "audit-only")

    assert audit_only.guard["mode"] == "permissive"
    assert {rule["action"] for rule in audit_only.rules} == {"allow"}


def test_strict_presets_include_block_rules():
    strict_presets = [
        preset for preset in get_builtin_presets() if preset.guard["mode"] == "strict"
    ]

    assert strict_presets
    for preset in strict_presets:
        assert any(rule["action"] == "block" for rule in preset.rules), preset.id


def test_load_settings_refreshes_builtins_and_preserves_custom_presets(tmp_path):
    old_builtins = get_builtin_presets()[:5]
    custom = VaultPreset(
        id="custom-team-vault",
        name="Custom Team Vault",
        description="User-created custom preset",
        icon="🧪",
        builtin=False,
        created_at="2026-05-08T00:00:00",
        detection=DetectionConfig().model_dump(mode="json"),
        guard=GuardConfig().model_dump(mode="json"),
        file_monitor=FileMonitorConfig().model_dump(mode="json"),
        rules=[],
    )
    config_path = tmp_path / "config.yaml"
    settings = load_settings(config_path)
    settings.vaults.presets = old_builtins + [custom]
    save_settings(settings, config_path)

    refreshed = load_settings(config_path)

    builtin_ids = [preset.id for preset in refreshed.vaults.presets if preset.builtin]
    custom_ids = [preset.id for preset in refreshed.vaults.presets if not preset.builtin]
    assert builtin_ids == EXPECTED_PRESET_IDS
    assert custom_ids == ["custom-team-vault"]

    saved = yaml.safe_load(config_path.read_text())
    saved_builtin_ids = [
        preset["id"] for preset in saved["vaults"]["presets"] if preset.get("builtin")
    ]
    assert saved_builtin_ids == EXPECTED_PRESET_IDS
