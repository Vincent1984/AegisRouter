"""Tests for local filesystem scanner behavior."""

from claw_vault.local_scan.models import ScanStatus
from claw_vault.local_scan.scanner import LocalScanner


def test_scan_skills_detects_precompiled_risk_patterns(tmp_path):
    risky_skill = tmp_path / "risky_skill.py"
    risky_skill.write_text(
        "def run():\n"
        "    eval('1 + 1')\n"
        "    token = 'API_KEY_PLACEHOLDER'\n",
        encoding="utf-8",
    )

    result = LocalScanner().scan_skills(str(tmp_path))
    factors = {
        finding.detail["factor"]
        for finding in result.findings
        if finding.finding_type == "skill_risk"
    }

    assert result.status == ScanStatus.COMPLETED
    assert result.files_scanned == 1
    assert "credential_access" in factors
    assert "dynamic_eval" in factors


def test_scan_skills_leaves_benign_file_without_skill_risk(tmp_path):
    benign_skill = tmp_path / "benign_skill.py"
    benign_skill.write_text(
        "def run(value):\n"
        "    return value.strip().lower()\n",
        encoding="utf-8",
    )

    result = LocalScanner().scan_skills(str(tmp_path))

    assert result.status == ScanStatus.COMPLETED
    assert result.files_scanned == 1
    assert [finding for finding in result.findings if finding.finding_type == "skill_risk"] == []
