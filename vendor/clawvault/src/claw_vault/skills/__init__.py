"""ClawVault Skill framework and built-in Skills for OpenClaw."""

from claw_vault.skills.base import BaseSkill, SkillContext, SkillResult, tool
from claw_vault.skills.registry import SkillRegistry

__all__ = ["BaseSkill", "SkillContext", "SkillResult", "SkillRegistry", "tool"]
