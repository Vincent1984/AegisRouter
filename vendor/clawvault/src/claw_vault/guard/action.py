"""Action definitions for guard decisions."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class Action(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    ASK_USER = "ask_user"


@dataclass
class ActionResult:
    action: Action
    reason: str
    risk_score: float
    details: list[str]
    modified_text: str | None = None  # set when action == SANITIZE
