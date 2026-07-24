"""Minimal proxy types stub."""
from typing import Optional


class UserAPIKeyAuth:
    """Stub for UserAPIKeyAuth used by CustomLogger."""
    api_key: Optional[str] = None
    user_id: Optional[str] = None
