"""Minimal litellm stub - provides only interfaces used by aegis_router."""
from typing import List, Union, Callable, Literal, Optional

__version__ = "1.40.0"

failure_callback: List[Union[str, Callable]] = []
service_callback: List[Union[str, Callable]] = []
_custom_logger_compatible_callbacks_literal = Literal["lago", "openmeter"]
callbacks: List[Union[Callable, str]] = []


def run_server():
    """Placeholder for litellm server entry point (use CLI instead)."""
    from litellm.cli import main
    main()
