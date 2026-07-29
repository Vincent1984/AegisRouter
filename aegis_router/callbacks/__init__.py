"""LiteLLM Custom Callbacks"""

from aegis_router.callbacks.base_router import BaseRouterCallback
from aegis_router.callbacks.degradation import (
    ComponentState,
    DegradationError,
    DegradationManager,
)
from aegis_router.callbacks.plugin_loader import (
    get_available_plugins,
    load_routing_plugin,
)
from aegis_router.callbacks.smart_router import SmartRouterCallback, smart_router_instance
from aegis_router.callbacks.transaction_router import TransactionRouterCallback
from aegis_router.callbacks.uds_pool import ClawVaultPool

__all__ = [
    "BaseRouterCallback",
    "ComponentState",
    "DegradationError",
    "DegradationManager",
    "SmartRouterCallback",
    "TransactionRouterCallback",
    "smart_router_instance",
    "ClawVaultPool",
    "get_available_plugins",
    "load_routing_plugin",
]
