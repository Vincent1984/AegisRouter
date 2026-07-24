"""Health check endpoint for AegisRouter.

Provides a `/health/components` endpoint that reports the real-time status
of AegisRouter's core components:
- ClawVault (PII masking companion process)
- Redis (PII mapping storage)
- RouteLLM (model classifier for intelligent routing)

The endpoint performs live probes with a short timeout to avoid blocking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Health probe timeout (seconds)
_HEALTH_PROBE_TIMEOUT: float = 2.0

health_router = APIRouter(tags=["health"])


def _get_smart_router_instance():
    """Lazy accessor for the global smart_router_instance.

    Separated into its own function to make it easily patchable in tests.
    """
    from aegis_router.callbacks.smart_router import smart_router_instance

    return smart_router_instance


async def _probe_clawvault(pool: Any) -> str:
    """Probe ClawVault connectivity by sending a lightweight ping RPC.

    Returns "up" if ClawVault responds, "down" otherwise.
    """
    if pool is None:
        return "down"

    try:
        result = await asyncio.wait_for(
            pool.call("ping", {}, timeout=_HEALTH_PROBE_TIMEOUT),
            timeout=_HEALTH_PROBE_TIMEOUT,
        )
        # pool.call returns None when ClawVault is unavailable
        return "up" if result is not None else "down"
    except Exception:
        return "down"


async def _probe_redis(degradation_manager: Any) -> str:
    """Probe Redis by calling DegradationManager.check_redis_health().

    Returns "up" if Redis is healthy, "down" otherwise.
    """
    if degradation_manager is None:
        return "down"

    try:
        from aegis_router.callbacks.degradation import ComponentState

        state = await asyncio.wait_for(
            degradation_manager.check_redis_health(),
            timeout=_HEALTH_PROBE_TIMEOUT,
        )
        return "up" if state == ComponentState.HEALTHY else "down"
    except Exception:
        return "down"


def _probe_routellm(classifier: Any) -> str:
    """Check RouteLLM classifier availability.

    Returns "up" if classifier is loaded and available, "down" otherwise.
    """
    if classifier is None:
        return "down"

    try:
        return "up" if classifier.is_available else "down"
    except Exception:
        return "down"


@health_router.get("/health/components")
async def health_components() -> JSONResponse:
    """Return the health status of all AegisRouter components.

    Response format:
    ```json
    {
        "status": "ok" | "degraded",
        "components": {
            "clawvault": "up" | "down",
            "redis": "up" | "down",
            "routellm": "up" | "down"
        }
    }
    ```

    Always returns HTTP 200 — the system operates in degraded mode when
    components are down rather than becoming fully unavailable.
    """
    instance = _get_smart_router_instance()

    pool = instance._pool
    degradation = instance._degradation
    classifier = instance._classifier

    # Run probes concurrently
    clawvault_status, redis_status = await asyncio.gather(
        _probe_clawvault(pool),
        _probe_redis(degradation),
    )
    routellm_status = _probe_routellm(classifier)

    components = {
        "clawvault": clawvault_status,
        "redis": redis_status,
        "routellm": routellm_status,
    }

    all_up = all(v == "up" for v in components.values())
    overall_status = "ok" if all_up else "degraded"

    return JSONResponse(
        status_code=200,
        content={
            "status": overall_status,
            "components": components,
        },
    )
