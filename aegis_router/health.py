"""Health check endpoint for AegisRouter.

Provides a `/health/components` endpoint that reports the real-time status
of AegisRouter's core components:
- ClawVault (PII masking companion process)
- Redis (PII mapping storage)
- RouteLLM (model classifier for intelligent routing)

Provides a `/health/routing` endpoint that reports the active routing plugin
type and plan summary (FR-9.1, FR-9.3).

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


# ---------------------------------------------------------------------------
# /health/routing — Active routing plugin info + plan summary (FR-9.1, FR-9.3)
# ---------------------------------------------------------------------------


def _get_routing_plugin_info() -> tuple[str, "Any"]:
    """Lazy accessor for the active routing plugin type and instance.

    Returns:
        Tuple of (plugin_type, plugin_instance).
    """
    from aegis_router.callbacks.plugin_loader import (
        get_active_plugin_instance,
        get_active_plugin_type,
    )

    return get_active_plugin_type(), get_active_plugin_instance()


def _build_plan_summary(plugin_instance: Any) -> dict[str, Any] | None:
    """Build a plan summary dict from a TransactionRouterCallback instance.

    Returns None if the instance is not a transaction plugin or has no plan_store.
    """
    from aegis_router.callbacks.transaction_router import TransactionRouterCallback

    if not isinstance(plugin_instance, TransactionRouterCallback):
        return None

    plan_store = plugin_instance.plan_store
    all_plans = plan_store.get_all_plans()

    total_templates = len(all_plans)
    total_mappings = len(plan_store)

    # Per-template agent count
    per_template: dict[str, int] = {
        tpl: len(agents) for tpl, agents in all_plans.items()
    }

    # Cross-template agent comparison (FR-9.3):
    # Find agents that appear in multiple templates
    agent_across_templates: dict[str, dict[str, str]] = {}
    for tpl, agents in all_plans.items():
        for agent, model in agents.items():
            agent_across_templates.setdefault(agent, {})[tpl] = model

    # Only keep agents that appear in more than one template
    cross_template_agents = {
        agent: mappings
        for agent, mappings in agent_across_templates.items()
        if len(mappings) > 1
    }

    return {
        "total_templates": total_templates,
        "total_agent_model_mappings": total_mappings,
        "per_template_agent_count": per_template,
        "plan_table": all_plans,
        "cross_template_agents": cross_template_agents,
    }


@health_router.get("/health/routing")
async def health_routing() -> JSONResponse:
    """Return the active routing plugin type and plan summary.

    Response format (transaction plugin):
    ```json
    {
        "routing_plugin": "transaction",
        "plan_summary": {
            "total_templates": 4,
            "total_agent_model_mappings": 13,
            "per_template_agent_count": {"resume_screening": 4, ...},
            "plan_table": {"resume_screening": {"intent_classifier": "local-7b", ...}, ...},
            "cross_template_agents": {"compliance_checker": {"resume_screening": "deepseek-v4-pro", ...}}
        }
    }
    ```

    Response format (conversation plugin):
    ```json
    {
        "routing_plugin": "conversation",
        "plan_summary": null
    }
    ```

    Always returns HTTP 200.
    """
    plugin_type, plugin_instance = _get_routing_plugin_info()

    plan_summary = None
    if plugin_instance is not None:
        plan_summary = _build_plan_summary(plugin_instance)

    return JSONResponse(
        status_code=200,
        content={
            "routing_plugin": plugin_type,
            "plan_summary": plan_summary,
        },
    )
