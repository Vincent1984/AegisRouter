"""Tests for /health/components endpoint (V5-5).

Verifies that the health endpoint correctly reports the status of
ClawVault, Redis, and RouteLLM components.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis_router.callbacks.degradation import ComponentState, DegradationManager
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.health import health_router

_PATCH_TARGET = "aegis_router.health._get_smart_router_instance"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a FastAPI app with the health router included."""
    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture
def mock_pool_healthy():
    """ClawVaultPool mock that responds to ping successfully."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock(return_value={"pong": True})
    return pool


@pytest.fixture
def mock_pool_down():
    """ClawVaultPool mock that simulates ClawVault being down."""
    pool = MagicMock(spec=ClawVaultPool)
    pool.call = AsyncMock(return_value=None)
    return pool


@pytest.fixture
def mock_redis_healthy():
    """Mock Redis client that reports healthy."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


@pytest.fixture
def mock_redis_down():
    """Mock Redis client that reports unhealthy."""
    client = MagicMock()
    client.health_check = AsyncMock(return_value={"status": "unhealthy", "error": "refused"})
    return client


@pytest.fixture
def mock_classifier_available():
    """Mock ModelClassifier that is available."""
    classifier = MagicMock()
    classifier.is_available = True
    return classifier


@pytest.fixture
def mock_classifier_down():
    """Mock ModelClassifier that is not available."""
    classifier = MagicMock()
    classifier.is_available = False
    return classifier


# ---------------------------------------------------------------------------
# Tests: All components healthy
# ---------------------------------------------------------------------------


class TestAllComponentsHealthy:
    """Endpoint returns correct status when all components are up."""

    def test_all_up_returns_ok(
        self,
        app,
        mock_pool_healthy,
        mock_redis_healthy,
        mock_classifier_available,
    ):
        """All components up -> status=ok, all components=up."""
        dm = DegradationManager(redis_client=mock_redis_healthy)

        mock_instance = MagicMock()
        mock_instance._pool = mock_pool_healthy
        mock_instance._degradation = dm
        mock_instance._classifier = mock_classifier_available

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["components"]["clawvault"] == "up"
        assert data["components"]["redis"] == "up"
        assert data["components"]["routellm"] == "up"


# ---------------------------------------------------------------------------
# Tests: ClawVault down
# ---------------------------------------------------------------------------


class TestClawVaultDown:
    """Endpoint returns 'down' for clawvault when it's unavailable."""

    def test_clawvault_down_returns_degraded(
        self,
        app,
        mock_pool_down,
        mock_redis_healthy,
        mock_classifier_available,
    ):
        """ClawVault down -> status=degraded, clawvault=down."""
        dm = DegradationManager(redis_client=mock_redis_healthy)

        mock_instance = MagicMock()
        mock_instance._pool = mock_pool_down
        mock_instance._degradation = dm
        mock_instance._classifier = mock_classifier_available

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["clawvault"] == "down"
        assert data["components"]["redis"] == "up"
        assert data["components"]["routellm"] == "up"

    def test_clawvault_pool_none_returns_down(
        self,
        app,
        mock_redis_healthy,
        mock_classifier_available,
    ):
        """Pool is None -> clawvault=down."""
        dm = DegradationManager(redis_client=mock_redis_healthy)

        mock_instance = MagicMock()
        mock_instance._pool = None
        mock_instance._degradation = dm
        mock_instance._classifier = mock_classifier_available

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["components"]["clawvault"] == "down"

    def test_clawvault_exception_returns_down(
        self,
        app,
        mock_redis_healthy,
        mock_classifier_available,
    ):
        """Pool.call raises exception -> clawvault=down."""
        pool = MagicMock(spec=ClawVaultPool)
        pool.call = AsyncMock(side_effect=OSError("connection refused"))

        dm = DegradationManager(redis_client=mock_redis_healthy)

        mock_instance = MagicMock()
        mock_instance._pool = pool
        mock_instance._degradation = dm
        mock_instance._classifier = mock_classifier_available

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["components"]["clawvault"] == "down"


# ---------------------------------------------------------------------------
# Tests: Redis down
# ---------------------------------------------------------------------------


class TestRedisDown:
    """Endpoint returns 'down' for redis when it's unavailable."""

    def test_redis_down_returns_degraded(
        self,
        app,
        mock_pool_healthy,
        mock_redis_down,
        mock_classifier_available,
    ):
        """Redis down -> status=degraded, redis=down."""
        dm = DegradationManager(redis_client=mock_redis_down)

        mock_instance = MagicMock()
        mock_instance._pool = mock_pool_healthy
        mock_instance._degradation = dm
        mock_instance._classifier = mock_classifier_available

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["clawvault"] == "up"
        assert data["components"]["redis"] == "down"
        assert data["components"]["routellm"] == "up"

    def test_redis_exception_returns_down(
        self,
        app,
        mock_pool_healthy,
        mock_classifier_available,
    ):
        """Redis health_check raises -> redis=down."""
        client_mock = MagicMock()
        client_mock.health_check = AsyncMock(side_effect=ConnectionError("lost"))
        dm = DegradationManager(redis_client=client_mock)

        mock_instance = MagicMock()
        mock_instance._pool = mock_pool_healthy
        mock_instance._degradation = dm
        mock_instance._classifier = mock_classifier_available

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["components"]["redis"] == "down"


# ---------------------------------------------------------------------------
# Tests: RouteLLM down
# ---------------------------------------------------------------------------


class TestRouteLLMDown:
    """Endpoint returns 'down' for routellm when classifier is unavailable."""

    def test_routellm_not_available_returns_degraded(
        self,
        app,
        mock_pool_healthy,
        mock_redis_healthy,
        mock_classifier_down,
    ):
        """Classifier not available -> status=degraded, routellm=down."""
        dm = DegradationManager(redis_client=mock_redis_healthy)

        mock_instance = MagicMock()
        mock_instance._pool = mock_pool_healthy
        mock_instance._degradation = dm
        mock_instance._classifier = mock_classifier_down

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["clawvault"] == "up"
        assert data["components"]["redis"] == "up"
        assert data["components"]["routellm"] == "down"

    def test_routellm_none_returns_down(
        self,
        app,
        mock_pool_healthy,
        mock_redis_healthy,
    ):
        """Classifier is None -> routellm=down."""
        dm = DegradationManager(redis_client=mock_redis_healthy)

        mock_instance = MagicMock()
        mock_instance._pool = mock_pool_healthy
        mock_instance._degradation = dm
        mock_instance._classifier = None

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["components"]["routellm"] == "down"


# ---------------------------------------------------------------------------
# Tests: Multiple components down
# ---------------------------------------------------------------------------


class TestMultipleComponentsDown:
    """Overall status is 'degraded' when any component is down."""

    def test_all_down_returns_degraded(self, app):
        """All components down -> status=degraded, all=down."""
        pool = MagicMock(spec=ClawVaultPool)
        pool.call = AsyncMock(return_value=None)

        redis_client = MagicMock()
        redis_client.health_check = AsyncMock(
            return_value={"status": "unhealthy", "error": "down"}
        )
        dm = DegradationManager(redis_client=redis_client)

        classifier = MagicMock()
        classifier.is_available = False

        mock_instance = MagicMock()
        mock_instance._pool = pool
        mock_instance._degradation = dm
        mock_instance._classifier = classifier

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["clawvault"] == "down"
        assert data["components"]["redis"] == "down"
        assert data["components"]["routellm"] == "down"

    def test_two_down_one_up(
        self, app, mock_pool_healthy, mock_redis_down
    ):
        """Two components down -> status=degraded."""
        dm = DegradationManager(redis_client=mock_redis_down)

        classifier = MagicMock()
        classifier.is_available = False

        mock_instance = MagicMock()
        mock_instance._pool = mock_pool_healthy
        mock_instance._degradation = dm
        mock_instance._classifier = classifier

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["clawvault"] == "up"
        assert data["components"]["redis"] == "down"
        assert data["components"]["routellm"] == "down"
