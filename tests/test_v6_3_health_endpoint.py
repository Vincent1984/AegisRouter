"""V6-3 Verification: Container /health endpoint accessible and returns 200.

Validates:
1. LiteLLM stub's /health endpoint returns HTTP 200
2. /health/components endpoint returns HTTP 200 regardless of component state
3. Dockerfile HEALTHCHECK directive is correctly configured
4. LiteLLM stub handler responds to /health with expected JSON
"""

from __future__ import annotations

import http.server
import importlib
import json
import re
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.request import urlopen

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegis_router.health import health_router

# Path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import the LiteLLM stub handler from docker/stubs directory
_STUB_DIR = str(PROJECT_ROOT / "docker" / "stubs" / "litellm")


def _load_stub_handler():
    """Load LiteLLMStubHandler from the stub source directory."""
    if _STUB_DIR not in sys.path:
        sys.path.insert(0, _STUB_DIR)
    # Import fresh to avoid conflicts with any installed litellm package
    spec = importlib.util.spec_from_file_location(
        "litellm_stub_cli",
        str(PROJECT_ROOT / "docker" / "stubs" / "litellm" / "litellm" / "cli.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LiteLLMStubHandler

_PATCH_TARGET = "aegis_router.health._get_smart_router_instance"


# ---------------------------------------------------------------------------
# 1. Dockerfile HEALTHCHECK validation
# ---------------------------------------------------------------------------


class TestDockerfileHealthcheck:
    """Verify Dockerfile HEALTHCHECK directive is properly configured."""

    @pytest.fixture
    def dockerfile_content(self) -> str:
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile not found at project root"
        return dockerfile_path.read_text(encoding="utf-8")

    def test_healthcheck_uses_curl(self, dockerfile_content: str):
        """HEALTHCHECK uses curl -f to probe localhost:8000/health."""
        assert "curl -f http://localhost:8000/health" in dockerfile_content

    def test_healthcheck_has_interval(self, dockerfile_content: str):
        """HEALTHCHECK specifies --interval."""
        match = re.search(r"--interval=(\d+)s", dockerfile_content)
        assert match is not None, "HEALTHCHECK missing --interval"
        interval = int(match.group(1))
        assert interval > 0, "Interval must be positive"

    def test_healthcheck_has_timeout(self, dockerfile_content: str):
        """HEALTHCHECK specifies --timeout."""
        match = re.search(r"--timeout=(\d+)s", dockerfile_content)
        assert match is not None, "HEALTHCHECK missing --timeout"
        timeout = int(match.group(1))
        assert timeout > 0, "Timeout must be positive"

    def test_healthcheck_has_retries(self, dockerfile_content: str):
        """HEALTHCHECK specifies --retries."""
        match = re.search(r"--retries=(\d+)", dockerfile_content)
        assert match is not None, "HEALTHCHECK missing --retries"
        retries = int(match.group(1))
        assert retries >= 1, "Retries must be at least 1"

    def test_healthcheck_exit_on_failure(self, dockerfile_content: str):
        """HEALTHCHECK CMD exits with 1 on curl failure."""
        assert "|| exit 1" in dockerfile_content


# ---------------------------------------------------------------------------
# 2. LiteLLM stub /health endpoint
# ---------------------------------------------------------------------------


class TestLiteLLMStubHealthEndpoint:
    """Test the litellm stub's HTTP handler responds to /health with 200."""

    @pytest.fixture
    def stub_server(self):
        """Start the LiteLLM stub HTTP server on a random port."""
        handler_cls = _load_stub_handler()
        server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield port
        server.shutdown()

    def test_health_returns_200(self, stub_server: int):
        """GET /health returns HTTP 200."""
        response = urlopen(f"http://127.0.0.1:{stub_server}/health")
        assert response.status == 200

    def test_health_returns_json_body(self, stub_server: int):
        """GET /health returns JSON with status=healthy."""
        response = urlopen(f"http://127.0.0.1:{stub_server}/health")
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "healthy"
        assert data["stub"] is True

    def test_health_content_type_json(self, stub_server: int):
        """GET /health returns application/json content type."""
        response = urlopen(f"http://127.0.0.1:{stub_server}/health")
        content_type = response.headers.get("Content-Type", "")
        assert "application/json" in content_type


# ---------------------------------------------------------------------------
# 3. /health/components endpoint returns 200 regardless of state
# ---------------------------------------------------------------------------


class TestHealthComponentsAlwaysReturns200:
    """Verify /health/components always returns HTTP 200."""

    @pytest.fixture
    def app(self):
        app = FastAPI()
        app.include_router(health_router)
        return app

    def _make_mock_instance(self, pool=None, degradation=None, classifier=None):
        """Create a mock smart_router_instance with given components."""
        instance = MagicMock()
        instance._pool = pool
        instance._degradation = degradation
        instance._classifier = classifier
        return instance

    def test_all_components_up_returns_200(self, app):
        """All healthy -> HTTP 200 with status=ok."""
        pool = MagicMock()
        pool.call = AsyncMock(return_value={"pong": True})

        redis_client = MagicMock()
        redis_client.health_check = AsyncMock(return_value={"status": "healthy"})

        from aegis_router.callbacks.degradation import DegradationManager

        dm = DegradationManager(redis_client=redis_client)

        classifier = MagicMock()
        classifier.is_available = True

        mock_instance = self._make_mock_instance(pool, dm, classifier)

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_all_components_down_still_returns_200(self, app):
        """All down -> HTTP 200 with status=degraded (never 5xx)."""
        pool = MagicMock()
        pool.call = AsyncMock(return_value=None)

        redis_client = MagicMock()
        redis_client.health_check = AsyncMock(
            return_value={"status": "unhealthy", "error": "down"}
        )

        from aegis_router.callbacks.degradation import DegradationManager

        dm = DegradationManager(redis_client=redis_client)

        classifier = MagicMock()
        classifier.is_available = False

        mock_instance = self._make_mock_instance(pool, dm, classifier)

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_partial_failure_returns_200(self, app):
        """Some down, some up -> still HTTP 200."""
        pool = MagicMock()
        pool.call = AsyncMock(return_value={"pong": True})

        redis_client = MagicMock()
        redis_client.health_check = AsyncMock(side_effect=ConnectionError("lost"))

        from aegis_router.callbacks.degradation import DegradationManager

        dm = DegradationManager(redis_client=redis_client)

        mock_instance = self._make_mock_instance(pool, dm, None)

        with patch(_PATCH_TARGET, return_value=mock_instance):
            client = TestClient(app)
            response = client.get("/health/components")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["clawvault"] == "up"
        assert data["components"]["redis"] == "down"
        assert data["components"]["routellm"] == "down"
