"""Tests for Kubernetes manifest validation (V6-5).

Validates that all K8s YAML manifests in k8s/ directory are correct,
consistent, and would deploy 3 Ready pods.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

K8S_DIR = Path(__file__).parent.parent / "k8s"

MANIFEST_FILES = [
    "deployment.yaml",
    "service.yaml",
    "configmap.yaml",
    "hpa.yaml",
    "pdb.yaml",
    "secret.yaml",
]


def load_manifest(filename: str) -> dict:
    """Load and parse a YAML manifest file."""
    filepath = K8S_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tests: YAML Parsing
# ---------------------------------------------------------------------------


class TestYAMLParsing:
    """All K8s YAML files must parse without errors."""

    @pytest.mark.parametrize("filename", MANIFEST_FILES)
    def test_yaml_file_exists(self, filename: str):
        """Each manifest file exists in k8s/ directory."""
        filepath = K8S_DIR / filename
        assert filepath.exists(), f"{filename} not found in {K8S_DIR}"

    @pytest.mark.parametrize("filename", MANIFEST_FILES)
    def test_yaml_parses_without_error(self, filename: str):
        """Each manifest file parses as valid YAML."""
        manifest = load_manifest(filename)
        assert manifest is not None, f"{filename} parsed as empty/None"
        assert isinstance(manifest, dict), f"{filename} did not parse as a dict"


# ---------------------------------------------------------------------------
# Tests: Required Fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """All manifests must contain required K8s fields."""

    @pytest.mark.parametrize("filename", MANIFEST_FILES)
    def test_has_api_version(self, filename: str):
        """Each manifest must have apiVersion."""
        manifest = load_manifest(filename)
        assert "apiVersion" in manifest, f"{filename} missing apiVersion"

    @pytest.mark.parametrize("filename", MANIFEST_FILES)
    def test_has_kind(self, filename: str):
        """Each manifest must have kind."""
        manifest = load_manifest(filename)
        assert "kind" in manifest, f"{filename} missing kind"

    @pytest.mark.parametrize("filename", MANIFEST_FILES)
    def test_has_metadata(self, filename: str):
        """Each manifest must have metadata."""
        manifest = load_manifest(filename)
        assert "metadata" in manifest, f"{filename} missing metadata"
        assert "name" in manifest["metadata"], f"{filename} missing metadata.name"

    @pytest.mark.parametrize(
        "filename",
        ["deployment.yaml", "service.yaml", "hpa.yaml", "pdb.yaml"],
    )
    def test_has_spec(self, filename: str):
        """Resources with behavior must have spec."""
        manifest = load_manifest(filename)
        assert "spec" in manifest, f"{filename} missing spec"


# ---------------------------------------------------------------------------
# Tests: Deployment Configuration
# ---------------------------------------------------------------------------


class TestDeploymentConfig:
    """Deployment must be correctly configured for 3 replicas."""

    @pytest.fixture
    def deployment(self) -> dict:
        return load_manifest("deployment.yaml")

    def test_kind_is_deployment(self, deployment):
        """Kind must be Deployment."""
        assert deployment["kind"] == "Deployment"

    def test_api_version(self, deployment):
        """API version must be apps/v1."""
        assert deployment["apiVersion"] == "apps/v1"

    def test_replicas_is_3(self, deployment):
        """Deployment must specify 3 replicas."""
        assert deployment["spec"]["replicas"] == 3

    def test_has_selector(self, deployment):
        """Deployment must have a selector."""
        selector = deployment["spec"]["selector"]
        assert "matchLabels" in selector
        assert selector["matchLabels"]["app"] == "aegis-router"

    def test_template_labels_match_selector(self, deployment):
        """Pod template labels must match deployment selector."""
        selector_labels = deployment["spec"]["selector"]["matchLabels"]
        template_labels = deployment["spec"]["template"]["metadata"]["labels"]
        for key, value in selector_labels.items():
            assert key in template_labels
            assert template_labels[key] == value

    def test_container_has_readiness_probe(self, deployment):
        """Container must have readinessProbe for Ready status."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "readinessProbe" in container
        probe = container["readinessProbe"]
        assert "httpGet" in probe
        assert probe["httpGet"]["path"] == "/health"
        assert probe["httpGet"]["port"] == 8000

    def test_container_has_liveness_probe(self, deployment):
        """Container must have livenessProbe for restart detection."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "livenessProbe" in container
        probe = container["livenessProbe"]
        assert "httpGet" in probe
        assert probe["httpGet"]["path"] == "/health"

    def test_container_has_resource_requests(self, deployment):
        """Container must have resource requests for scheduling."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "resources" in container
        assert "requests" in container["resources"]
        assert "cpu" in container["resources"]["requests"]
        assert "memory" in container["resources"]["requests"]

    def test_container_has_resource_limits(self, deployment):
        """Container must have resource limits."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "limits" in container["resources"]
        assert "cpu" in container["resources"]["limits"]
        assert "memory" in container["resources"]["limits"]

    def test_rolling_update_strategy(self, deployment):
        """Deployment must use RollingUpdate strategy."""
        strategy = deployment["spec"]["strategy"]
        assert strategy["type"] == "RollingUpdate"
        assert strategy["rollingUpdate"]["maxUnavailable"] == 0


# ---------------------------------------------------------------------------
# Tests: ConfigMap Consistency
# ---------------------------------------------------------------------------


class TestConfigMapConsistency:
    """ConfigMap name must match the volume reference in deployment."""

    def test_configmap_name(self):
        """ConfigMap must be named aegis-router-config."""
        cm = load_manifest("configmap.yaml")
        assert cm["metadata"]["name"] == "aegis-router-config"

    def test_configmap_name_matches_deployment_volume(self):
        """ConfigMap name in deployment volume must match actual ConfigMap."""
        cm = load_manifest("configmap.yaml")
        deployment = load_manifest("deployment.yaml")

        cm_name = cm["metadata"]["name"]

        volumes = deployment["spec"]["template"]["spec"]["volumes"]
        config_volume = next(
            (v for v in volumes if v.get("configMap")), None
        )
        assert config_volume is not None, "No configMap volume found in deployment"
        assert config_volume["configMap"]["name"] == cm_name

    def test_configmap_has_data(self):
        """ConfigMap must have data section."""
        cm = load_manifest("configmap.yaml")
        assert "data" in cm
        assert "config.yaml" in cm["data"]


# ---------------------------------------------------------------------------
# Tests: Secret Consistency
# ---------------------------------------------------------------------------


class TestSecretConsistency:
    """Secret name must match the envFrom reference in deployment."""

    def test_secret_name(self):
        """Secret must be named aegis-router-secrets."""
        secret = load_manifest("secret.yaml")
        assert secret["metadata"]["name"] == "aegis-router-secrets"

    def test_secret_name_matches_deployment_envfrom(self):
        """Secret name in deployment envFrom must match actual Secret."""
        secret = load_manifest("secret.yaml")
        deployment = load_manifest("deployment.yaml")

        secret_name = secret["metadata"]["name"]

        container = deployment["spec"]["template"]["spec"]["containers"][0]
        env_from = container.get("envFrom", [])
        secret_refs = [
            e["secretRef"]["name"] for e in env_from if "secretRef" in e
        ]
        assert secret_name in secret_refs, (
            f"Secret '{secret_name}' not referenced in deployment envFrom. "
            f"Found: {secret_refs}"
        )

    def test_secret_has_string_data(self):
        """Secret must have stringData with required keys."""
        secret = load_manifest("secret.yaml")
        assert "stringData" in secret
        required_keys = ["AEGIS_MASTER_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
        for key in required_keys:
            assert key in secret["stringData"], f"Secret missing key: {key}"


# ---------------------------------------------------------------------------
# Tests: Service Consistency
# ---------------------------------------------------------------------------


class TestServiceConsistency:
    """Service selector must match deployment labels."""

    def test_service_selector_matches_deployment(self):
        """Service selector app: aegis-router must match deployment labels."""
        service = load_manifest("service.yaml")
        deployment = load_manifest("deployment.yaml")

        svc_selector = service["spec"]["selector"]
        deploy_labels = deployment["spec"]["template"]["metadata"]["labels"]

        for key, value in svc_selector.items():
            assert key in deploy_labels, (
                f"Service selector key '{key}' not in deployment labels"
            )
            assert deploy_labels[key] == value, (
                f"Service selector {key}={value} doesn't match "
                f"deployment label {key}={deploy_labels[key]}"
            )

    def test_service_type(self):
        """Service must be ClusterIP type."""
        service = load_manifest("service.yaml")
        assert service["spec"]["type"] == "ClusterIP"

    def test_service_port_matches_container(self):
        """Service targetPort must match container port."""
        service = load_manifest("service.yaml")
        deployment = load_manifest("deployment.yaml")

        svc_target_port = service["spec"]["ports"][0]["targetPort"]
        container_port = deployment["spec"]["template"]["spec"]["containers"][0]["ports"][0]["containerPort"]
        assert svc_target_port == container_port


# ---------------------------------------------------------------------------
# Tests: PDB Consistency
# ---------------------------------------------------------------------------


class TestPDBConsistency:
    """PodDisruptionBudget selector must match deployment labels."""

    def test_pdb_selector_matches_deployment(self):
        """PDB selector must match deployment pod labels."""
        pdb = load_manifest("pdb.yaml")
        deployment = load_manifest("deployment.yaml")

        pdb_selector = pdb["spec"]["selector"]["matchLabels"]
        deploy_labels = deployment["spec"]["template"]["metadata"]["labels"]

        for key, value in pdb_selector.items():
            assert key in deploy_labels, (
                f"PDB selector key '{key}' not in deployment labels"
            )
            assert deploy_labels[key] == value

    def test_pdb_min_available(self):
        """PDB minAvailable must be less than replicas."""
        pdb = load_manifest("pdb.yaml")
        deployment = load_manifest("deployment.yaml")

        min_available = pdb["spec"]["minAvailable"]
        replicas = deployment["spec"]["replicas"]
        assert min_available < replicas, (
            f"PDB minAvailable ({min_available}) must be < replicas ({replicas})"
        )


# ---------------------------------------------------------------------------
# Tests: HPA Consistency
# ---------------------------------------------------------------------------


class TestHPAConsistency:
    """HPA must reference the correct deployment."""

    def test_hpa_references_correct_deployment(self):
        """HPA scaleTargetRef must point to aegis-router deployment."""
        hpa = load_manifest("hpa.yaml")
        deployment = load_manifest("deployment.yaml")

        target_ref = hpa["spec"]["scaleTargetRef"]
        assert target_ref["apiVersion"] == "apps/v1"
        assert target_ref["kind"] == "Deployment"
        assert target_ref["name"] == deployment["metadata"]["name"]

    def test_hpa_min_replicas_matches_deployment(self):
        """HPA minReplicas should match deployment replicas."""
        hpa = load_manifest("hpa.yaml")
        deployment = load_manifest("deployment.yaml")

        assert hpa["spec"]["minReplicas"] == deployment["spec"]["replicas"]

    def test_hpa_max_replicas_greater_than_min(self):
        """HPA maxReplicas must be greater than minReplicas."""
        hpa = load_manifest("hpa.yaml")
        assert hpa["spec"]["maxReplicas"] > hpa["spec"]["minReplicas"]

    def test_hpa_has_metrics(self):
        """HPA must have at least one metric defined."""
        hpa = load_manifest("hpa.yaml")
        assert "metrics" in hpa["spec"]
        assert len(hpa["spec"]["metrics"]) > 0


# ---------------------------------------------------------------------------
# Tests: Cross-Resource Naming Consistency
# ---------------------------------------------------------------------------


class TestCrossResourceNaming:
    """All resources use consistent naming conventions."""

    def test_all_resources_use_aegis_router_prefix(self):
        """All resource names should use aegis-router prefix."""
        expected_names = {
            "deployment.yaml": "aegis-router",
            "service.yaml": "aegis-router",
            "configmap.yaml": "aegis-router-config",
            "secret.yaml": "aegis-router-secrets",
            "hpa.yaml": "aegis-router-hpa",
            "pdb.yaml": "aegis-router-pdb",
        }
        for filename, expected_name in expected_names.items():
            manifest = load_manifest(filename)
            actual_name = manifest["metadata"]["name"]
            assert actual_name == expected_name, (
                f"{filename}: expected name '{expected_name}', got '{actual_name}'"
            )

    def test_deployment_app_label_consistent(self):
        """The app label is consistent across all selectors."""
        deployment = load_manifest("deployment.yaml")
        service = load_manifest("service.yaml")
        pdb = load_manifest("pdb.yaml")

        app_label = "aegis-router"

        # Deployment metadata labels
        assert deployment["metadata"]["labels"]["app"] == app_label
        # Deployment selector
        assert deployment["spec"]["selector"]["matchLabels"]["app"] == app_label
        # Pod template labels
        assert deployment["spec"]["template"]["metadata"]["labels"]["app"] == app_label
        # Service selector
        assert service["spec"]["selector"]["app"] == app_label
        # PDB selector
        assert pdb["spec"]["selector"]["matchLabels"]["app"] == app_label
