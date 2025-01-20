# Copyright 2021 Canonical Ltd.
# See LICENSE file for licensing details.

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from charmed_kubeflow_chisme.exceptions import ErrorWithStatus
from ops.model import ActiveStatus, BlockedStatus, WaitingStatus
from ops.pebble import ChangeError, Service
from ops.testing import Harness
from serialized_data_interface import NoCompatibleVersions, NoVersionsListed

from charm import MlflowCharm

EXPECTED_SERVICE = {
    "mlflow-server": Service(
        "mlflow-server",
        raw={
            "summary": "Entrypoint of mlflow-server image",
            "startup": "enabled",
            "override": "replace",
            "command": "mlflow server --host 0.0.0.0 --port 5000  --backend-store-uri test --default-artifact-root s3:/// --expose-prometheus /metrics",  # noqa: E501
            "environment": {"MLFLOW_TRACKING_URI": "test"},
        },
    )
}
EXPECTED_SERVICE_ARTIFACTS = {
    "mlflow-server": Service(
        "mlflow-server",
        raw={
            "summary": "Entrypoint of mlflow-server image",
            "startup": "enabled",
            "override": "replace",
            "command": "mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts --artifacts-destination s3:/// --backend-store-uri test --default-artifact-root s3:/// --expose-prometheus /metrics",  # noqa: E501
            "environment": {"MLFLOW_TRACKING_URI": "test"},
        },
    )
}
BUCKET_NAME = "mlflow"
CHARM_NAME = "mlflow-server"

OBJECT_STORAGE_DATA = {
    "access-key": "minio-access-key",
    "namespace": "namespace",
    "port": 1234,
    "secret-key": "minio-super-secret-key",
    "secure": True,
    "service": "service",
}

RELATIONAL_DB_DATA = {
    "database": "database",
    "host": "host",
    "username": "username",
    "password": "lorem-ipsum",
    "port": "port",
}

EXPECTED_ENVIRONMENT = {
    "AWS_ACCESS_KEY_ID": "minio-access-key",
    "AWS_ENDPOINT_URL": "http://service.namespace:1234",
    "AWS_SECRET_ACCESS_KEY": "minio-super-secret-key",
    "DB_ROOT_PASSWORD": "lorem-ipsum",
    "MLFLOW_S3_ENDPOINT_URL": "http://service.namespace:1234",
    "MLFLOW_TRACKING_URI": "mysql+pymysql://username:lorem-ipsum@host:port/mlflow",
    "USE_SSL": "true",
}

SECRETS_TEST_FILES = ["tests/test_data/secret.yaml.j2"]

INGRESS_DATA = {
    "prefix": "/mlflow/",
    "rewrite": "/",
    "service": "mlflow-server",
    "namespace": None,
    "port": 5000,
}


class _FakeChangeError(ChangeError):
    """Used to simulate a ChangeError during testing."""

    def __init__(self, err, change):
        super().__init__(err, change)


@pytest.fixture(scope="function")
def harness() -> Harness:
    """Create and return Harness for testing."""

    harness = Harness(MlflowCharm)

    # setup container networking simulation
    harness.set_can_connect("mlflow-server", True)

    return harness


def enable_exporter_container(harness: harness) -> Harness:
    """Enable mlflow-prometheus-exporter for connections."""
    harness.set_can_connect("mlflow-prometheus-exporter", True)
    return harness


def add_object_storage_to_harness(harness: Harness):
    """Helper function to handle object storage relation"""
    object_storage_data = {"_supported_versions": "- v1", "data": yaml.dump(OBJECT_STORAGE_DATA)}
    harness.set_leader(True)
    object_storage_relation_id = harness.add_relation("object-storage", "storage-provider")
    harness.add_relation_unit(object_storage_relation_id, "storage-provider/0")
    harness.update_relation_data(
        object_storage_relation_id, "storage-provider", object_storage_data
    )
    return harness


class TestCharm:
    """Test class for TrainingOperatorCharm."""

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_log_forwarding(self, harness: Harness):
        with patch("charm.LogForwarder") as mock_logging:
            harness.begin()
            mock_logging.assert_called_once_with(charm=harness.charm)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_check_leader_failure(self, harness: Harness):
        harness.begin_with_initial_hooks()
        assert harness.charm.model.unit.status == WaitingStatus("Waiting for leadership")

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_check_leader_success(self, harness: Harness):
        harness.set_leader(True)
        harness.begin_with_initial_hooks()
        assert harness.charm.model.unit.status != WaitingStatus("Waiting for leadership")

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def tests_on_pebble_ready_failure(self):
        harness = Harness(MlflowCharm)
        harness.set_can_connect("mlflow-server", False)
        harness.begin()
        with pytest.raises(ErrorWithStatus):
            harness.charm._on_pebble_ready(None)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def tests_on_pebble_ready_success(self, harness: Harness):
        harness.begin()
        harness.charm._on_event = MagicMock()
        harness.charm._on_pebble_ready(None)
        harness.charm._on_event.assert_called()

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.get_interfaces")
    def test_get_interfaces_failure_no_versions_listed(
        self, get_interfaces: MagicMock, harness: Harness
    ):
        relation = MagicMock()
        relation.name = "A"
        relation.id = "1"
        get_interfaces.side_effect = NoVersionsListed(relation)
        harness.begin()
        with pytest.raises(ErrorWithStatus) as e_info:
            harness.charm._get_interfaces()

        assert e_info.value.status_type(WaitingStatus)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.get_interfaces")
    def test_get_interfaces_failure_no_compatible_versions(
        self, get_interfaces: MagicMock, harness: Harness
    ):
        relation_error = MagicMock()
        relation_error.name = "A"
        relation_error.id = "1"
        get_interfaces.side_effect = NoCompatibleVersions(relation_error, [], [])
        harness.begin()
        with pytest.raises(ErrorWithStatus) as e_info:
            harness.charm._get_interfaces()

        assert e_info.value.status_type(BlockedStatus)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_get_interfaces_success(self, harness: Harness):
        harness = add_object_storage_to_harness(harness)
        harness.set_leader(True)
        harness.begin()
        interfaces = harness.charm._get_interfaces()
        assert interfaces["object-storage"] is not None

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.MlflowCharm._get_interfaces")
    def test_get_object_storage_data_failure_missing_storage_object(
        self, _get_interfaces: MagicMock, harness: Harness
    ):
        _get_interfaces.return_value = {"object-storage": ""}
        harness.set_leader(True)
        harness.begin_with_initial_hooks()
        assert harness.charm.model.unit.status == WaitingStatus(
            "Waiting for object-storage relation data"
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.MlflowCharm._get_interfaces")
    def test_get_object_storage_data_failure_bad_storage_object(
        self, _get_interfaces: MagicMock, harness: Harness
    ):
        storage_object = MagicMock()
        storage_object.get_data.return_value = ["a"]
        _get_interfaces.return_value = {"object-storage": storage_object}
        harness.set_leader(True)
        harness.begin_with_initial_hooks()
        assert harness.charm.model.unit.status == BlockedStatus(
            "Unexpected error unpacking object storage data - data format not as expected. "
            "Caught exception: ''list' object has no attribute 'values''"
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_get_object_storage_data_success(self, harness: Harness):
        harness = add_object_storage_to_harness(harness)
        harness.begin_with_initial_hooks()
        assert harness.charm.model.unit.status == BlockedStatus(
            "Please add relation to the database"
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_get_relational_db_data_success(self, harness: Harness):
        database = MagicMock()
        fetch_relation_data = MagicMock()
        fetch_relation_data.return_value = {
            "test-db-data": {
                "endpoints": "host:port",
                "username": "username",
                "password": "password",
            }
        }
        database.fetch_relation_data = fetch_relation_data
        harness.model.get_relation = MagicMock()
        harness.begin()
        harness.charm.database = database
        res = harness.charm._get_relational_db_data()
        assert res == {
            "host": "host",
            "password": "password",
            "port": "port",
            "username": "username",
        }

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_get_relational_db_data_failure_wrong_data(self, harness: Harness):
        """Test with missing username and password in databag"""
        database = MagicMock()
        fetch_relation_data = MagicMock()
        fetch_relation_data.return_value = {"test-db-data": {"endpoints": "host:port"}}
        database.fetch_relation_data = fetch_relation_data
        harness.model.get_relation = MagicMock()
        harness.begin()
        harness.charm.database = database
        with pytest.raises(ErrorWithStatus) as e_info:
            harness.charm._get_relational_db_data()
        assert e_info.value.status_type(WaitingStatus)
        assert "Incorrect data found in relation relational-db" in str(e_info)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_get_relational_db_data_failure_waiting(self, harness: Harness):
        database = MagicMock()
        fetch_relation_data = MagicMock()
        fetch_relation_data.return_value = {}
        database.fetch_relation_data = fetch_relation_data
        harness.begin()
        harness.charm.database = database
        with pytest.raises(ErrorWithStatus) as e_info:
            harness.charm._get_relational_db_data()

        assert e_info.value.status_type(BlockedStatus)
        assert "Please add relation to the database" in str(e_info)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.validate_s3_bucket_name")
    def test_validate_default_s3_bucket_failure_invalid_bucket(
        self, validate_s3_bucket_name: MagicMock, harness: Harness
    ):
        validate_s3_bucket_name.return_value = False
        harness.begin()
        with pytest.raises(ErrorWithStatus) as exc_info:
            harness.charm._validate_default_s3_bucket_name_and_access(BUCKET_NAME, None)
        assert "Invalid value for config default_artifact_root" in str(exc_info)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.validate_s3_bucket_name")
    def test_validate_default_s3_bucket_success_bucket_not_accessible(
        self,
        validate_s3_bucket_name: MagicMock,
        harness: Harness,
    ):
        s3_wrapper = MagicMock()
        s3_wrapper.check_if_bucket_accessible.return_value = False
        validate_s3_bucket_name.return_value = True
        harness.begin()
        value = harness.charm._validate_default_s3_bucket_name_and_access(BUCKET_NAME, s3_wrapper)
        assert not value

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.validate_s3_bucket_name")
    def test_validate_default_s3_bucket_success_bucket_accessible(
        self,
        validate_s3_bucket_name: MagicMock,
        harness: Harness,
    ):
        s3_wrapper = MagicMock()
        s3_wrapper.check_if_bucket_accessible.return_value = True
        validate_s3_bucket_name.return_value = True
        harness.begin()
        value = harness.charm._validate_default_s3_bucket_name_and_access(BUCKET_NAME, s3_wrapper)
        assert value

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.validate_s3_bucket_name")
    def test_validate_default_s3_bucket_failure_wrong_name(
        self, validate_s3_bucket_name: MagicMock, harness: Harness
    ):
        validate_s3_bucket_name.return_value = False
        harness.begin()
        with pytest.raises(ErrorWithStatus) as exc_info:
            harness.charm._validate_default_s3_bucket_name_and_access(BUCKET_NAME, None)
        assert exc_info.value.status_type(WaitingStatus)
        assert "Invalid value for config default_artifact_root" in str(exc_info)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_validate_default_s3_bucket_failure_bucket_creation_not_allowed(
        self,
        harness: Harness,
    ):
        harness.update_config({"create_default_artifact_root_if_missing": False})
        s3_wrapper = MagicMock()
        check_if_bucket_accessible = MagicMock()
        check_if_bucket_accessible.return_value = False
        s3_wrapper.check_if_bucket_accessible = check_if_bucket_accessible
        harness.begin()
        with pytest.raises(ErrorWithStatus) as exc_info:
            harness.charm._validate_default_s3_bucket_name_and_access(BUCKET_NAME, s3_wrapper)

        assert exc_info.value.status_type(BlockedStatus)
        assert "Error with default S3 artifact store - " in str(exc_info)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.MlflowCharm.container")
    def test_update_layer_failure_container_problem(
        self,
        container: MagicMock,
        harness: Harness,
    ):
        change = MagicMock()
        change.tasks = []
        container.replan.side_effect = _FakeChangeError("Fake problem during layer update", change)
        harness.begin()
        with pytest.raises(ErrorWithStatus) as exc_info:
            harness.charm._update_layer(container, harness.charm._container_name, MagicMock())

        assert exc_info.value.status_type(BlockedStatus)
        assert "Failed to replan with error: " in str(exc_info)

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_update_layer_success(
        self,
        harness: Harness,
    ):
        harness.begin()
        harness.charm._update_layer(
            harness.charm.container,
            harness.charm._container_name,
            harness.charm._charmed_mlflow_layer({"MLFLOW_TRACKING_URI": "test"}, ""),
        )
        assert harness.charm.container.get_plan().services == EXPECTED_SERVICE

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_config_artifact_success(
        self,
        harness: Harness,
    ):
        harness.update_config({"serve_artifacts": True, "artifacts_destination": "s3:///"})
        harness.begin()
        harness.charm._update_layer(
            harness.charm.container,
            harness.charm._container_name,
            harness.charm._charmed_mlflow_layer({"MLFLOW_TRACKING_URI": "test"}, ""),
        )
        updated_plan = harness.get_container_pebble_plan('mlflow-server').to_dict()
        assert harness.charm.container.get_plan().services == EXPECTED_SERVICE_ARTIFACTS

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_get_env_vars(
        self,
        harness: Harness,
    ):
        harness.begin()
        envs = harness.charm._get_env_vars(RELATIONAL_DB_DATA, OBJECT_STORAGE_DATA)
        assert envs == EXPECTED_ENVIRONMENT

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_create_manifests(self, harness: Harness):
        secrets_context = {
            "access_key": "a",
            "secret_access_key": "s",
        }
        harness.begin()
        manifests_items = harness.charm._create_manifests(SECRETS_TEST_FILES, secrets_context)
        manifests_as_json = json.dumps([item.manifest for item in manifests_items])
        assert (
            manifests_as_json
            == '[{"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "mlpipeline-minio-artifact"}, "stringData": {"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s"}}]'  # noqa: E501
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch("charm.MlflowCharm._create_manifests")
    @patch("charm.MlflowCharm.secrets_manifests_wrapper")
    def test_send_manifests(
        self, secrets_manifests_wrapper: MagicMock, create_manifests: MagicMock, harness: Harness
    ):
        tmp_manifests = "[]"
        create_manifests.return_value = tmp_manifests
        secrets_manifests_wrapper = MagicMock()
        harness.begin()
        harness.charm._send_manifests({}, [""], secrets_manifests_wrapper)
        secrets_manifests_wrapper.send_data.assert_called_once()

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch(
        "charm.MlflowCharm._validate_default_s3_bucket_name_and_access", lambda *args, **kw: True
    )
    @patch(
        "charm.S3BucketWrapper.__init__",
        lambda *args, **kw: None,
    )
    @patch("charm.MlflowCharm._get_object_storage_data", return_value=OBJECT_STORAGE_DATA)
    @patch("charm.MlflowCharm._get_relational_db_data", return_value=RELATIONAL_DB_DATA)
    def test_on_event_wainting_for_exporter(
        self,
        _: MagicMock,
        __: MagicMock,
        harness: Harness,
    ):
        harness.set_leader(True)
        harness.begin()
        harness.charm._on_event(None)
        assert harness.charm.model.unit.status == WaitingStatus(
            "Container mlflow-prometheus-exporter is not ready"
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    @patch(
        "charm.MlflowCharm._validate_default_s3_bucket_name_and_access", lambda *args, **kw: True
    )
    @patch(
        "charm.S3BucketWrapper.__init__",
        lambda *args, **kw: None,
    )
    @patch("charm.MlflowCharm._get_object_storage_data", return_value=OBJECT_STORAGE_DATA)
    @patch("charm.MlflowCharm._get_relational_db_data", return_value=RELATIONAL_DB_DATA)
    def test_on_event(
        self,
        _: MagicMock,
        __: MagicMock,
        harness: Harness,
    ):
        harness = enable_exporter_container(harness)
        harness.set_leader(True)
        harness.begin()
        harness.charm._on_event(None)
        assert harness.charm.model.unit.status == ActiveStatus()

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_on_database_relation_removed(
        self,
        harness: Harness,
    ):
        harness.begin()
        harness.charm._on_database_relation_removed(None)
        assert harness.charm.model.unit.status == BlockedStatus(
            "Please add relation to the database"
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_on_get_minio_credentials_failure(self, harness: Harness):
        event = MagicMock()
        harness.begin()
        harness.charm._on_get_minio_credentials(event)
        event.fail.assert_called_with(
            "Minio is not reachable yet. Please try again in a few minutes."
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_on_get_minio_credentials_success(self, harness: Harness):
        harness = add_object_storage_to_harness(harness)
        event = MagicMock()
        harness.begin()
        harness.charm._on_get_minio_credentials(event)
        event.set_results.assert_called_with(
            {
                "access-key": OBJECT_STORAGE_DATA["access-key"],
                "secret-access-key": OBJECT_STORAGE_DATA["secret-key"],
            }
        )

    @patch(
        "charm.KubernetesServicePatch",
        lambda x, y, service_name, service_type, refresh_event: None,
    )
    def test_send_ingress_info_success(self, harness: Harness):
        harness.begin()
        ingress = MagicMock()
        interfaces = {"ingress": ingress}
        harness.charm._send_ingress_info(interfaces)
        ingress.send_data.assert_called_with(INGRESS_DATA)
