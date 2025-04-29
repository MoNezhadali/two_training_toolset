# test_integration.py
import time
import requests
import pytest
import docker

BASE_URL = "http://localhost:5000"


class TestIrisApiIntegration:
    @pytest.fixture(scope="class", autouse=True)
    def start_ml_api_service(self):
        client = docker.from_env()

        # Start service
        container = client.containers.run(
            image="ml-api-service",  # Adjust if needed
            detach=True,
            ports={"5000/tcp": 5000},
            name="ml_api_test_container",
            remove=True,  # Auto-remove after stop
        )

        # Wait until the service is up
        timeout = 30
        for _ in range(timeout):
            try:
                r = requests.get(f"{BASE_URL}/health")
                if r.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            container.stop()
            pytest.fail("API service failed to become healthy in time")

        self.container = container
        yield  # Tests run here
        container.stop()

    def test_health_check(self):
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "sepal_length,sepal_width,petal_length,petal_width",
        [
            (5.1, 3.5, 1.4, 0.2),
            (6.0, 2.2, 5.0, 1.5),
            (6.9, 3.1, 5.4, 2.1),
        ],
    )
    def test_prediction(self, sepal_length, sepal_width, petal_length, petal_width):
        payload = {
            "request_id": "integration-test-id",
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }

        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "model_version" in data
        assert data["request_id"] == "integration-test-id"

    @pytest.mark.parametrize(
        "missing_field",
        [
            "sepal_length",
            "petal_width",
        ],
    )
    def test_prediction_missing_field(self, missing_field):
        payload = {
            "request_id": "integration-test-id",
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
        payload.pop(missing_field)

        r = requests.post(f"{BASE_URL}/predict", json=payload)
        assert r.status_code == 422, f"Expected failure status, got {r.status_code}"
