import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.serve.api_utils.authentication import dummy_authenticator
from src.serve.app import app, MODEL_VERSION, SPECIES

client = TestClient(app)


class TestIrisApp:

    @pytest.mark.parametrize("health_endpoint", ["/health"])
    def test_health_check(self, health_endpoint):
        response = client.get(health_endpoint)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "sepal_length,sepal_width,petal_length,petal_width,pred_idx",
        [
            (5.1, 3.5, 1.4, 0.2, 0),
            (6.0, 2.2, 5.0, 1.5, 1),
            (6.9, 3.1, 5.4, 2.1, 2),
        ],
    )
    def test_successful_prediction(
        self, sepal_length, sepal_width, petal_length, petal_width, pred_idx
    ):
        test_data = {
            "request_id": "test-uuid",
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        }

        with patch("src.serve.app.dummy_authenticator", return_value=True), patch(
            "src.serve.app.model.predict", return_value=[pred_idx]
        ):

            response = client.post("/predict", json=test_data)

        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == pred_idx
        assert body["prediction_label"] == SPECIES[pred_idx]
        assert body["model_version"] == MODEL_VERSION
        assert body["request_id"] == "test-uuid"

    def test_authentication_failure(self):
        test_data = {
            "request_id": "unauth-test-uuid",
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        # OVERRIDE dependency to simulate authentication failure
        app.dependency_overrides[dummy_authenticator] = lambda: False

        response = client.post("/predict", json=test_data)

        # Clean up overrides after test
        app.dependency_overrides = {}

        assert response.status_code == 401
        assert response.json()["detail"] == "Unauthorized"

    def test_prediction_failure(self):
        test_data = {
            "request_id": "error-test-uuid",
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }

        with patch("src.serve.app.dummy_authenticator", return_value=True), patch(
            "src.serve.app.model.predict", side_effect=Exception("Model crash")
        ):

            response = client.post("/predict", json=test_data)

        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]
