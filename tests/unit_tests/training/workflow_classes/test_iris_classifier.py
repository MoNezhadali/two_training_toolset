import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.training.workflow_classes.iris_classifier import IrisClassifier


@pytest.fixture
def config():
    return {
        "sepal_bins": 3,
        "petal_scaler_range": [0, 1],
        "logreg_max_iter": 100,
        "output_dir": tempfile.mkdtemp(),
        "model_version": "1.0",
    }


@pytest.fixture
def iris_classifier(config):
    return IrisClassifier(config)


class TestIrisClassifier:

    def test_load_data(self, iris_classifier):
        iris_classifier.load_data()
        assert iris_classifier.X is not None
        assert iris_classifier.y is not None
        assert not iris_classifier.X.empty
        assert len(iris_classifier.X) == len(iris_classifier.y)

    def test_split_data(self, iris_classifier):
        iris_classifier.load_data()
        iris_classifier.split_data()
        assert iris_classifier.X_train is not None
        assert iris_classifier.X_test is not None
        assert iris_classifier.y_train is not None
        assert iris_classifier.y_test is not None
        assert len(iris_classifier.X_train) > 0
        assert len(iris_classifier.X_test) > 0

    def test_build_pipeline(self, iris_classifier):
        iris_classifier.load_data()
        iris_classifier.split_data()
        iris_classifier.build_pipeline()
        assert iris_classifier.pipeline is not None
        steps = dict(iris_classifier.pipeline.named_steps)
        assert "preprocessor" in steps
        assert "model" in steps

    def test_train_model(self, iris_classifier):
        iris_classifier.load_data()
        iris_classifier.split_data()
        iris_classifier.build_pipeline()
        iris_classifier.train_model()
        assert hasattr(iris_classifier, "accuracy")
        assert 0 <= iris_classifier.accuracy <= 1

    def test_save_model(self, iris_classifier):
        iris_classifier.load_data()
        iris_classifier.split_data()
        iris_classifier.build_pipeline()
        iris_classifier.train_model()
        iris_classifier.save_model()

        model_path = (
            Path(iris_classifier.config["output_dir"])
            / f"model-v{iris_classifier.config['model_version']}.joblib"
        )
        metadata_path = Path(iris_classifier.config["output_dir"]) / "metadata.json"

        assert model_path.exists()
        assert metadata_path.exists()

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["model_version"] == iris_classifier.config["model_version"]
        assert "accuracy" in metadata
        assert "trained_at" in metadata
        assert "sklearn_version" in metadata
        assert "config" in metadata

    def test_end_to_end_workflow(self, iris_classifier):
        iris_classifier.load_data()
        iris_classifier.split_data()
        iris_classifier.build_pipeline()
        iris_classifier.train_model()
        iris_classifier.save_model()

        model_path = (
            Path(iris_classifier.config["output_dir"])
            / f"model-v{iris_classifier.config['model_version']}.joblib"
        )
        assert model_path.exists()
        assert iris_classifier.accuracy > 0.5  # Expect some decent accuracy (>50%)

    @pytest.fixture(autouse=True)
    def cleanup(self, request, config):
        """Cleanup output_dir after each test class finishes."""

        def remove_dir():
            shutil.rmtree(config["output_dir"], ignore_errors=True)

        request.addfinalizer(remove_dir)
