"""
Iris Classifier workflow implementation.
This module implements a workflow for training a classifier on the Iris dataset.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler

from src.training.workflow_classes.registry import register_workflow
from src.training.workflow_classes.workflow_tempalate import WorkflowTemplate


@register_workflow
class IrisClassifier(WorkflowTemplate):
    """Iris Classifier workflow implementation."""

    def __init__(self, config):
        """Initialize the IrisClassifier workflow."""
        self.config = config

    def load_data(self):
        """Load the Iris dataset."""
        iris = load_iris(as_frame=True)
        self.X = iris.data
        self.y = iris.target

    def split_data(self):
        """Split the dataset into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def build_pipeline(self):
        """Build the preprocessing and modeling pipeline."""
        sepal_cols = ["sepal length (cm)", "sepal width (cm)"]
        petal_cols = ["petal length (cm)", "petal width (cm)"]

        sepal_pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "discretize",
                    KBinsDiscretizer(
                        n_bins=self.config["sepal_bins"],
                        encode="onehot-dense",
                        strategy="quantile",
                    ),
                ),
            ]
        )

        petal_scaler = MinMaxScaler(
            feature_range=tuple(map(float, self.config["petal_scaler_range"]))
        )

        preprocessor = ColumnTransformer(
            [
                ("sepal_branch", sepal_pipe, sepal_cols),
                ("petal_branch", petal_scaler, petal_cols),
            ]
        )

        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=self.config["logreg_max_iter"],
                        multi_class="multinomial",
                    ),
                ),
            ]
        )

    def train_model(self):
        """Train the model."""
        self.pipeline.fit(self.X_train, self.y_train)
        self.accuracy = self.pipeline.score(self.X_test, self.y_test)

    def save_model(self):
        """Save the trained model and metadata."""
        out_dir = Path(self.config["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / f"model-v{self.config['model_version']}.joblib"
        joblib.dump(self.pipeline, model_path)

        metadata = {
            "model_version": self.config["model_version"],
            "accuracy": self.accuracy,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "sklearn_version": sklearn.__version__,
            "config": self.config,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        print(f"Model saved at {model_path} with accuracy={self.accuracy:.3f}")
