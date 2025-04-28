"""
Train Iris classifier.

Don forget to fill this place with the description of the script.
"""

from metaflow import FlowSpec, step, Parameter
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import json
from datetime import datetime
import sklearn


class IrisClassifierFlow(FlowSpec):

    model_version = Parameter(
        "model_version", help="Version of the model", default="1.0"
    )
    output_dir = Parameter(
        "output_dir", help="Directory to save the model", default="artifacts"
    )
    sepal_bins = Parameter(
        "sepal_bins", help="Number of bins for sepal features", default=3
    )
    petal_scaler_range = Parameter(
        "petal_scaler_range",
        help="Scaler range for petal features",
        default="[0.0, 1.0]",
    )
    logreg_max_iter = Parameter(
        "logreg_max_iter", help="Max iterations for logistic regression", default=300
    )

    @step
    def start(self):
        """Start of the flow."""
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load the Iris dataset."""
        iris = load_iris(as_frame=True)
        self.X = iris.data
        self.y = iris.target
        self.next(self.split_data)

    @step
    def split_data(self):
        """Split into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.next(self.build_pipeline)

    @step
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
                        n_bins=self.sepal_bins,
                        encode="onehot-dense",
                        strategy="quantile",
                    ),
                ),
            ]
        )

        petal_scaler = MinMaxScaler(
            feature_range=tuple(
                map(float, self.petal_scaler_range.strip("[]").split(","))
            )
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
                        max_iter=self.logreg_max_iter,
                        multi_class="multinomial",
                    ),
                ),
            ]
        )
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the model."""
        self.pipeline.fit(self.X_train, self.y_train)
        self.accuracy = self.pipeline.score(self.X_test, self.y_test)
        self.next(self.save_model)

    @step
    def save_model(self):
        """Save the trained model and metadata."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / f"model-v{self.model_version}.joblib"
        joblib.dump(self.pipeline, model_path)

        metadata = {
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "sklearn_version": sklearn.__version__,
            "config": {
                "sepal_bins": self.sepal_bins,
                "petal_scaler_range": self.petal_scaler_range,
                "logreg_max_iter": self.logreg_max_iter,
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        print(f"✔ Model saved at {model_path} with accuracy={self.accuracy:.3f}")
        self.next(self.end)

    @step
    def end(self):
        """End of the flow."""
        print("✅ Flow completed successfully.")


if __name__ == "__main__":
    IrisClassifierFlow()
