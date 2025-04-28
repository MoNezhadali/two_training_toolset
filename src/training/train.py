"""
Train Iris classifier.

Don forget to fill this place with the description of the script.
"""

import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer


TRAIN_CONFIG_FILE_PATH = "IRIS_TRAIN_CONFIG"


def build_pipeline(cfg) -> Pipeline:
    sepal_cols = ["sepal length (cm)", "sepal width (cm)"]
    petal_cols = ["petal length (cm)", "petal width (cm)"]

    sepal_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "discretize",
                KBinsDiscretizer(
                    n_bins=cfg.get("sepal_bins", 3),
                    encode="onehot-dense",
                    strategy="quantile",
                ),
            ),
        ]
    )

    petal_scaler = MinMaxScaler(
        feature_range=tuple(cfg.get("petal_scaler_range", [0.0, 1.0]))
    )

    preprocessor = ColumnTransformer(
        [
            ("sepal_branch", sepal_pipe, sepal_cols),
            ("petal_branch", petal_scaler, petal_cols),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=cfg.get("logreg_max_iter", 300),
                    multi_class="multinomial",
                ),
            ),
        ]
    )


def train(cfg: dict) -> Path:
    iris = load_iris(as_frame=True)
    X: pd.DataFrame = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    out_dir = Path(cfg.get("output_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model-v{cfg['model_version']}.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_version": cfg["model_version"],
        "accuracy": score,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "sklearn_version": sklearn.__version__,
        "config": cfg,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"✔ Trained model saved to {model_path} (accuracy={score:.3f})")
    return model_path


def main() -> None:
    cfg_path = os.getenv(TRAIN_CONFIG_FILE_PATH)
    if not cfg_path:
        sys.exit(
            f"[ERROR] Set the YAML config path in the {TRAIN_CONFIG_FILE_PATH}"
            " environment variable."
        )

    cfg_file = Path(cfg_path)
    if not cfg_file.is_file():
        sys.exit(f"[ERROR] No YAML file found at {cfg_file}")

    cfg = yaml.safe_load(cfg_file.read_text())
    # Minimal sanity-check
    required_keys = {"model_version"}
    missing = required_keys - set(cfg)
    if missing:
        sys.exit(
            "[ERROR] Missing required config " f"keys: {', '.join(sorted(missing))}"
        )

    train(cfg)


if __name__ == "__main__":
    main()
