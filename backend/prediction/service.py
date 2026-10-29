import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "traffic_model.joblib"
METADATA_PATH = REPO_ROOT / "models" / "traffic_model_metadata.json"

EXPECTED_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
]


@lru_cache(maxsize=1)
def load_prediction_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("feature_columns") != EXPECTED_FEATURES:
        raise ValueError("Model metadata feature order does not match the API.")

    model = joblib.load(MODEL_PATH)

    if getattr(model, "n_features_in_", None) != len(EXPECTED_FEATURES):
        raise ValueError("The model does not accept the expected 12 features.")

    return model, metadata


def predict_traffic(features: dict) -> float:
    model, _ = load_prediction_artifacts()

    values = np.array(
        [[features[name] for name in EXPECTED_FEATURES]],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError("Prediction features must be finite numbers.")

    prediction = float(model.predict(values)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return max(0.0, prediction)