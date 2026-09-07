"""
predict.py
Load the saved model, scaler, and feature list, and run inference
on new raw sensor readings. Used directly by api/main.py.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "MODELS"

_model = None
_scaler = None
_feature_columns = None


def _load_artifacts():
    """Lazy-load so importing this module doesn't require the model
    files to exist yet (useful during early development)."""
    global _model, _scaler, _feature_columns
    if _model is None:
        _model = joblib.load(MODELS_DIR / "model.pkl")
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        _feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return _model, _scaler, _feature_columns


def build_features(raw_input: dict) -> pd.DataFrame:
    """Recreate the same engineered features used during training,
    from a single raw sensor reading dict.

    Expected raw_input keys:
        air_temp, process_temp, rot_speed, torque, tool_wear, type
        (type is one of 'L', 'M', 'H')
    """
    type_map = {"L": 0, "M": 1, "H": 2}

    air_temp = raw_input["air_temp"]
    process_temp = raw_input["process_temp"]
    rot_speed = raw_input["rot_speed"]
    torque = raw_input["torque"]
    tool_wear = raw_input["tool_wear"]
    machine_type = raw_input.get("type", "M")

    features = {
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rot_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "temp_diff": process_temp - air_temp,
        "power": torque * rot_speed * (2 * np.pi / 60),
        "wear_torque_interaction": tool_wear * torque,
        "type_encoded": type_map.get(machine_type, 1),
    }
    return pd.DataFrame([features])


def predict(raw_input: dict) -> dict:
    model, scaler, feature_columns = _load_artifacts()

    df = build_features(raw_input)
    # Align exactly to the columns/order the model was trained on
    df = df.reindex(columns=feature_columns, fill_value=0)

    X_scaled = scaler.transform(df)
    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0][1])

    return {
        "prediction": pred,
        "prediction_label": "Failure" if pred == 1 else "No Failure",
        "failure_probability": round(proba, 4),
    }


if __name__ == "__main__":
    # Quick manual test
    sample = {
        "air_temp": 300.0,
        "process_temp": 310.0,
        "rot_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
        "type": "M",
    }
    print(predict(sample))
