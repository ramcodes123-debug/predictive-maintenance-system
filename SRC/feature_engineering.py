"""
feature_engineering.py
Domain-informed feature creation and correlation-based feature selection
for the AI4I 2020 Predictive Maintenance dataset.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "MODELS"
MODELS_DIR.mkdir(exist_ok=True)

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Columns to always exclude from feature selection:
# - UDI is just a row identifier, not a real signal.
# - TWF/HDF/PWF/OSF/RNF are failure-SUBTYPE flags. In the real AI4I 2020 data
#   these are set only when a failure occurs, so they are effectively
#   post-outcome information (data leakage) with respect to `machine_failure` -
#   a model trained with them would look great in evaluation but be useless
#   for real-time prediction, since those flags aren't known in advance.
LEAKAGE_COLS = ["UDI", "TWF", "HDF", "PWF", "OSF", "RNF"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed features from raw sensor columns."""
    df = df.copy()
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["power"] = df["torque"] * df["rot_speed"] * (2 * np.pi / 60)  # rotational power (Watts)
    df["wear_torque_interaction"] = df["tool_wear"] * df["torque"]
    df["type_encoded"] = df["Type"].map(TYPE_MAP)
    return df


def select_features(df: pd.DataFrame, target: str = "machine_failure",
                     corr_threshold: float = 0.02) -> list:
    """Rank numeric features by absolute correlation with the target
    and drop near-uninformative ones."""
    numeric_df = df.select_dtypes(include=[np.number])
    drop_cols = [c for c in LEAKAGE_COLS if c in numeric_df.columns]
    numeric_df = numeric_df.drop(columns=drop_cols)
    if drop_cols:
        print(f"[feature_engineering] Excluded identifier/leakage columns: {drop_cols}")
    corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
    print("\n[feature_engineering] Feature correlation with target:")
    print(corr)
    selected = corr[corr > corr_threshold].index.tolist()
    print(f"\n[feature_engineering] Selected {len(selected)} features: {selected}")

    # Persist the selected feature list so train.py, predict.py, and the API
    # all agree on exact column order at inference time.
    joblib.dump(selected, MODELS_DIR / "feature_columns.pkl")
    return selected


def run_feature_engineering(df: pd.DataFrame, target: str = "machine_failure") -> tuple:
    df = engineer_features(df)
    selected_features = select_features(df, target=target)
    return df, selected_features


from SRC.preprocessing import run_preprocessing


if __name__ == "__main__":
    df = run_preprocessing()
    df, features = run_feature_engineering(df)

    print(df[features].head())