"""
preprocessing.py
Loading, cleaning, missing-value handling, and outlier handling
for the AI4I 2020 Predictive Maintenance dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "DATA" / "RAW" / "ai4i2020.csv"

# Numeric sensor columns after renaming
NUMERIC_COLS = ["air_temp", "process_temp", "rot_speed", "torque", "tool_wear"]

COLUMN_RENAME_MAP = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rot_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[preprocessing] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for easier downstream handling."""
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def handle_missing_values(df: pd.DataFrame, cols: list = NUMERIC_COLS) -> pd.DataFrame:
    """AI4I 2020 has no missing values by design, but this guards against
    it for robustness (and is worth a line in the technical report)."""
    for col in cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            print(f"[preprocessing] Filling {n_missing} missing values in '{col}' with median")
            df[col] = df[col].fillna(df[col].median())
    return df


def handle_outliers(df: pd.DataFrame, cols: list = NUMERIC_COLS) -> pd.DataFrame:
    """Cap outliers at 1.5*IQR bounds instead of dropping rows.
    Failure events are rare, and extreme sensor readings are often the
    signal we care about, so capping preserves that signal instead of
    discarding it."""
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        print(f"[preprocessing] '{col}': capped {n_outliers} outliers to [{lower:.2f}, {upper:.2f}]")
        df[col] = df[col].clip(lower, upper)
    return df


def run_preprocessing(path: Path = RAW_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    df = clean_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    return df


if __name__ == "__main__":
    df = run_preprocessing()
    print(df.head())
