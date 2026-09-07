"""
train.py
Train and compare Random Forest and Logistic Regression classifiers,
save the best-performing model along with its scaler and feature list.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from SRC.preprocessing import run_preprocessing
from SRC.feature_engineering import run_feature_engineering

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "MODELS"
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_PATH = PROJECT_ROOT / "DATA" / "PROCESSED" / "processed_data.csv"


def prepare_data():
    df = run_preprocessing()
    df, features = run_feature_engineering(df)

    target = "machine_failure"
    df[features + [target]].to_csv(PROCESSED_PATH, index=False)
    print(f"[train] Saved processed dataset to {PROCESSED_PATH}")

    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), features


def train_and_compare(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        score = f1_score(y_test, preds)
        results[name] = (model, score)
        print(f"[train] {name}: F1 = {score:.4f}")

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_score = results[best_name]
    print(f"\n[train] Best model: {best_name} (F1 = {best_score:.4f})")

    return best_model, scaler, best_name


def run_training():
    (X_train, X_test, y_train, y_test), features = prepare_data()
    best_model, scaler, best_name = train_and_compare(X_train, X_test, y_train, y_test)

    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    # feature_columns.pkl already saved by feature_engineering.select_features
    print(f"[train] Saved model.pkl ({best_name}) and scaler.pkl to {MODELS_DIR}/")

    return best_model, scaler, features, (X_test, y_test)


if __name__ == "__main__":
    run_training()
