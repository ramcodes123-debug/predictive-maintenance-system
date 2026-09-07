"""
evaluate.py
Evaluate the saved model on a held-out test set and produce a
confusion matrix figure for the technical report.
"""

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from SRC.train import prepare_data, train_and_compare

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "MODELS"
SCREENSHOTS_DIR = PROJECT_ROOT / "SCREENSHOTS"
SCREENSHOTS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    print("\n[evaluate] Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n[evaluate] Classification report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()
    print(f"[evaluate] Saved confusion matrix to {SCREENSHOTS_DIR}/confusion_matrix.png")

    return metrics


def run_evaluation():
    (X_train, X_test, y_train, y_test), features = prepare_data()
    best_model, scaler, best_name = train_and_compare(X_train, X_test, y_train, y_test)
    return evaluate_model(best_model, scaler, X_test, y_test)


if __name__ == "__main__":
    run_evaluation()
