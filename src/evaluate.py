"""
evaluate.py

Responsible for:
- Computing performance metrics (accuracy, precision, recall, f1)
- Generating evaluation reports
- Ensuring unbiased assessment on test data

Two public entry points:
- `evaluate_model(...)`: minority-class scoring (the original, unchanged behavior).
- `evaluate_detailed(...)`: per-class scoring + balanced accuracy + confusion
  matrix, used by the Baseline + Class-Imbalance Comparison module to satisfy
  the "Report per-class metrics if working on imbalanced classification"
  guideline.
"""
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute evaluation metrics on test data (minority-class oriented).

    NOTE: This signature is preserved exactly so existing callers
    (`train.py`, etc.) keep working.

    Args:
        model: Trained model artifact.
        X_test: Processed test features.
        y_test: Test labels.

    Returns:
        dict: Dictionary of performance metrics for the positive class.
    """
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }


def evaluate_detailed(model, X_test, y_test, label: str = "model") -> dict:
    """
    Compute per-class evaluation metrics required for imbalanced binary
    classification.

    Returns a structured dict with:
      - label (caller-supplied name, e.g. "baseline_most_frequent")
      - overall accuracy and balanced accuracy
      - per-class precision / recall / f1 / support for class 0 and class 1
      - confusion matrix (raw counts)

    Args:
        model: A fitted classifier exposing .predict().
        X_test: Processed test features (same shape used to train the model).
        y_test: Ground-truth labels for the test set.
        label:  Free-form name used when printing comparison tables.

    Returns:
        dict suitable for tabular display and persistence.
    """
    predictions = model.predict(X_test)

    # Per-class scores. zero_division=0 so a "predicts only class 0" baseline
    # produces precision=0 / recall=0 instead of an UndefinedMetricWarning.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    return {
        "label": label,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "confusion_matrix": cm.tolist(),  # rows = actual, cols = predicted
    }
