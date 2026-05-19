"""
load_and_verify.py

Fresh-environment load + predict + verify script for the Model
Persistence module.

This script is INTENTIONALLY designed to run as a separate Python
subprocess invoked from `src/model_persistence.py`. It imports
nothing from the orchestrator — only the project's data-loading
utilities, the test-set construction logic, and `pickle.load`. That
mirrors as closely as possible the "restart the kernel and load
fresh" workflow the assignment asks for.

Contract with the caller (model_persistence.py):
- Reads:  models/persisted_pipeline.pkl
- Writes: reports/load_and_verify.json   (predictions + metrics)
- Exits: 0 on success, non-zero on any failure.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import BASE_DIR, RAW_DATA_PATH
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data


PERSISTED_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "persisted_pipeline.pkl")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "reports", "load_and_verify.json")


def main() -> int:
    print("[fresh process] Starting load + verify ...")

    # 1. Sanity-check: the .pkl file exists.
    if not os.path.exists(PERSISTED_PIPELINE_PATH):
        print(f"[fresh process] ERROR: pickle file not found at {PERSISTED_PIPELINE_PATH}")
        return 2

    # 2. Load via pickle.load() — DO NOT retrain.
    print(f"[fresh process] pickle.load() <- {PERSISTED_PIPELINE_PATH}")
    with open(PERSISTED_PIPELINE_PATH, "rb") as fp:
        pipeline = pickle.load(fp)
    print(f"[fresh process] Loaded object type: {type(pipeline).__module__}.{type(pipeline).__name__}")

    # 3. Rebuild the SAME test split. random_state=42 + stratify=y in
    #    split_data() makes this deterministic across processes.
    df = clean_data(load_data(RAW_DATA_PATH))
    _, X_test, _, y_test = split_data(df)
    print(f"[fresh process] X_test shape = {X_test.shape}")

    # 4. Predict with the loaded pipeline — no retraining.
    predictions = pipeline.predict(X_test)

    # 5. Compute the same metrics the orchestrator computes.
    metrics = {
        "label": "Loaded (subprocess)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_1": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_1": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "predictions_hash": int(hash(predictions.tobytes())),
        "predictions": predictions.tolist(),
    }
    print(
        f"[fresh process] accuracy={metrics['accuracy']:.4f}  "
        f"P(1)={metrics['precision_1']:.4f}  "
        f"R(1)={metrics['recall_1']:.4f}  "
        f"F1(1)={metrics['f1_1']:.4f}"
    )
    cm = metrics["confusion_matrix"]
    print(f"[fresh process] confusion matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    # 6. Write the result for the orchestrator to read + diff.
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"[fresh process] Wrote metrics + predictions -> {OUTPUT_JSON_PATH}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
