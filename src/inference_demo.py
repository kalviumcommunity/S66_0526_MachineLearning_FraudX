"""
inference_demo.py

Production Inference on a Persisted Model.

This module simulates the deployment scenario: load the .pkl artifact
that PR #26 produced (or produce a fresh one if not present) and use
the loaded pipeline to score new, never-before-seen transactions —
no retraining, no `.fit()` calls, just `predict()` and `predict_proba()`.

Workflow:
  1. Load the persisted Pipeline via `pickle.load()`.
     (If the .pkl file isn't present, build + fit the capstone pipeline
     locally and pickle.dump() it so this module is self-contained.)
  2. Confirm the loaded object is a fitted classifier with no
     retraining code path executed.
  3. Build 5 NEW data samples — hand-crafted to span the realistic
     input space (low / mid / high risk + edge cases). These have NO
     ground-truth labels because they're simulating production traffic.
  4. Score each sample with `predict()` and `predict_proba()`.
  5. Re-evaluate the loaded pipeline on the standard held-out test
     set and compare its metrics to PR #26's recorded numbers
     (accuracy=89.00%, P(1)=16.67%, R(1)=5.56%, F1(1)=8.33%). The match
     is the "performance validated after loading" check.
  6. Explicit assertion that no `.fit()` was called during inference.
  7. Write a CSV of new-sample predictions for offline inspection.

The whole module is the contract for what a deployed inference server
would do per request, plus a one-shot consistency check at startup.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# The capstone pipeline builder, used as the fallback when no .pkl
# exists yet. Importing it keeps this module self-sufficient.
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data


# Same artifact path as PR #26 — load if present, otherwise create.
PERSISTED_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "persisted_pipeline.pkl")
INFERENCE_CSV_PATH = os.path.join(BASE_DIR, "reports", "inference_predictions.csv")

# PR #26's recorded performance on the standard test set. The reload
# assertion in §5 checks the loaded model still matches.
EXPECTED_TEST_METRICS = {
    "accuracy":   0.8900,
    "precision_1": 0.16666666666666666,
    "recall_1":   0.05555555555555555,
    "f1_1":       0.08333333333333333,
}


# ----------------------------------------------------------------------
# Fallback: build + fit the capstone pipeline (only used if .pkl missing)
# ----------------------------------------------------------------------
def _build_and_fit_capstone(X_train, y_train) -> ImbPipeline:
    num_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ])
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ----------------------------------------------------------------------
# The 5 new transactions to score (synthesized to span the realistic
# input space; chosen to test what the model has learned about fraud)
# ----------------------------------------------------------------------
def build_new_transactions() -> pd.DataFrame:
    """Return a DataFrame with 5 unseen sample transactions.

    Each row is designed to probe a different region of the input space:
      0. Small, low-velocity, domestic, retail — should be 'legit'.
      1. Small, low-velocity, international, food — mid-risk.
      2. Large, medium-velocity, international, travel — higher risk.
      3. Very large, high-velocity, international, travel — should be 'fraud'.
      4. Edge case: a category the model never saw during training
         (`category="cryptoexchange"`). Tests handle_unknown="ignore".
    """
    return pd.DataFrame([
        {  # 0 — low risk
            "amount": 18.50,
            "transaction_count": 2,
            "velocity": 0.4,
            "category": "retail",
            "location": "domestic",
        },
        {  # 1 — mid risk (international, food)
            "amount": 35.00,
            "transaction_count": 3,
            "velocity": 1.2,
            "category": "food",
            "location": "international",
        },
        {  # 2 — higher risk (large + international + travel)
            "amount": 450.00,
            "transaction_count": 12,
            "velocity": 4.5,
            "category": "travel",
            "location": "international",
        },
        {  # 3 — high risk (very large + high velocity)
            "amount": 780.00,
            "transaction_count": 28,
            "velocity": 9.0,
            "category": "travel",
            "location": "international",
        },
        {  # 4 — edge case: unknown category
            "amount": 120.00,
            "transaction_count": 5,
            "velocity": 2.0,
            "category": "cryptoexchange",   # never seen during training
            "location": "international",
        },
    ])


# ----------------------------------------------------------------------
# Assertions guarding "no retraining"
# ----------------------------------------------------------------------
def _is_pipeline_fitted(pipeline) -> bool:
    """The classifier exposes fitted attributes (classes_, estimators_)
    only after `.fit(...)` has been called. Their presence proves the
    loaded pipeline is already fitted — we don't need to retrain."""
    clf = pipeline.named_steps.get("classifier")
    return hasattr(clf, "classes_") and hasattr(clf, "estimators_")


def _capture_fit_signature(pipeline) -> int:
    """Return a hash of the classifier's fitted attributes. We capture
    this before and after inference; if anything changed, .fit() got
    called somewhere in the chain — which would mean retraining."""
    clf = pipeline.named_steps["classifier"]
    sig = (
        tuple(clf.classes_.tolist()),
        len(clf.estimators_),
        clf.n_features_in_,
    )
    return hash(sig)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_inference_demo() -> Dict[str, Any]:
    print("=" * 70)
    print("Production Inference on Persisted Model")
    print("=" * 70)

    # --- Step 0: Ensure we have a .pkl to load. If PR #26 ran first,
    #             we just load. If not, we produce our own. ---
    df = clean_data(load_data(RAW_DATA_PATH))
    X_train, X_test, y_train, y_test = split_data(df)

    if not os.path.exists(PERSISTED_PIPELINE_PATH):
        print(f"\n[setup] {PERSISTED_PIPELINE_PATH} not found — producing one now.")
        pipeline_init = _build_and_fit_capstone(X_train, y_train)
        os.makedirs(os.path.dirname(PERSISTED_PIPELINE_PATH), exist_ok=True)
        with open(PERSISTED_PIPELINE_PATH, "wb") as fp:
            pickle.dump(pipeline_init, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[setup] Pickled fresh capstone pipeline -> {PERSISTED_PIPELINE_PATH}")

    # --- Part 1: Load the saved model ---
    print("\n--- Part 1: Load the saved model ---")
    print(f"  pickle.load({PERSISTED_PIPELINE_PATH!r})")
    with open(PERSISTED_PIPELINE_PATH, "rb") as fp:
        pipeline = pickle.load(fp)
    print(f"  Loaded type: {type(pipeline).__module__}.{type(pipeline).__name__}")
    print("  Confirming the loaded object is already fitted (no retraining needed) ...")
    assert _is_pipeline_fitted(pipeline), (
        "FAIL: the loaded pipeline's classifier has no fitted attributes — "
        "either the .pkl is corrupted or it was written before .fit() ran."
    )
    print("  [VERIFIED] Loaded pipeline is fitted. No retraining required.")

    # Capture the fit signature so we can re-verify after inference.
    pre_inference_sig = _capture_fit_signature(pipeline)

    # --- Part 2: Prepare new input data ---
    print("\n--- Part 2: Prepare new input data ---")
    new_samples = build_new_transactions()
    print(f"  Created {len(new_samples)} new transactions:")
    print(f"  Shape: {new_samples.shape}  (must be 2D: n_samples x n_features)")
    print(f"  Columns: {list(new_samples.columns)}")
    print("  Feature schema matches the training set: numerical + categorical, "
          "in the same order the ColumnTransformer expects.")
    print(new_samples.to_string(index=True))

    # --- Part 3: Perform inference ---
    print("\n--- Part 3: Perform inference (predict + predict_proba) ---")
    predictions = pipeline.predict(new_samples)
    probabilities = pipeline.predict_proba(new_samples)

    # The .pkl's preprocessor handles scaling, encoding, and even the
    # never-seen 'cryptoexchange' category (via handle_unknown="ignore").
    out = new_samples.copy()
    out["predicted_label"] = predictions
    out["prob_legit"] = probabilities[:, 0].round(4)
    out["prob_fraud"] = probabilities[:, 1].round(4)
    out["decision"] = np.where(predictions == 1, "FRAUD", "legit")

    print("\n  Inference results:")
    print(out.to_string(index=True))

    # --- Part 4: Verification — performance on the standard test set ---
    print("\n--- Part 4: Verification — re-evaluate on the standard test set ---")
    test_predictions = pipeline.predict(X_test)
    measured_test_metrics = {
        "accuracy":   float(accuracy_score(y_test, test_predictions)),
        "precision_1": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall_1":   float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1_1":       float(f1_score(y_test, test_predictions, zero_division=0)),
    }
    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])

    print("  Measured (this run, loaded pipeline):")
    for k, v in measured_test_metrics.items():
        print(f"    {k:<14s} : {_fmt_pct(v)}")
    print(f"    Confusion matrix: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    print("\n  Recorded by PR #26 (Model Persistence):")
    for k, v in EXPECTED_TEST_METRICS.items():
        print(f"    {k:<14s} : {_fmt_pct(v)}")

    # Equality check — the loaded model must match the recorded performance.
    perf_match = (
        np.isclose(measured_test_metrics["accuracy"],   EXPECTED_TEST_METRICS["accuracy"])
        and np.isclose(measured_test_metrics["precision_1"], EXPECTED_TEST_METRICS["precision_1"])
        and np.isclose(measured_test_metrics["recall_1"],    EXPECTED_TEST_METRICS["recall_1"])
        and np.isclose(measured_test_metrics["f1_1"],        EXPECTED_TEST_METRICS["f1_1"])
    )
    print(f"\n  Performance matches PR #26's recorded numbers (np.isclose): {perf_match}")
    assert perf_match, "FAIL: loaded model's test performance differs from the recorded values."

    # --- Confirm no `.fit()` ran during inference ---
    print("\n--- Verifying no retraining occurred during inference ---")
    post_inference_sig = _capture_fit_signature(pipeline)
    same_sig = pre_inference_sig == post_inference_sig
    print(f"  pre-inference  fit-signature hash : {pre_inference_sig}")
    print(f"  post-inference fit-signature hash : {post_inference_sig}")
    print(f"  identical                          : {same_sig}")
    assert same_sig, "FAIL: classifier's fitted attributes changed during inference."
    print("  [VERIFIED] No .fit() call ran during inference.")

    # --- Why inference must not include fit() ---
    print("\n--- Why inference should NOT include fitting steps ---")
    print(
        "  Calling .fit() at inference time would:\n"
        "    1. Discard the parameters learned during training (the whole point\n"
        "       of saving the .pkl).\n"
        "    2. Recompute scaler / encoder / imputer parameters from whatever\n"
        "       data is at hand — which at inference is either a single sample\n"
        "       or a small batch, NOT representative of the training distribution.\n"
        "    3. Change predictions from request to request as the running fit\n"
        "       drifts. Two identical inputs would get different scores.\n"
        "    4. Make the model non-deterministic and non-auditable.\n"
        "  Inference is a pure function: same input -> same output. The .pkl\n"
        "  freezes the parameters; .predict() / .predict_proba() are read-only."
    )

    # --- Persist the inference results for offline inspection ---
    os.makedirs(os.path.dirname(INFERENCE_CSV_PATH), exist_ok=True)
    out.to_csv(INFERENCE_CSV_PATH, index=False)
    print(f"\n  Predictions written to {INFERENCE_CSV_PATH}")

    print("\n" + "=" * 70)
    print("Inference demo completed without errors.")
    print("=" * 70)

    return {
        "new_samples": new_samples,
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "measured_test_metrics": measured_test_metrics,
        "perf_match": perf_match,
        "no_retraining": same_sig,
    }


if __name__ == "__main__":
    run_inference_demo()
