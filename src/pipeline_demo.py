"""
pipeline_demo.py

Scikit-Learn Pipeline Integration demonstration.

This module satisfies the Pipeline Integration assignment by running TWO
classification workflows on the *exact same* train/test split and showing
the difference in behaviour:

A. **Manual workflow (without Pipeline)** — what an engineer might write
   on a first pass. We deliberately make the most common mistake: fitting
   the preprocessing on the *entire* dataset before splitting, then
   evaluating with cross-validation on the pre-transformed training array.
   This is data leakage. The reported CV score is optimistic.

B. **Proper Pipeline workflow** — wraps preprocessing and model in a single
   `Pipeline(ColumnTransformer + RandomForestClassifier)` object and runs
   `cross_val_score` on the pipeline itself, so the ColumnTransformer is
   re-fit inside each CV fold on that fold's training rows only. This is
   the leakage-safe pattern.

The side-by-side comparison makes the cost of skipping Pipeline visible
at runtime: Approach A's CV score is inflated because the preprocessing
already saw the validation rows; Approach B's CV score is honest.

Optional hyperparameter tuning inside the pipeline is deliberately
out of scope here — see PR #18 (`feature/hyperparameter-tuning`) for the
complete RandomizedSearchCV-with-Pipeline implementation.
"""
from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
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


# Module-level constants. Single scoring metric used everywhere, per the
# assignment's "consistent metric throughout" guideline.
CV_SPLITS = 5
SCORING = "f1"
SKLEARN_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "sklearn_pipeline.pkl")


# ----------------------------------------------------------------------
# Approach A: Manual workflow (the bad pattern). DO NOT use this in
# production. It exists in this module purely to demonstrate the failure
# mode the assignment is testing for understanding of.
# ----------------------------------------------------------------------
def _approach_a_manual_with_leakage(df: pd.DataFrame, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    """
    Manual preprocessing that fits on the FULL dataset before splitting —
    the classic leakage pattern. Reports a CV score that looks great
    because every fold's preprocessing has already seen the validation rows.

    Returns dict with: train_f1, test_f1, cv_mean, cv_std, notes.
    """
    print("\n--- [Approach A] Manual preprocessing WITH leakage ---")
    print("  (fitting scaler / encoder on the FULL dataset before splitting)")

    X_full = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y_full = df["is_fraud"].copy()

    # Bug 1: fit scaler on FULL dataset (X_train + X_test combined).
    scaler = StandardScaler()
    X_num_scaled_full = scaler.fit_transform(X_full[NUMERICAL_FEATURES])

    # Bug 2: fit encoder on FULL dataset.
    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    X_cat_encoded_full = encoder.fit_transform(X_full[CATEGORICAL_FEATURES])

    X_processed_full = np.hstack([X_num_scaled_full, X_cat_encoded_full])

    # Use the SAME train/test indices as the proper workflow so the comparison is honest.
    train_idx = X_train.index
    test_idx = X_test.index
    full_idx_to_pos = {idx: i for i, idx in enumerate(X_full.index)}
    train_pos = np.array([full_idx_to_pos[i] for i in train_idx])
    test_pos = np.array([full_idx_to_pos[i] for i in test_idx])

    X_train_processed = X_processed_full[train_pos]
    X_test_processed = X_processed_full[test_pos]

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)

    train_f1 = f1_score(y_train, model.predict(X_train_processed), zero_division=0)
    test_f1 = f1_score(y_test, model.predict(X_test_processed), zero_division=0)

    # The misleading CV: scoring on pre-transformed training arrays. Each
    # fold's "validation" rows were ALREADY seen by the scaler/encoder
    # during their fit on X_full. Leakage.
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        RandomForestClassifier(random_state=RANDOM_STATE),
        X_train_processed,
        y_train,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
    )

    print(f"  train F1: {train_f1:.4f}")
    print(f"  test  F1: {test_f1:.4f}")
    print(f"  CV mean F1: {cv_scores.mean():.4f}  (std {cv_scores.std():.4f})  <-- LEAKAGE: inflated")

    return {
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "label": "Manual (with leakage)",
    }


# ----------------------------------------------------------------------
# Approach B: Proper Pipeline workflow (the right pattern).
# ----------------------------------------------------------------------
def build_proper_pipeline() -> Pipeline:
    """
    Construct the canonical preprocessing + model Pipeline used by this
    module.

    Numerical pipeline:  SimpleImputer(median) -> StandardScaler
    Categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder
    Combined via ColumnTransformer, then wrapped with the classifier in
    a single Pipeline.

    Returning a single Pipeline object means downstream callers can call
    `.fit(X_train, y_train)`, `.predict(X_test)`, `cross_val_score(pipe,
    X, y, ...)`, etc. — and preprocessing is automatically re-fit on the
    appropriate slice in every case.
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    return pipeline


def _approach_b_pipeline(X_train, X_test, y_train, y_test) -> Dict[str, float]:
    """
    Proper workflow: a single Pipeline runs end-to-end, and
    cross_val_score re-fits the preprocessing inside every fold.
    """
    print("\n--- [Approach B] Proper Pipeline workflow ---")
    print("  (ColumnTransformer + RandomForestClassifier in a single Pipeline)")

    pipeline = build_proper_pipeline()

    # CV runs on the Pipeline. sklearn clones the estimator for every fold,
    # so the ColumnTransformer's fit_transform is called on that fold's
    # training rows only — no information from validation rows leaks into
    # the scaler / encoder.
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1)

    # Fit on the FULL training set (the same data CV used) for final
    # test-set scoring. The test set has NEVER been touched by any
    # transformer fit at this point.
    pipeline.fit(X_train, y_train)
    train_f1 = f1_score(y_train, pipeline.predict(X_train), zero_division=0)
    test_f1 = f1_score(y_test, pipeline.predict(X_test), zero_division=0)

    print(f"  train F1: {train_f1:.4f}")
    print(f"  test  F1: {test_f1:.4f}")
    print(f"  CV mean F1: {cv_scores.mean():.4f}  (std {cv_scores.std():.4f})  <-- honest")

    # Persist the fitted pipeline so it can be reloaded at inference.
    os.makedirs(os.path.dirname(SKLEARN_PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline, SKLEARN_PIPELINE_PATH)
    print(f"  fitted pipeline saved -> {SKLEARN_PIPELINE_PATH}")

    return {
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "label": "Pipeline (no leakage)",
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _print_comparison(approach_a: Dict[str, float], approach_b: Dict[str, float]) -> None:
    print("\n" + "=" * 80)
    print("Side-by-side: Manual workflow vs Proper Pipeline (same split, same metric)")
    print("=" * 80)
    header = f"{'Approach':<28s}{'Train F1':>11s}{'Test F1':>11s}{'CV mean':>11s}{'CV std':>10s}"
    print(header)
    print("-" * 80)
    for r in (approach_a, approach_b):
        print(
            f"{r['label']:<28s}"
            f"{_fmt_pct(r['train_f1']):>11s}"
            f"{_fmt_pct(r['test_f1']):>11s}"
            f"{_fmt_pct(r['cv_mean']):>11s}"
            f"{_fmt_pct(r['cv_std']):>10s}"
        )
    print("=" * 80)


def _interpret(approach_a: Dict[str, float], approach_b: Dict[str, float]) -> None:
    cv_gap = approach_a["cv_mean"] - approach_b["cv_mean"]
    test_gap = approach_a["test_f1"] - approach_b["test_f1"]

    print("\n--- Reading ---")
    print(f"  CV mean gap (A - B)  : {_fmt_pct(cv_gap):>7s}")
    print(f"  Test F1 gap (A - B)  : {_fmt_pct(test_gap):>7s}")

    if cv_gap > 0.05:
        print(
            "\n  VERDICT: Approach A's CV score is meaningfully higher than B's. That\n"
            "  difference is NOT a sign that A is the better workflow. It is leakage —\n"
            "  the preprocessing already saw the validation rows during fit, so the\n"
            "  CV score is reporting on a model that effectively cheated. Approach B's\n"
            "  CV score is the honest one."
        )
    elif cv_gap > 0.0:
        print(
            "\n  VERDICT: Approach A's CV score is slightly higher than B's. On this\n"
            "  specific dataset the leakage signal is small (likely because the\n"
            "  categorical encoder was the main offender and 'category' / 'location'\n"
            "  have stable, low-cardinality values across train and test). The gap\n"
            "  would widen on a dataset with rare categories or skewed numerical\n"
            "  features. Approach B's CV score remains the only trustworthy one."
        )
    elif cv_gap == 0.0 and approach_b["cv_mean"] == 0.0:
        print(
            "\n  VERDICT: Both approaches report CV mean F1 = 0 because the underlying\n"
            "  RandomForestClassifier is unable to learn the minority class on this\n"
            "  imbalanced dataset (see PR #17). The leakage that Approach A would\n"
            "  normally introduce has nothing to act on. Approach B is still the\n"
            "  right pattern — it just doesn't change anything on a degenerate CV\n"
            "  surface. The fix here is class_weight / resampling, not workflow."
        )
    else:
        print(
            "\n  VERDICT: Approach A's CV does not exceed B's in this run. The\n"
            "  Pipeline workflow is still strictly preferred — leakage is a\n"
            "  correctness issue, not a metric-magnitude issue."
        )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_pipeline_demo() -> Dict[str, Dict[str, float]]:
    """
    End-to-end runner. Loads + cleans + splits, runs both approaches on
    the *same* split, prints the comparison + verdict, persists the
    proper Pipeline. Returns the metric dicts.
    """
    print("=" * 70)
    print("Scikit-Learn Pipeline Integration Demonstration")
    print("=" * 70)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    approach_a = _approach_a_manual_with_leakage(df, X_train, X_test, y_train, y_test)
    approach_b = _approach_b_pipeline(X_train, X_test, y_train, y_test)

    _print_comparison(approach_a, approach_b)
    _interpret(approach_a, approach_b)

    print("\n" + "=" * 70)
    print("Pipeline demonstration completed without errors.")
    print("=" * 70)
    return {"manual_with_leakage": approach_a, "pipeline_no_leakage": approach_b}


if __name__ == "__main__":
    run_pipeline_demo()
