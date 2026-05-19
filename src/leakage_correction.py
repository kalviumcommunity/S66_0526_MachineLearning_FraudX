"""
leakage_correction.py

Data Leakage Detection and Pipeline Correction (Assignment module).

The module runs two end-to-end workflows on the EXACT same train/test split
and compares them with identical metrics. The first workflow is wrong on
purpose — it stacks FOUR independent leakage types so the audit story is
complete. The second is the correct, single-Pipeline replacement.

Layered leakage in the INCORRECT workflow
-----------------------------------------
1.  **Scaler leakage**: `StandardScaler().fit_transform(X_full[NUMERICAL_FEATURES])`
    learns mean / variance over the union of train and test rows. The
    train rows are therefore normalised using statistics that already
    incorporated the test set.

2.  **Imputer leakage**: `SimpleImputer(strategy="median").fit_transform(X_full)`
    computes the median over the full dataset. Even if there are no NaNs
    in the FraudX data (so the imputer is a no-op here), the *pattern*
    is structurally wrong — a project with sparse data would hand the
    model a median that already includes test rows.

3.  **Encoder leakage**: `OneHotEncoder().fit_transform(X_full[CATEGORICAL_FEATURES])`
    learns a category vocabulary that includes any category appearing
    *only* in the test set. The resulting one-hot columns make the
    training feature space depend on what's about to become the held-out
    set, which biases the model in subtle ways.

4.  **Feature-selection leakage** (the most pernicious): `SelectKBest(
    score_func=f_classif, k=K).fit(X_full_processed, y_full)` ranks
    features by ANOVA F-score *against y_full*. The chosen feature
    subset is therefore implicitly tuned to the labels of rows that
    will become the test set. This one survives most casual code
    reviews because the offending line *looks* defensive (it's
    "feature selection", a virtue) and because the leakage signal
    is invisible without a controlled comparison.

Then we call `cross_val_score(model, X_train_processed_selected, y_train,
cv=5)` on the pre-processed arrays. Each fold's "validation" rows have
already participated in the scaler / imputer / encoder / feature
selector fits, so the reported CV score is structurally inflated.

The CORRECT workflow
--------------------
A single `Pipeline(ColumnTransformer + SelectKBest + RandomForestClassifier)`
is fit once via `pipeline.fit(X_train, y_train)`. `cross_val_score` is
called on the *Pipeline*, so sklearn clones the whole estimator per
fold; each clone's preprocessing + feature selector + classifier all
fit on that fold's training subset only. The test set is `.transform()`-ed
exactly once at the very end, when we evaluate `pipeline.score(X_test,
y_test)`.

Identical metric (`f1` on the fraud / positive class) is used everywhere
so the comparison is honest.
"""
from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
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


CV_SPLITS = 5
SCORING = "f1"
# Top-K features after one-hot encoding. With 3 numericals + ~5 OHE columns
# we have roughly 8 features; selecting 6 forces SelectKBest to actually
# choose something, exposing the leakage path.
K_BEST_FEATURES = 6

CORRECT_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "leakage_correction_pipeline.pkl")


# ----------------------------------------------------------------------
# Approach A — INCORRECT workflow (4 layered leakage types)
# ----------------------------------------------------------------------
def _incorrect_workflow(df: pd.DataFrame, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    """
    Run the workflow with all four leakage types fit on the FULL dataset
    BEFORE the train/test split is honoured. Returns metrics dict.
    """
    print("\n--- [Incorrect Workflow] Preprocessing fit on the FULL dataset ---")

    X_full = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y_full = df["is_fraud"].copy()

    # Leakage 1: scaler fit on full dataset.
    scaler = StandardScaler()
    X_num_scaled_full = scaler.fit_transform(X_full[NUMERICAL_FEATURES])

    # Leakage 2: imputer fit on full dataset (structural, even if no-op here).
    imputer = SimpleImputer(strategy="median")
    X_num_imputed_full = imputer.fit_transform(X_num_scaled_full)

    # Leakage 3: encoder fit on full dataset.
    encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    X_cat_encoded_full = encoder.fit_transform(X_full[CATEGORICAL_FEATURES])

    X_processed_full = np.hstack([X_num_imputed_full, X_cat_encoded_full])

    # Leakage 4: feature selection fit on full processed data + full labels.
    selector = SelectKBest(score_func=f_classif, k=min(K_BEST_FEATURES, X_processed_full.shape[1]))
    X_selected_full = selector.fit_transform(X_processed_full, y_full)

    print(f"  Scaler   data_mean_     = {scaler.mean_}")
    print(f"  Encoder  categories     = {[list(c) for c in encoder.categories_]}")
    print(f"  Selector chosen feature indices = {np.flatnonzero(selector.get_support()).tolist()}")
    print(f"  Selector scores          = {np.round(selector.scores_, 4).tolist()}")
    print(f"  k selected = {selector.k}  of  total features = {X_processed_full.shape[1]}")

    # Map the train/test row indices into positional indices into X_processed_full.
    full_idx_to_pos = {idx: i for i, idx in enumerate(X_full.index)}
    train_pos = np.array([full_idx_to_pos[i] for i in X_train.index])
    test_pos = np.array([full_idx_to_pos[i] for i in X_test.index])

    X_train_processed = X_selected_full[train_pos]
    X_test_processed = X_selected_full[test_pos]

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)

    train_f1 = f1_score(y_train, model.predict(X_train_processed), zero_division=0)
    test_f1 = f1_score(y_test, model.predict(X_test_processed), zero_division=0)

    # Misleading CV: each "validation fold" was already seen by scaler /
    # imputer / encoder / selector during their fits above.
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        RandomForestClassifier(random_state=RANDOM_STATE),
        X_train_processed,
        y_train,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
    )

    print(f"  train F1   : {train_f1:.4f}")
    print(f"  test  F1   : {test_f1:.4f}")
    print(f"  CV mean F1 : {cv_scores.mean():.4f}  (std {cv_scores.std():.4f})  <-- LEAKAGE: structurally biased")

    return {
        "label": "Incorrect (4 leakage types)",
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }


# ----------------------------------------------------------------------
# Approach B — CORRECT workflow (single Pipeline)
# ----------------------------------------------------------------------
def build_correct_pipeline() -> Pipeline:
    """
    Construct the leakage-safe pipeline: ColumnTransformer →
    SelectKBest → RandomForestClassifier. Every step re-fits inside
    each CV fold on that fold's training rows only.
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
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def _correct_workflow(X_train, X_test, y_train, y_test) -> Dict[str, float]:
    """Proper Pipeline workflow. Same train/test split as the incorrect one."""
    print("\n--- [Correct Workflow] Single Pipeline; preprocessing + selector re-fit per fold ---")

    pipeline = build_correct_pipeline()
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # The honest CV: sklearn clones the pipeline per fold; each fold's
    # preprocessing + SelectKBest fit on that fold's training rows only.
    cv_scores = cross_val_score(pipeline, X_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1)

    # Final fit on full training data, single sealed test-set evaluation.
    pipeline.fit(X_train, y_train)
    train_f1 = f1_score(y_train, pipeline.predict(X_train), zero_division=0)
    test_f1 = f1_score(y_test, pipeline.predict(X_test), zero_division=0)

    # Surface which features were chosen by the in-pipeline selector on the
    # full training set. This is the auditable analogue of the incorrect
    # workflow's `selector.get_support()` printout above.
    fitted_selector = pipeline.named_steps["selector"]
    feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out()
    chosen_features = feature_names_out[fitted_selector.get_support()].tolist()
    print(f"  Pipeline-selector chosen features: {chosen_features}")
    print(f"  train F1   : {train_f1:.4f}")
    print(f"  test  F1   : {test_f1:.4f}")
    print(f"  CV mean F1 : {cv_scores.mean():.4f}  (std {cv_scores.std():.4f})  <-- honest")

    os.makedirs(os.path.dirname(CORRECT_PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline, CORRECT_PIPELINE_PATH)
    print(f"  fitted pipeline saved -> {CORRECT_PIPELINE_PATH}")

    return {
        "label": "Correct (Pipeline)",
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _print_comparison(a: Dict[str, float], b: Dict[str, float]) -> None:
    print("\n" + "=" * 90)
    print("Performance Comparison Table (identical train/test split, identical scoring metric)")
    print("=" * 90)
    header = f"{'Workflow':<32s}{'Train F1':>11s}{'Test F1':>11s}{'CV mean':>11s}{'CV std':>10s}"
    print(header)
    print("-" * 90)
    for r in (a, b):
        print(
            f"{r['label']:<32s}"
            f"{_fmt_pct(r['train_f1']):>11s}"
            f"{_fmt_pct(r['test_f1']):>11s}"
            f"{_fmt_pct(r['cv_mean']):>11s}"
            f"{_fmt_pct(r['cv_std']):>10s}"
        )
    print("=" * 90)


def _print_final_conclusion(a: Dict[str, float], b: Dict[str, float]) -> None:
    cv_gap = a["cv_mean"] - b["cv_mean"]
    test_gap = a["test_f1"] - b["test_f1"]

    print("\n--- Final Conclusion ---")
    print(f"  Incorrect-minus-Correct CV  : {_fmt_pct(cv_gap):>7s}")
    print(f"  Incorrect-minus-Correct Test: {_fmt_pct(test_gap):>7s}")

    if cv_gap > 0.05:
        print(
            "\n  The incorrect workflow's CV score is meaningfully higher than the\n"
            "  Pipeline workflow's. That gap is leakage, not learning. The model\n"
            "  reported by the incorrect workflow looks better than the model\n"
            "  reported by the correct workflow only because the incorrect workflow\n"
            "  cheated. In production, where the model only sees data it has not\n"
            "  trained or preprocessed on, performance will degrade toward the\n"
            "  Pipeline workflow's number — or worse. The Pipeline workflow's\n"
            "  apparently-lower CV is the trustworthy number."
        )
    elif cv_gap > 0.0:
        print(
            "\n  The incorrect workflow's CV score is slightly higher than the\n"
            "  Pipeline workflow's. On a richer dataset (rare categories, large\n"
            "  outliers, sparse imputation) the gap would widen. Pipeline is still\n"
            "  the strictly correct workflow."
        )
    elif cv_gap == 0.0 and b["cv_mean"] == 0.0:
        print(
            "\n  Both workflows report CV mean F1 = 0 because the underlying\n"
            "  RandomForestClassifier collapses to majority-class predictions on\n"
            "  this 91/9-imbalanced FraudX dataset (see PR #17). The leakage in\n"
            "  the incorrect workflow has nothing to act on — there is no\n"
            "  learning signal to inflate. The structural bug is still present\n"
            "  in the incorrect workflow's code, and on any less-degenerate\n"
            "  classification problem it would produce a visibly-inflated CV.\n"
            "  The Pipeline workflow is unconditionally preferred."
        )
    else:
        print(
            "\n  The incorrect workflow's CV did not exceed the Pipeline workflow's\n"
            "  in this run. The Pipeline workflow is still preferred — leakage is\n"
            "  a correctness issue, not a metric-magnitude issue."
        )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_leakage_correction() -> Dict[str, Dict[str, float]]:
    """End-to-end runner. Returns the two metric dicts for downstream use."""
    print("=" * 70)
    print("Data Leakage Detection and Pipeline Correction")
    print("=" * 70)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    incorrect = _incorrect_workflow(df, X_train, X_test, y_train, y_test)
    correct = _correct_workflow(X_train, X_test, y_train, y_test)

    _print_comparison(incorrect, correct)
    _print_final_conclusion(incorrect, correct)

    print("\n" + "=" * 70)
    print("Leakage Correction module completed without errors.")
    print("=" * 70)
    return {"incorrect": incorrect, "correct": correct}


if __name__ == "__main__":
    run_leakage_correction()
