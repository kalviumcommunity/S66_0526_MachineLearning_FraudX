"""
oversampling.py

Oversampling for Imbalanced Classification — Random Oversampling + SMOTE.

This module is the *resampling* counterpart to PR #22 (class weighting):
- PR #22 changed the LOSS (per-class weights inside the impurity criterion).
- THIS module changes the TRAINING DATA itself by adding minority-class
  rows. Two flavours:
    1. RandomOverSampler — duplicates randomly chosen minority rows until
       both classes have the same count.
    2. SMOTE — synthesises new minority rows by interpolating between a
       minority row and its k=5 nearest neighbours (also in the minority
       class).

The module trains three models on the SAME train/test split:
  1. Baseline RF (no resampling)
  2. RandomOverSampler -> RF
  3. SMOTE -> RF

All three live inside an `imblearn.pipeline.Pipeline` so cross_val_score
re-runs the resampler INSIDE every CV fold, on that fold's training rows
only. This is the only leakage-safe way to combine a sampler with CV:
sklearn's Pipeline can't host samplers because it has no notion of an
estimator that changes the row count, but imblearn's Pipeline does.

Discipline (Part 4 mandatory):
- Resamplers are fit on training data ONLY.
- Test set is sealed throughout. cross_val_score uses the same training
  set; each fold's validation rows are predicted (with .predict()), not
  resampled.
- Identical scoring metric (`f1` on the fraud / positive class) across
  baseline, Random OS, and SMOTE so the comparison is honest.
"""
from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
HEATMAP_PATH = os.path.join(PLOTS_DIR, "oversampling_confusion_matrices.png")
SMOTE_MODEL_PATH = os.path.join(BASE_DIR, "models", "smote_fraud_model.pkl")

CV_SPLITS = 5
SCORING = "f1"


# ----------------------------------------------------------------------
# Pipeline construction
# ----------------------------------------------------------------------
def _preprocessor() -> ColumnTransformer:
    """The standard preprocessor (same as every other module)."""
    num_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ])


def build_baseline_pipeline() -> ImbPipeline:
    """No resampling — just preprocessor + RF, wrapped in imblearn.Pipeline
    so it composes cleanly with cross_val_score in the same way as the
    other two."""
    return ImbPipeline(steps=[
        ("preprocessor", _preprocessor()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def build_random_os_pipeline() -> ImbPipeline:
    """RandomOverSampler step lives BETWEEN preprocessing and the
    classifier. Inside cross_val_score, the sampler runs per fold on
    that fold's training rows only — never on validation rows."""
    return ImbPipeline(steps=[
        ("preprocessor", _preprocessor()),
        ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def build_smote_pipeline() -> ImbPipeline:
    """SMOTE with k=5 neighbours (sklearn / imblearn default).

    SMOTE synthesises new minority rows: for each minority sample, it
    picks a random one of its k=5 nearest minority neighbours and creates
    a new sample on the line segment between them. The result is a
    larger, more diverse minority class — but the new rows are derived
    from existing minority rows only, so SMOTE cannot manufacture truly
    novel signal.
    """
    return ImbPipeline(steps=[
        ("preprocessor", _preprocessor()),
        ("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


# ----------------------------------------------------------------------
# Class distribution accounting (required output)
# ----------------------------------------------------------------------
def _print_class_distribution_before_after(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[int, int]]:
    """
    Print and return class-count snapshots:
      - before any resampling (the original training set)
      - after RandomOverSampler
      - after SMOTE

    The samplers need preprocessed numeric features (encoded categoricals
    in particular) to operate, so we apply a fitted ColumnTransformer to
    X_train first, then run each sampler on the result. This is purely
    for the BOOKKEEPING print-out — it does NOT contaminate the main
    pipelines, which fit their own preprocessing per fold.
    """
    preprocessor = _preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)

    counts_before = pd.Series(y_train).value_counts().sort_index().to_dict()

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    _, y_ros = ros.fit_resample(X_train_processed, y_train)
    counts_ros = pd.Series(y_ros).value_counts().sort_index().to_dict()

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    _, y_smote = smote.fit_resample(X_train_processed, y_train)
    counts_smote = pd.Series(y_smote).value_counts().sort_index().to_dict()

    print("\n--- Class distribution before and after oversampling (training set only) ---")
    print(f"{'Stage':<28s}{'class 0 (legit)':>18s}{'class 1 (fraud)':>18s}{'total':>10s}")
    print("-" * 74)
    for label, c in [
        ("Before (original train)", counts_before),
        ("After RandomOverSampler", counts_ros),
        ("After SMOTE",             counts_smote),
    ]:
        total = c.get(0, 0) + c.get(1, 0)
        print(f"{label:<28s}{c.get(0,0):>18d}{c.get(1,0):>18d}{total:>10d}")

    return {"before": counts_before, "random_os": counts_ros, "smote": counts_smote}


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def _evaluate(label: str, fitted, X_test, y_test) -> Dict[str, object]:
    y_pred = fitted.predict(X_test)
    return {
        "label": label,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_1": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_1": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }


def _cv_score(pipeline) -> Dict[str, float]:
    """5-fold stratified CV on X_train / y_train via imblearn Pipeline.

    The X_train and y_train arrays live in this function's enclosing
    scope through a closure — set in `run_oversampling_analysis` below.
    Each fold's sampler is re-fit on the fold's training subset only,
    never on validation rows. That's the entire point of using
    imblearn.Pipeline here.
    """
    raise NotImplementedError  # Replaced by closure below.


def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "  n/a  "
    return f"{x * 100:6.2f}%"


def _print_metric_table(rows, cv_results: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 100)
    print("Comparative performance table — baseline vs Random OS vs SMOTE  (test set, same X/y, same metric)")
    print("=" * 100)
    header = (
        f"{'Model':<22s}{'Accuracy':>11s}{'Prec (1)':>11s}{'Recall (1)':>12s}"
        f"{'F1 (1)':>10s}{'CV mean':>11s}{'CV std':>10s}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        cv = cv_results[r["label"]]
        print(
            f"{r['label']:<22s}"
            f"{_fmt_pct(r['accuracy']):>11s}"
            f"{_fmt_pct(r['precision_1']):>11s}"
            f"{_fmt_pct(r['recall_1']):>12s}"
            f"{_fmt_pct(r['f1_1']):>10s}"
            f"{_fmt_pct(cv['mean']):>11s}"
            f"{_fmt_pct(cv['std']):>10s}"
        )
    print("=" * 100)
    print("Legend: Prec/Recall/F1 = metrics for class 1 (fraud).")
    print("        CV columns: 5-fold StratifiedKFold mean and std of F1 (positive class).")
    print("        For oversampled pipelines, the sampler ran INSIDE each fold (no leakage).")


def _print_confusion_matrices(rows) -> None:
    print("\nConfusion matrices (rows = actual, cols = predicted [class 0, class 1]):")
    for r in rows:
        cm = r["confusion_matrix"]
        print(f"\n  {r['label']}:")
        print(f"    actual=0 (legit): predicted_0={cm[0][0]:>4d}  predicted_1={cm[0][1]:>4d}")
        print(f"    actual=1 (fraud): predicted_0={cm[1][0]:>4d}  predicted_1={cm[1][1]:>4d}")
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")


def _plot_confusion_heatmaps(rows, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping heatmap.")
        return

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, rows):
        cm = np.array(r["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(r["label"], fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["class 0\n(legit)", "class 1\n(fraud)"])
        ax.set_yticklabels(["class 0\n(legit)", "class 1\n(fraud)"])
        max_v = cm.max() if cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                cell = cm[i, j]
                text_color = "white" if cell > max_v * 0.55 else "black"
                ax.text(j, i, str(cell), ha="center", va="center", color=text_color,
                        fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion matrices on test set — Baseline vs Random Oversampling vs SMOTE",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  heatmap saved -> {path}")


# ----------------------------------------------------------------------
# Part 3/4 interpretation + business recommendation
# ----------------------------------------------------------------------
def _interpret_results(rows, cv_results, dist) -> None:
    base = rows[0]; ros = rows[1]; sm = rows[2]

    print("\n--- Interpretation ---")
    print(
        "  Class distribution after each resampler (training only):\n"
        f"    Baseline:           class 0 = {dist['before'][0]:>4d}, class 1 = {dist['before'][1]:>4d}\n"
        f"    RandomOverSampler:  class 0 = {dist['random_os'][0]:>4d}, class 1 = {dist['random_os'][1]:>4d}  (minority duplicated to match)\n"
        f"    SMOTE:              class 0 = {dist['smote'][0]:>4d}, class 1 = {dist['smote'][1]:>4d}  (minority synthesised via k-NN)"
    )

    print("\n  Recall (class 1) trajectory:")
    print(f"    Baseline:           {_fmt_pct(base['recall_1']).strip()}")
    print(f"    RandomOverSampler:  {_fmt_pct(ros['recall_1']).strip()}")
    print(f"    SMOTE:              {_fmt_pct(sm['recall_1']).strip()}")

    print("\n  Precision (class 1) trajectory:")
    print(f"    Baseline:           {_fmt_pct(base['precision_1']).strip()}")
    print(f"    RandomOverSampler:  {_fmt_pct(ros['precision_1']).strip()}")
    print(f"    SMOTE:              {_fmt_pct(sm['precision_1']).strip()}")

    print("\n  F1 (class 1) trajectory:")
    print(f"    Baseline:           {_fmt_pct(base['f1_1']).strip()}")
    print(f"    RandomOverSampler:  {_fmt_pct(ros['f1_1']).strip()}")
    print(f"    SMOTE:              {_fmt_pct(sm['f1_1']).strip()}")

    best_recall = max(rows, key=lambda r: r["recall_1"])
    best_f1 = max(rows, key=lambda r: r["f1_1"])

    print("\n--- Recall-precision trade-off ---")
    print(
        "  Resampling increases the model's exposure to minority examples, biasing it\n"
        "  toward predicting class 1 more eagerly. Two consequences:\n"
        "    - Recall ↑: more true positives caught (and more false positives).\n"
        "    - Precision ↓: every additional false positive lowers precision.\n"
        "  The right operating point depends on the business cost ratio between\n"
        "  false negatives and false positives. On THIS run the changes were:"
    )
    print(
        f"    Best recall on fraud   : {best_recall['label']} ({_fmt_pct(best_recall['recall_1']).strip()})\n"
        f"    Best F1 on fraud       : {best_f1['label']} ({_fmt_pct(best_f1['f1_1']).strip()})"
    )

    print("\n--- Final recommendation (business perspective) ---")
    if best_recall["recall_1"] > base["recall_1"] and best_f1["f1_1"] >= base["f1_1"]:
        print(
            f"  RECOMMENDATION: ship the {best_f1['label']} configuration. It improves\n"
            "  minority-class recall over the baseline without sacrificing F1, which is\n"
            "  the right joint metric on imbalanced data. Operate at the default 0.5\n"
            "  threshold to start; tune the threshold with predict_proba once a\n"
            "  business cost ratio (c_FN / c_FP) is set."
        )
    elif best_recall["recall_1"] > base["recall_1"]:
        print(
            f"  RECOMMENDATION: the {best_recall['label']} configuration improves recall\n"
            "  on the fraud class but at a precision cost. The decision depends on the\n"
            "  business cost ratio between false negatives (missed fraud) and false\n"
            "  positives (legit blocked). At a c_FN/c_FP of 50+ this trade is justified;\n"
            "  at low ratios it isn't. Next step: compute precision-recall curve from\n"
            "  predict_proba and pick the threshold against the cost ratio."
        )
    else:
        print(
            "  RECOMMENDATION: do NOT ship yet. Neither resampler lifted recall above\n"
            "  the baseline at the default 0.5 threshold on this dataset. The trees do\n"
            "  not find fraud-specific splits even after the minority class is\n"
            "  rebalanced. Next iteration: combine resampling with threshold tuning\n"
            "  (lower threshold below 0.5) OR with hyperparameter tuning (see PR #18).\n"
            "  The SMOTE pipeline is saved to models/smote_fraud_model.pkl for that\n"
            "  follow-up."
        )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_oversampling_analysis() -> Dict[str, object]:
    print("=" * 70)
    print("Oversampling for Imbalanced Classification (Random + SMOTE)")
    print("=" * 70)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Required output: distribution before/after.
    dist = _print_class_distribution_before_after(X_train, y_train)

    # Build three pipelines.
    pipelines = {
        "Baseline RF":         build_baseline_pipeline(),
        "RandomOverSampler+RF": build_random_os_pipeline(),
        "SMOTE+RF":            build_smote_pipeline(),
    }

    # Fit each pipeline on X_train / y_train. Each pipeline's resampler
    # (where present) only ever sees X_train / y_train — never the test
    # set. .predict() on the test set bypasses the sampler entirely.
    print("\n--- Training each pipeline on the SAME training data ---")
    fitted = {}
    for label, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        fitted[label] = pipe
        print(f"  {label} fitted on {len(y_train)} training rows.")

    # Cross-validation via imblearn.Pipeline (Part 4 mandatory).
    print("\n--- Part 4: 5-fold StratifiedKFold CV via imblearn.Pipeline ---")
    print("  Each fold's resampler is re-fit on that fold's training rows only.")
    print("  Validation rows are predicted (with .predict()), never resampled.")
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for label, pipe in pipelines.items():
        # Re-build the pipeline so each CV call works on an untouched estimator.
        scores = cross_val_score(
            build_baseline_pipeline() if label == "Baseline RF"
            else build_random_os_pipeline() if label == "RandomOverSampler+RF"
            else build_smote_pipeline(),
            X_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1,
        )
        cv_results[label] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {label:<22s}  CV mean F1 = {scores.mean():.4f}  std = {scores.std():.4f}")

    # Test-set evaluation (single sealed evaluation, identical metric).
    print("\n--- Test-set evaluation (one shot per model) ---")
    rows = []
    for label in ("Baseline RF", "RandomOverSampler+RF", "SMOTE+RF"):
        r = _evaluate(label, fitted[label], X_test, y_test)
        rows.append(r)
        cm = r["confusion_matrix"]
        print(
            f"  {label:<22s}  acc={_fmt_pct(r['accuracy']).strip():>6s}  "
            f"P1={_fmt_pct(r['precision_1']).strip():>6s}  "
            f"R1={_fmt_pct(r['recall_1']).strip():>6s}  "
            f"F1={_fmt_pct(r['f1_1']).strip():>6s}  "
            f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}"
        )

    _print_metric_table(rows, cv_results)
    _print_confusion_matrices(rows)
    _plot_confusion_heatmaps(rows, HEATMAP_PATH)
    _interpret_results(rows, cv_results, dist)

    # Persist the SMOTE pipeline for later threshold tuning.
    os.makedirs(os.path.dirname(SMOTE_MODEL_PATH), exist_ok=True)
    joblib.dump(fitted["SMOTE+RF"], SMOTE_MODEL_PATH)
    print(f"\n  SMOTE pipeline saved -> {SMOTE_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("Oversampling analysis completed without errors.")
    print("=" * 70)
    return {"rows": rows, "cv_results": cv_results, "dist": dist}


if __name__ == "__main__":
    run_oversampling_analysis()
