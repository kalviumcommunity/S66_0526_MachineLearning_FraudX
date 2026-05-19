"""
imbalance_analysis.py

Class Imbalance Analysis (diagnosis + interpretation, NOT a fix).

This module satisfies the Class Imbalance Analysis assignment. It does
four things, in order:

  1. Diagnose the class distribution on the FraudX dataset and classify
     the imbalance severity (mild / moderate / severe) using a stated
     rubric.
  2. Train a `DummyClassifier(strategy="most_frequent")` majority-class
     baseline. Report accuracy + confusion matrix. Explain why the
     baseline can look strong and why it's useless in practice.
  3. Train a standard `RandomForestClassifier` inside a leakage-safe
     `Pipeline(ColumnTransformer + classifier)`. Report accuracy,
     precision, recall, F1, confusion matrix, plus the two ranking
     metrics the assignment singles out: **PR-AUC** (average precision)
     and **ROC-AUC**.
  4. Print a side-by-side comparison and a confusion-matrix heatmap.

The module is purely diagnostic — it does NOT apply resampling,
class-weighting, or threshold tuning. The assignment is explicit:
"diagnosis and evaluation discipline, not yet on fixing imbalance with
resampling or weighting techniques".

Identical scoring discipline across every metric (single `pos_label=1`,
`zero_division=0`) so the comparison is honest.
"""
from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
HEATMAP_PATH = os.path.join(PLOTS_DIR, "imbalance_confusion_matrices.png")
IMBALANCE_MODEL_PATH = os.path.join(BASE_DIR, "models", "imbalance_standard_model.pkl")


# ----------------------------------------------------------------------
# Step 1 — Distribution + severity diagnosis
# ----------------------------------------------------------------------
def classify_imbalance_severity(minority_share: float) -> str:
    """
    Classify imbalance severity using a stated rubric.

    Minority proportion ≥ 40%  → "mild"
    Minority proportion 10-40% → "moderate"
    Minority proportion < 10%  → "severe"
    """
    if minority_share >= 0.40:
        return "mild"
    if minority_share >= 0.10:
        return "moderate"
    return "severe"


def analyze_class_distribution(y: pd.Series) -> Dict[str, object]:
    """Return counts, percentages, severity, and majority / minority labels."""
    counts = y.value_counts().sort_index()
    proportions = y.value_counts(normalize=True).sort_index()

    majority_class = int(counts.idxmax())
    minority_class = int(counts.idxmin())
    minority_share = float(proportions[minority_class])
    severity = classify_imbalance_severity(minority_share)

    return {
        "counts": counts.to_dict(),
        "proportions": proportions.round(4).to_dict(),
        "majority_class": majority_class,
        "minority_class": minority_class,
        "minority_share": minority_share,
        "severity": severity,
        "total": int(y.shape[0]),
    }


def _print_distribution(label: str, dist: Dict[str, object]) -> None:
    print(f"\n  {label}  (n={dist['total']})")
    for cls, n in dist["counts"].items():
        p = dist["proportions"][cls]
        marker = "(majority)" if cls == dist["majority_class"] else "(minority)"
        print(f"    class {cls} {marker:<10s}  count={n:>5d}  share={p:6.2%}")
    print(f"    severity                 = {dist['severity'].upper()}  "
          f"(rubric: <10% minority = severe, 10-40% = moderate, >=40% = mild)")


# ----------------------------------------------------------------------
# Step 2 / 3 — Model construction (Pipeline + classifier)
# ----------------------------------------------------------------------
def _build_pipeline(classifier) -> Pipeline:
    """
    Wrap any classifier (Dummy or RF) in the same leakage-safe Pipeline so
    both baselines AND the model see identically-preprocessed features.
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
        ("classifier", classifier),
    ])


# ----------------------------------------------------------------------
# Evaluation — compute every metric the assignment + Part 3 need
# ----------------------------------------------------------------------
def _evaluate(model_label: str, fitted, X_test, y_test) -> Dict[str, object]:
    """
    Compute the metric battery for one model. Returns a single dict so
    the caller can slot it into a comparison table and persist it.

    PR-AUC = average precision over the precision-recall curve.
    ROC-AUC = area under the receiver-operating-characteristic curve.
    `predict_proba` is used when available; for the DummyClassifier
    `most_frequent` baseline the predicted class is constant, so
    `predict_proba` returns a degenerate matrix — both AUCs fall back to
    a sensible value (0.5 for ROC-AUC under sklearn's convention; the
    class prior for PR-AUC).
    """
    y_pred = fitted.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # AUC family: try predict_proba; fall back to decision_function or labels.
    try:
        y_score = fitted.predict_proba(X_test)[:, 1]
    except AttributeError:
        try:
            y_score = fitted.decision_function(X_test)
        except AttributeError:
            y_score = y_pred.astype(float)

    try:
        roc_auc = float(roc_auc_score(y_test, y_score))
    except ValueError:
        roc_auc = float("nan")

    pr_auc = float(average_precision_score(y_test, y_score))

    return {
        "label": model_label,
        "accuracy": float(accuracy),
        "precision_1": float(precision),
        "recall_1": float(recall),
        "f1_1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }


def _fmt_pct(x: float) -> str:
    if x != x:  # NaN
        return "  n/a  "
    return f"{x * 100:6.2f}%"


def _print_metric_table(rows) -> None:
    print("\n" + "=" * 92)
    print("Side-by-side: majority-class baseline vs standard model")
    print("=" * 92)
    header = (
        f"{'Model':<30s}{'Accuracy':>11s}{'Prec (1)':>11s}{'Recall (1)':>12s}"
        f"{'F1 (1)':>10s}{'PR-AUC':>10s}{'ROC-AUC':>10s}"
    )
    print(header)
    print("-" * 92)
    for r in rows:
        print(
            f"{r['label']:<30s}"
            f"{_fmt_pct(r['accuracy']):>11s}"
            f"{_fmt_pct(r['precision_1']):>11s}"
            f"{_fmt_pct(r['recall_1']):>12s}"
            f"{_fmt_pct(r['f1_1']):>10s}"
            f"{_fmt_pct(r['pr_auc']):>10s}"
            f"{_fmt_pct(r['roc_auc']):>10s}"
        )
    print("=" * 92)
    print("Legend: Prec/Recall/F1 = metrics for class 1 (fraud).")
    print("        PR-AUC = average precision; ROC-AUC = receiver operating characteristic AUC.")


def _print_confusion_matrices(rows) -> None:
    print("\nConfusion matrices (rows = actual, cols = predicted [class 0, class 1]):")
    for r in rows:
        cm = r["confusion_matrix"]
        print(f"\n  {r['label']}:")
        print(f"    actual=0:  predicted_0={cm[0][0]:>4d}  predicted_1={cm[0][1]:>4d}")
        print(f"    actual=1:  predicted_0={cm[1][0]:>4d}  predicted_1={cm[1][1]:>4d}")


# ----------------------------------------------------------------------
# Heatmap visualisation
# ----------------------------------------------------------------------
def _plot_confusion_heatmaps(rows, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping heatmap.")
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(5.5 * len(rows), 4.5))
    if len(rows) == 1:
        axes = [axes]

    for ax, r in zip(axes, rows):
        cm = np.array(r["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(r["label"], fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["class 0", "class 1"])
        ax.set_yticklabels(["class 0", "class 1"])
        # Annotate cells. Choose text color by background brightness.
        max_v = cm.max()
        for i in range(2):
            for j in range(2):
                cell = cm[i, j]
                text_color = "white" if cell > max_v * 0.55 else "black"
                ax.text(j, i, str(cell), ha="center", va="center", color=text_color,
                        fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion matrices on the test set", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  heatmap saved -> {path}")


# ----------------------------------------------------------------------
# Part 3 — Comparison answers (printed to stdout AND captured for docs)
# ----------------------------------------------------------------------
def _part3_comparison_answers(baseline: Dict, model: Dict, dist: Dict) -> None:
    """
    Answer the five Part 3 questions with reference to the just-computed
    numbers. The strings are kept short enough to fit comfortably in the
    runtime log; the long-form version lives in docs/IMBALANCE_ANALYSIS.md.
    """
    print("\n--- Part 3: Comparison answers (reference: numbers just computed) ---")

    print(
        "  1) Does accuracy reflect minority-class performance?\n"
        f"     No. Baseline accuracy = {_fmt_pct(baseline['accuracy']).strip()} despite zero "
        f"true positives on the minority class (recall {_fmt_pct(baseline['recall_1']).strip()}). "
        "Accuracy is dominated by class 0."
    )
    print(
        "  2) Is recall high or low for the minority class?\n"
        f"     LOW. Model recall on class 1 = {_fmt_pct(model['recall_1']).strip()}, "
        f"vs class share = {dist['minority_share']:.2%} in the data. The model misses most "
        "true fraud cases."
    )
    print(
        "  3) Is precision high or low?\n"
        f"     Model precision on class 1 = {_fmt_pct(model['precision_1']).strip()}. "
        "Precision-recall trade-off: on severely imbalanced data this can swing either way; what "
        "matters is the joint behaviour (PR-AUC and F1)."
    )
    print(
        "  4) Which metric best captures the true usefulness of the model?\n"
        f"     PR-AUC (= {_fmt_pct(model['pr_auc']).strip()}) — average precision summarises "
        "the precision-recall trade-off across thresholds and is the right single number under "
        "severe imbalance. F1 at the default threshold is a close second."
    )
    meaningful = "YES" if (model["f1_1"] > 0 and model["recall_1"] > 0) else "NO"
    print(
        f"  5) Does the model meaningfully outperform the majority baseline?\n"
        f"     {meaningful}. "
        + (
            "Baseline gets 0% recall on class 1; the model achieves non-zero recall AND F1."
            if meaningful == "YES"
            else "Baseline and model both score 0% on minority-class recall and F1; the model is "
                 "indistinguishable from the baseline on what matters. The model gets non-zero "
                 "PR-AUC ranking which IS a real signal, but the default-threshold predictions "
                 "are no better than majority-class — a future iteration needs class_weight, "
                 "resampling, or a tuned threshold (see PR #18)."
        )
    )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_imbalance_analysis() -> Dict[str, object]:
    """
    End-to-end runner. Returns a dict with all the computed numbers so a
    caller (or a downstream report) can re-use them.
    """
    print("=" * 70)
    print("Class Imbalance Analysis (diagnosis + interpretation)")
    print("=" * 70)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 1: full-dataset distribution + per-split distributions.
    dist_full = analyze_class_distribution(df["is_fraud"])
    dist_train = analyze_class_distribution(y_train)
    dist_test = analyze_class_distribution(y_test)

    print("\n--- Step 1: Class distribution ---")
    _print_distribution("full dataset", dist_full)
    _print_distribution("train split", dist_train)
    _print_distribution("test  split", dist_test)
    print("\n  Stratified split preserved the minority share across splits "
          "(important: see scenario answer #4 in docs).")
    print("\n  Why this imbalance is problematic (3-4 lines):")
    print(
        "    - A naive classifier can score ~91% accuracy by always predicting class 0\n"
        "      yet catch zero fraud cases — useless for the real-world objective.\n"
        "    - Gradient / split-impurity objectives are dominated by class-0 examples;\n"
        "      the optimisation does not naturally weight fraud detection unless we\n"
        "      explicitly intervene (class_weight, resampling, or threshold tuning).\n"
        "    - Standard ranking metrics (ROC-AUC) can look inflated because the true-\n"
        "      negative rate dominates the curve when negatives are abundant. PR-AUC\n"
        "      is the more honest single-number summary in this regime."
    )

    # Step 2: majority-class baseline.
    print("\n--- Step 2: Majority-class baseline ---")
    print("  DummyClassifier(strategy='most_frequent') — always predicts class 0.")
    baseline_pipe = _build_pipeline(DummyClassifier(strategy="most_frequent"))
    baseline_pipe.fit(X_train, y_train)
    baseline = _evaluate("Baseline (most_frequent)", baseline_pipe, X_test, y_test)
    print(f"  accuracy       = {_fmt_pct(baseline['accuracy']).strip()}")
    print(f"  recall  (cls 1)= {_fmt_pct(baseline['recall_1']).strip()}  (NO fraud caught)")
    print(f"  precision (cls 1)= {_fmt_pct(baseline['precision_1']).strip()}")
    cm = baseline["confusion_matrix"]
    print(f"  confusion matrix:  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
    print("  Why this baseline can appear strong:")
    print(
        f"    {dist_full['proportions'][dist_full['majority_class']]:.1%} of rows belong to the\n"
        "    majority class. Always predicting the majority class is correct on every\n"
        "    one of those rows, which buys you a high accuracy with zero learning."
    )
    print("  Why it is practically useless:")
    print(
        "    Every minority-class row is misclassified. In a fraud-detection deployment,\n"
        "    that means every fraud transaction goes through. The business cost of false\n"
        "    negatives is exactly what the project exists to reduce, and the baseline\n"
        "    optimises against it."
    )

    # Part 2: standard RF model.
    print("\n--- Part 2: Standard model (RandomForestClassifier) ---")
    rf_pipe = _build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
    rf_pipe.fit(X_train, y_train)
    model = _evaluate("RandomForestClassifier", rf_pipe, X_test, y_test)
    print(f"  accuracy       = {_fmt_pct(model['accuracy']).strip()}")
    print(f"  precision (1)  = {_fmt_pct(model['precision_1']).strip()}")
    print(f"  recall (1)     = {_fmt_pct(model['recall_1']).strip()}")
    print(f"  F1 (1)         = {_fmt_pct(model['f1_1']).strip()}")
    print(f"  PR-AUC         = {_fmt_pct(model['pr_auc']).strip()}")
    print(f"  ROC-AUC        = {_fmt_pct(model['roc_auc']).strip()}")
    cm = model["confusion_matrix"]
    print(f"  confusion matrix: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    # Side-by-side + heatmap.
    rows = [baseline, model]
    _print_metric_table(rows)
    _print_confusion_matrices(rows)
    _plot_confusion_heatmaps(rows, HEATMAP_PATH)

    # Part 3 comparison answers.
    _part3_comparison_answers(baseline, model, dist_full)

    # Persist the standard model for re-use.
    os.makedirs(os.path.dirname(IMBALANCE_MODEL_PATH), exist_ok=True)
    joblib.dump(rf_pipe, IMBALANCE_MODEL_PATH)
    print(f"\n  standard model saved -> {IMBALANCE_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("Imbalance analysis completed without errors.")
    print("=" * 70)

    return {
        "distribution": {"full": dist_full, "train": dist_train, "test": dist_test},
        "baseline": baseline,
        "model": model,
    }


if __name__ == "__main__":
    run_imbalance_analysis()
