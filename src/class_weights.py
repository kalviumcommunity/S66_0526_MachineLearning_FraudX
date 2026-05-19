"""
class_weights.py

Class Weights / Cost-Sensitive Learning module.

This module is the *payoff* to the earlier diagnostic work in PR #17
(Baseline Comparison) and PR #21 (Imbalance Analysis). Both modules
concluded that the default `RandomForestClassifier` collapses to
majority-class predictions under FraudX's 91/9 imbalance. The natural
fix flagged in both PRs was to give the model an asymmetric loss via
`class_weight="balanced"`.

This module finally executes that fix and quantifies the *trade-off*
it introduces. The assignment is explicit that the goal is to
understand the trade-off — not just to push a single metric.

Workflow:
  1. Stratified train/test split (reuses src/data_preprocessing).
  2. **Baseline** RandomForest: `RandomForestClassifier(random_state=42)`
     (no class weights). Identical to PR #17 / #21's main model.
  3. **Weighted** RandomForest:
     `RandomForestClassifier(random_state=42, class_weight="balanced")`.
     Same hyperparameters everywhere else so the only thing different is
     the class-weight setting.
  4. Both models are wrapped in the canonical
     `Pipeline(ColumnTransformer + classifier)` so preprocessing is
     identical and leakage-safe.
  5. Identical metric battery for both: accuracy, per-class
     precision / recall / F1, confusion matrix.
  6. Print a side-by-side comparison + a Part-3 inline interpretation
     answering all five required questions with the numbers we just
     computed.
  7. Generate a side-by-side confusion-matrix heatmap.
  8. Persist the weighted model to disk.

`class_weight="balanced"` formula: weights are inversely proportional
to class frequencies in the input data. For a binary problem with
n_samples=800 and class counts {0: 727, 1: 73}, sklearn's default
"balanced" rule computes:
    weight_class = n_samples / (n_classes * n_class_examples)
i.e. ~ 800 / (2 * 727) ≈ 0.55 for class 0 and 800 / (2 * 73) ≈ 5.48
for class 1. The minority class gets weight ~10x the majority class,
which is what re-balances the impurity / loss objective.
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
HEATMAP_PATH = os.path.join(PLOTS_DIR, "class_weights_confusion_matrices.png")
WEIGHTED_MODEL_PATH = os.path.join(BASE_DIR, "models", "weighted_fraud_model.pkl")


# ----------------------------------------------------------------------
# Pipeline construction
# ----------------------------------------------------------------------
def _build_pipeline(class_weight=None) -> Pipeline:
    """
    Build the canonical Pipeline with a configurable class_weight.

    The two callers in this module pass:
      - class_weight=None     → baseline RF
      - class_weight="balanced" → weighted RF

    Same random_state, same preprocessing, same everything else.
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
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        )),
    ])


# ----------------------------------------------------------------------
# Evaluation — single metric helper used by both models
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


def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _print_metric_table(rows) -> None:
    print("\n" + "=" * 80)
    print("Comparison table (assignment Part 3) — same test set, same scoring")
    print("=" * 80)
    print(f"{'Metric':<22s}{'Without Weights':>22s}{'With Weights (balanced)':>26s}")
    print("-" * 80)
    for metric_name, key in [
        ("Accuracy", "accuracy"),
        ("Precision (minority)", "precision_1"),
        ("Recall (minority)", "recall_1"),
        ("F1-score (minority)", "f1_1"),
    ]:
        print(
            f"{metric_name:<22s}"
            f"{_fmt_pct(rows[0][key]):>22s}"
            f"{_fmt_pct(rows[1][key]):>26s}"
        )
    print("=" * 80)


def _print_confusion_matrices(rows) -> None:
    print("\nConfusion matrices (rows = actual, cols = predicted [class 0, class 1]):")
    for r in rows:
        cm = r["confusion_matrix"]
        print(f"\n  {r['label']}:")
        print(f"    actual=0 (legit): predicted_0={cm[0][0]:>4d}  predicted_1={cm[0][1]:>4d}")
        print(f"    actual=1 (fraud): predicted_0={cm[1][0]:>4d}  predicted_1={cm[1][1]:>4d}")
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")


# ----------------------------------------------------------------------
# Heatmap
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

    fig.suptitle("Confusion matrices on the test set — class weights effect",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  heatmap saved -> {path}")


# ----------------------------------------------------------------------
# Part 3 — Comparative analysis answers (runtime, using actual numbers)
# ----------------------------------------------------------------------
def _part3_comparative_answers(baseline: Dict, weighted: Dict) -> None:
    print("\n--- Part 3: Comparative Analysis (answers reference numbers above) ---")

    recall_delta = weighted["recall_1"] - baseline["recall_1"]
    precision_delta = weighted["precision_1"] - baseline["precision_1"]
    accuracy_delta = weighted["accuracy"] - baseline["accuracy"]

    print(
        "  1) How did recall change for the minority class?\n"
        f"     Recall changed by {_fmt_pct(recall_delta).strip()} "
        f"({_fmt_pct(baseline['recall_1']).strip()} -> {_fmt_pct(weighted['recall_1']).strip()}). "
        + ("Class weighting moved the model from catching no fraud to catching some — "
           "exactly what the weighting is supposed to do."
           if recall_delta > 0 else
           "Class weighting did NOT improve recall on this run — likely because the "
           "default RF hyperparameters with `class_weight=balanced` still cannot find "
           "a useful split. Threshold tuning or tree-depth tuning is the next step.")
    )

    print(
        "  2) Did precision increase or decrease? Why?\n"
        f"     Precision changed by {_fmt_pct(precision_delta).strip()} "
        f"({_fmt_pct(baseline['precision_1']).strip()} -> {_fmt_pct(weighted['precision_1']).strip()}). "
        "Weighting makes the model more eager to predict class 1, so it raises recall "
        "at the cost of more false positives — i.e. lower precision. This is the "
        "characteristic trade-off of cost-sensitive learning."
    )

    print(
        "  3) Why did accuracy possibly drop?\n"
        f"     Accuracy changed by {_fmt_pct(accuracy_delta).strip()} "
        f"({_fmt_pct(baseline['accuracy']).strip()} -> {_fmt_pct(weighted['accuracy']).strip()}). "
        "The baseline got ~91% accuracy by predicting class 0 for every row (free credit from "
        "the class prior). The weighted model now predicts class 1 sometimes, "
        "which introduces false positives that didn't exist before. Each FP costs one accuracy "
        "point. The drop is not a regression — it's the visible cost of pursuing recall."
    )

    if weighted["recall_1"] > baseline["recall_1"] and weighted["f1_1"] > baseline["f1_1"]:
        better = "weighted model"
        why = ("the weighted model has both higher recall AND higher F1 on the fraud class. In a "
               "fraud-detection deployment where false negatives have asymmetric cost, that is "
               "the right trade.")
    elif weighted["recall_1"] > baseline["recall_1"]:
        better = "weighted model (with caveats)"
        why = ("the weighted model has higher recall on the fraud class. Even if F1 is slightly "
               "lower, the business will usually prefer catching more fraud and paying more "
               "manual-review cost over missing fraud entirely. Final call belongs to the "
               "business cost function.")
    else:
        better = "neither, on this run"
        why = ("the weighted model did not lift recall above the baseline. The class-weighting "
               "lever alone was not enough; the next step is threshold tuning, tree-depth "
               "tuning, or resampling (SMOTE).")

    print(
        f"  4) Which model is more appropriate? -> {better}.\n"
        f"     Why: {why}"
    )

    print(
        "  5) Does applying class weights completely solve imbalance? Why or why not?\n"
        "     No. `class_weight=\"balanced\"` re-weights the loss but does not change the data "
        "distribution. The model still trains on the same imbalanced features; it just pays "
        "more for missing fraud than for false alarms. Class weights help, but a complete "
        "solution typically combines class weighting + threshold tuning + resampling, with the "
        "operating point chosen from the precision-recall curve and a stated cost function."
    )


def _print_business_recommendation(baseline: Dict, weighted: Dict) -> None:
    """
    Final recommendation, mandated by the assignment's 'Required Outputs in PR'
    list. The recommendation is conditional on the numbers we just saw.
    """
    print("\n--- Final recommendation (business perspective) ---")
    print(
        "  In a fraud-detection deployment, the cost asymmetry is sharp:\n"
        "    - False negative (missed fraud)  = bank eats the full fraud amount.\n"
        "    - False positive (legit blocked) = customer-friction cost (call / re-auth).\n"
        "  A typical FN costs hundreds-to-thousands of dollars; a typical FP costs minutes.\n"
        "  Even at a 50:1 cost ratio, the weighted model is preferred whenever it catches\n"
        "  meaningful additional fraud — provided the false-positive rate stays inside\n"
        "  the team's manual-review capacity."
    )

    if weighted["recall_1"] > baseline["recall_1"]:
        rec_text = (
            f"  RECOMMENDATION: ship the weighted model. It moves fraud recall from "
            f"{_fmt_pct(baseline['recall_1']).strip()} to {_fmt_pct(weighted['recall_1']).strip()}.\n"
            f"  The accuracy drop ({_fmt_pct(baseline['accuracy']).strip()} -> "
            f"{_fmt_pct(weighted['accuracy']).strip()}) is acceptable because it represents\n"
            f"  the cost of catching fraud the baseline could not catch, not a model regression.\n"
            f"  Operate at the default 0.5 threshold; next iteration should tune the threshold\n"
            f"  using the precision-recall curve and the business cost ratio."
        )
    else:
        rec_text = (
            "  RECOMMENDATION: do not ship yet. `class_weight=\"balanced\"` alone did not\n"
            "  lift fraud recall above the baseline on this dataset. The next step is\n"
            "  either threshold tuning (use predict_proba scores to lower the decision\n"
            "  threshold below 0.5) or resampling (SMOTE / random undersample) on the\n"
            "  training set only. The weighted model is still preferred over the baseline\n"
            "  for a future threshold-tuning iteration because its predict_proba scores\n"
            "  are calibrated against an asymmetric loss."
        )
    print(rec_text)


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_class_weights_analysis() -> Dict[str, object]:
    print("=" * 70)
    print("Class Weights / Cost-Sensitive Learning")
    print("=" * 70)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Class distribution summary (required output).
    print("\n--- Class distribution summary ---")
    full_counts = df["is_fraud"].value_counts().sort_index()
    full_shares = df["is_fraud"].value_counts(normalize=True).sort_index()
    print(f"  total rows = {len(df)}")
    for cls in [0, 1]:
        print(f"  class {cls}: count={full_counts[cls]:>4d}  share={full_shares[cls]:6.2%}")
    print(f"  train rows = {len(X_train)}  | test rows = {len(X_test)}")

    # Part 1: Baseline.
    print("\n--- Part 1: Baseline model (no class weights) ---")
    baseline_pipe = _build_pipeline(class_weight=None)
    baseline_pipe.fit(X_train, y_train)
    baseline = _evaluate("Baseline RF (no weights)", baseline_pipe, X_test, y_test)
    print(f"  accuracy            = {_fmt_pct(baseline['accuracy']).strip()}")
    print(f"  precision (cls 1)   = {_fmt_pct(baseline['precision_1']).strip()}")
    print(f"  recall    (cls 1)   = {_fmt_pct(baseline['recall_1']).strip()}")
    print(f"  F1        (cls 1)   = {_fmt_pct(baseline['f1_1']).strip()}")
    cm = baseline["confusion_matrix"]
    print(f"  confusion matrix: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
    print(
        "  Behaviour toward minority class: the model rarely or never predicts class 1.\n"
        "  Accuracy does NOT reflect true usefulness — see PR #17 / #21 for the full diagnosis."
    )

    # Part 2: Weighted.
    print("\n--- Part 2: Weighted model (class_weight='balanced') ---")
    weighted_pipe = _build_pipeline(class_weight="balanced")
    weighted_pipe.fit(X_train, y_train)
    weighted = _evaluate("Weighted RF (balanced)", weighted_pipe, X_test, y_test)
    print(f"  accuracy            = {_fmt_pct(weighted['accuracy']).strip()}")
    print(f"  precision (cls 1)   = {_fmt_pct(weighted['precision_1']).strip()}")
    print(f"  recall    (cls 1)   = {_fmt_pct(weighted['recall_1']).strip()}")
    print(f"  F1        (cls 1)   = {_fmt_pct(weighted['f1_1']).strip()}")
    cm = weighted["confusion_matrix"]
    print(f"  confusion matrix: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    # Part 3: comparative analysis.
    rows = [baseline, weighted]
    _print_metric_table(rows)
    _print_confusion_matrices(rows)
    _plot_confusion_heatmaps(rows, HEATMAP_PATH)
    _part3_comparative_answers(baseline, weighted)

    # Business recommendation (required output).
    _print_business_recommendation(baseline, weighted)

    # Persist the weighted model.
    os.makedirs(os.path.dirname(WEIGHTED_MODEL_PATH), exist_ok=True)
    joblib.dump(weighted_pipe, WEIGHTED_MODEL_PATH)
    print(f"\n  weighted model saved -> {WEIGHTED_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("Class weights module completed without errors.")
    print("=" * 70)

    return {"baseline": baseline, "weighted": weighted}


if __name__ == "__main__":
    run_class_weights_analysis()
