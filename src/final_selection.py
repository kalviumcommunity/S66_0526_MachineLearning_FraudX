"""
final_selection.py

Final Model Selection and Use-Case Alignment — capstone module.

This module does NOT train a new model class. Its job is to synthesise
the candidates evaluated across PRs #17 / #18 / #21 / #22 / #23 / #24
and make a single, justified, deployment-level decision aligned to the
fraud-detection use case.

The use-case constraints:
- Fraud detection
- False Negatives (missed fraud) are SIGNIFICANTLY more costly than
  False Positives (legit transactions blocked)
- Some model interpretability is preferred but not mandatory
- The system must handle moderate real-time traffic

Selection rule (encoded explicitly so the docs can audit it):
  1. Primary metric: **recall on the fraud (positive) class**.
     This is the metric that maps directly to the FN cost objective.
  2. Tie-break #1: F1 on the fraud class (joint precision-recall).
  3. Tie-break #2: CV std (lower = more stable = more deployment-ready).
  4. Tie-break #3: interpretability + inference cost (favour LR > GB > RF).

The slate (six candidates):
  1. Logistic Regression                    — linear baseline.
  2. Random Forest (default)                — incumbent.
  3. Gradient Boosting                      — strongest tabular ensemble candidate.
  4. RandomForest + class_weight="balanced" — cost-sensitive loss (PR #22).
  5. RandomForest + RandomOverSampler       — duplication-based resampling (PR #23).
  6. RandomForest + SMOTE                   — synthetic resampling (PR #23).

All six live in the same `Pipeline(preprocessor + [optional sampler] +
classifier)`. Candidates 4-6 use `imblearn.pipeline.Pipeline` so
resamplers run INSIDE every CV fold (leakage-safe).

Identical scoring metric (`f1` on the fraud class) used for every
candidate's CV. Identical test set used for every candidate's single
sealed evaluation. Identical preprocessing.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
COMPARISON_PLOT_PATH = os.path.join(PLOTS_DIR, "final_selection_comparison.png")
SELECTED_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_selected_model.pkl")

CV_SPLITS = 5
PRIMARY_SCORING = "f1"        # for CV; F1 captures joint precision-recall


# ----------------------------------------------------------------------
# Preprocessor (identical across every candidate)
# ----------------------------------------------------------------------
def _preprocessor() -> ColumnTransformer:
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


# ----------------------------------------------------------------------
# Six candidate pipelines (all wrapped in imblearn.Pipeline so they
# compose with cross_val_score uniformly, including the resampler ones)
# ----------------------------------------------------------------------
def build_candidates() -> Dict[str, ImbPipeline]:
    return {
        "LogisticRegression": ImbPipeline(steps=[
            ("preprocessor", _preprocessor()),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE,
                                              max_iter=1000, solver="lbfgs")),
        ]),
        "RandomForest": ImbPipeline(steps=[
            ("preprocessor", _preprocessor()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "GradientBoosting": ImbPipeline(steps=[
            ("preprocessor", _preprocessor()),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        "RF + class_weight=balanced": ImbPipeline(steps=[
            ("preprocessor", _preprocessor()),
            ("classifier", RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced")),
        ]),
        "RF + RandomOverSampler": ImbPipeline(steps=[
            ("preprocessor", _preprocessor()),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
        "RF + SMOTE": ImbPipeline(steps=[
            ("preprocessor", _preprocessor()),
            ("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]),
    }


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def _evaluate(label: str, fitted, X_test, y_test) -> Dict[str, Any]:
    y_pred = fitted.predict(X_test)
    return {
        "label": label,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_1": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_1": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }


def _cv(pipeline, X_train, y_train) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring=PRIMARY_SCORING, cv=cv, n_jobs=-1)
    return {"mean": float(scores.mean()), "std": float(scores.std())}


def _fmt_pct(x: float) -> str:
    if x != x:
        return "  n/a  "
    return f"{x * 100:6.2f}%"


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _print_comparison_table(rows, cv_results) -> None:
    print("\n" + "=" * 112)
    print("Comparison table (Part 1) — all 6 candidates, identical preprocessing, "
          "identical 5-fold CV, identical metric")
    print("=" * 112)
    header = (
        f"{'Model':<32s}{'CV mean':>10s}{'CV std':>10s}"
        f"{'Test acc':>11s}{'Test P(1)':>11s}{'Test R(1)':>11s}{'Test F1(1)':>13s}"
    )
    print(header)
    print("-" * 112)
    for r in rows:
        cv = cv_results[r["label"]]
        print(
            f"{r['label']:<32s}"
            f"{_fmt_pct(cv['mean']):>10s}"
            f"{_fmt_pct(cv['std']):>10s}"
            f"{_fmt_pct(r['accuracy']):>11s}"
            f"{_fmt_pct(r['precision_1']):>11s}"
            f"{_fmt_pct(r['recall_1']):>11s}"
            f"{_fmt_pct(r['f1_1']):>13s}"
        )
    print("=" * 112)


def _print_confusion_matrices(rows) -> None:
    print("\nConfusion matrices (test set; rows=actual, cols=predicted):")
    for r in rows:
        cm = r["confusion_matrix"]
        print(f"\n  {r['label']}:")
        print(f"    TN={cm[0][0]:>4d}  FP={cm[0][1]:>4d}  FN={cm[1][0]:>4d}  TP={cm[1][1]:>4d}")


def _print_best_numerical_and_most_stable(rows, cv_results) -> None:
    by_test_recall = sorted(rows, key=lambda r: -r["recall_1"])
    by_test_f1 = sorted(rows, key=lambda r: -r["f1_1"])
    by_cv_std = sorted(cv_results.items(), key=lambda kv: kv[1]["std"])

    print("\n--- Highest-numerical and most-stable (Part 1) ---")
    print(f"  Best test recall (cls 1) : {by_test_recall[0]['label']} "
          f"({_fmt_pct(by_test_recall[0]['recall_1']).strip()})")
    print(f"  Best test F1    (cls 1) : {by_test_f1[0]['label']} "
          f"({_fmt_pct(by_test_f1[0]['f1_1']).strip()})")
    print(f"  Most stable (lowest CV std) : {by_cv_std[0][0]} "
          f"(std = {_fmt_pct(by_cv_std[0][1]['std']).strip()})")


# ----------------------------------------------------------------------
# Selection rule
# ----------------------------------------------------------------------
INTERPRETABILITY_PREFERENCE = {
    "LogisticRegression": 4,  # highest (coefficient-level explanations)
    "GradientBoosting": 2,
    "RandomForest": 2,
    "RF + class_weight=balanced": 2,
    "RF + RandomOverSampler": 2,
    "RF + SMOTE": 2,
}


def _select_final(rows: List[Dict[str, Any]],
                  cv_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Encoded selection rule, in order:
      1. Highest test recall on the fraud class (primary metric, maps to
         the FN-cost objective)
      2. Tie-break: highest test F1 (joint precision-recall — guards against
         degenerate "predict class 1 everywhere" solutions)
      3. Tie-break: lowest CV std (stability)
      4. Tie-break: better interpretability score
    """
    keyed_cv = lambda r: cv_results[r["label"]]
    ranked = sorted(
        rows,
        key=lambda r: (
            -r["recall_1"],
            -r["f1_1"],
            keyed_cv(r)["std"],
            -INTERPRETABILITY_PREFERENCE.get(r["label"], 0),
        ),
    )
    return ranked[0]


def _print_selection_and_justification(
    selected_row: Dict[str, Any],
    rows: List[Dict[str, Any]],
    cv_results: Dict[str, Dict[str, float]],
) -> None:
    sel_cv = cv_results[selected_row["label"]]

    print("\n" + "=" * 112)
    print(f"FINAL SELECTED MODEL: {selected_row['label']}")
    print("=" * 112)
    print(
        "  Selection rule (encoded; tie-broken in order):\n"
        "    1. Highest test recall on the fraud class    (primary metric for this use case)\n"
        "    2. Highest test F1 on the fraud class        (joint precision-recall)\n"
        "    3. Lowest CV std                              (stability)\n"
        "    4. Better interpretability                    (operational preference)"
    )

    print("\n--- Use-case alignment (Part 2) ---")
    print(
        "  Scenario: fraud detection. FN cost >> FP cost. Interpretability\n"
        "  preferred but not mandatory. Moderate real-time traffic.\n"
        "\n"
        "  Prioritised metric: **recall on the fraud (positive) class**.\n"
        "  Reason: a False Negative is a fraud transaction that goes through —\n"
        "  the bank eats the full amount. A False Positive is a customer-friction\n"
        "  cost (minutes of re-auth). The cost ratio is typically 50:1 or higher,\n"
        "  so the model that catches more fraud is worth tolerating extra FPs."
    )

    print("\n--- Justification (Part 2 / Part 3) ---")
    print(
        f"  Metric evidence       : recall (cls 1) = {_fmt_pct(selected_row['recall_1']).strip()}, "
        f"F1 (cls 1) = {_fmt_pct(selected_row['f1_1']).strip()}, "
        f"precision (cls 1) = {_fmt_pct(selected_row['precision_1']).strip()}, "
        f"accuracy = {_fmt_pct(selected_row['accuracy']).strip()}."
    )
    print(
        f"  CV evidence           : CV mean F1 = {_fmt_pct(sel_cv['mean']).strip()} "
        f"(std = {_fmt_pct(sel_cv['std']).strip()})."
    )
    print(
        "  Train-test gap        : reported via the imbalance-handling modules (PRs #22/#23).\n"
        "                          Resampler models lift recall at the cost of some accuracy;\n"
        "                          this is the documented trade-off, not a regression."
    )
    cm = selected_row["confusion_matrix"]
    print(
        f"  Confusion matrix      : TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}.\n"
        f"                          {cm[1][1]} true positive(s) on a test set with {cm[1][0]+cm[1][1]} fraud cases.\n"
        f"                          {cm[0][1]} false positive(s) on a test set with {cm[0][0]+cm[0][1]} legit cases."
    )

    print("\n--- Holistic evaluation (Part 3) ---")
    print(
        "  - Interpretability:\n"
        "      Logistic Regression has the strongest interpretability story\n"
        "      (per-feature coefficients). RF / GB / resampler variants don't,\n"
        "      without SHAP / per-tree inspection. If the deployment must justify\n"
        "      every prediction to a regulator, LR is the right choice — even at\n"
        "      lower recall. For this scenario, interpretability is 'preferred\n"
        "      but not mandatory', so it does not override the recall priority."
    )
    print(
        "  - Computational cost at inference:\n"
        "      LR: O(features) per prediction — single dot product.\n"
        "      RF / RF+ClassWeight / RF+RandomOS / RF+SMOTE: O(n_trees × depth).\n"
        "      GB: O(n_trees × depth) sequential.\n"
        "      For 'moderate real-time traffic', all six are easily fast enough."
    )
    print(
        "  - Stability across CV folds:\n"
        "      CV stds in the table above. Lower = more stable across\n"
        "      training-set shuffles. A model that's stable AND has reasonable\n"
        "      recall is preferred over a higher-recall-but-noisier alternative."
    )
    print(
        "  - Improvement over baseline:\n"
        "      Baseline (majority-class predictor) gets 0% recall on fraud.\n"
        "      Any candidate with non-zero recall is a meaningful improvement\n"
        "      on the only metric that matters for this use case."
    )

    if selected_row["recall_1"] == 0.0:
        print(
            "\n  CAVEAT: the selected model still catches 0 fraud cases at the default\n"
            "  threshold. This is not a model-class problem — it's the imbalance\n"
            "  ceiling identified in PRs #17/#21/#22. The recommendation is to\n"
            "  ship the SELECTED model and pair it with threshold tuning\n"
            "  (use predict_proba; lower the decision threshold below 0.5;\n"
            "  pick the operating point from the precision-recall curve with a\n"
            "  stated business cost ratio). The selected model's predict_proba\n"
            "  scores are the right input for that step."
        )

    print(
        "\n  What might change this decision in a different business context?\n"
        "    - If FP cost dominates (e.g., customer churn risk too high): switch to\n"
        "      higher-precision candidate; prioritise precision over recall.\n"
        "    - If interpretability is mandatory (regulated industry): switch to LR\n"
        "      and accept lower recall.\n"
        "    - If inference cost matters (very high QPS): switch to LR.\n"
        "    - If the dataset is much larger (millions of rows): SMOTE becomes\n"
        "      slower; RandomOverSampler or class_weight is preferable.\n"
        "    - If features change such that signal becomes linear: LR may pull\n"
        "      ahead of the tree-based candidates on recall too."
    )


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
def _plot_comparison_barchart(rows, cv_results, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping chart.")
        return

    labels = [r["label"] for r in rows]
    recall = [r["recall_1"] * 100 for r in rows]
    f1 = [r["f1_1"] * 100 for r in rows]
    cv_mean = [cv_results[l]["mean"] * 100 for l in labels]
    cv_std = [cv_results[l]["std"] * 100 for l in labels]

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width, recall, width, label="Test recall (cls 1)", color="#C44E52")
    b2 = ax.bar(x,         f1,     width, label="Test F1 (cls 1)",     color="#4C72B0")
    b3 = ax.bar(x + width, cv_mean, width, yerr=cv_std, capsize=4,
                label="CV mean F1 (with std)", color="#55A868")

    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Final selection — 6-model fair comparison (test recall / test F1 / CV mean F1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  comparison chart saved -> {path}")


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_final_selection() -> Dict[str, Any]:
    print("=" * 70)
    print("Final Model Selection and Use-Case Alignment (capstone)")
    print("=" * 70)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    candidates = build_candidates()
    cv_results: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}

    print(f"\n--- Evaluating {len(candidates)} candidates "
          f"(5-fold CV, scoring='{PRIMARY_SCORING}', same test set) ---")

    for label, pipe in candidates.items():
        # CV on a fresh pipeline (the dict's pipe will be fitted on full train below).
        cv = _cv(pipe, X_train, y_train)
        cv_results[label] = cv

        # Single fit on full train; single sealed test evaluation.
        pipe.fit(X_train, y_train)
        fitted[label] = pipe
        r = _evaluate(label, pipe, X_test, y_test)
        rows.append(r)
        print(
            f"  {label:<32s}  CV mean={_fmt_pct(cv['mean']).strip():>7s}  "
            f"std={_fmt_pct(cv['std']).strip():>7s}  "
            f"test recall(1)={_fmt_pct(r['recall_1']).strip():>7s}  "
            f"test F1(1)={_fmt_pct(r['f1_1']).strip():>7s}"
        )

    _print_comparison_table(rows, cv_results)
    _print_confusion_matrices(rows)
    _print_best_numerical_and_most_stable(rows, cv_results)
    _plot_comparison_barchart(rows, cv_results, COMPARISON_PLOT_PATH)

    # Selection.
    selected_row = _select_final(rows, cv_results)
    _print_selection_and_justification(selected_row, rows, cv_results)

    # Persist the selected pipeline.
    os.makedirs(os.path.dirname(SELECTED_MODEL_PATH), exist_ok=True)
    joblib.dump(fitted[selected_row["label"]], SELECTED_MODEL_PATH)
    print(f"\n  Selected model artifact -> {SELECTED_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("Final selection module completed without errors.")
    print("=" * 70)

    return {
        "rows": rows,
        "cv_results": cv_results,
        "selected": selected_row["label"],
    }


if __name__ == "__main__":
    run_final_selection()
