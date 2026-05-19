"""
model_comparison.py

Multi-Model Comparison with Cross-Validation.

The assignment is explicit: train at least three different models and
compare them using a FAIR and CONSISTENT evaluation strategy.
"Disciplined comparison and reasoned model selection" — not just chasing
the highest score.

This module trains three classifiers on the SAME train/test split, with
the SAME preprocessing pipeline, the SAME 5-fold StratifiedKFold CV
strategy, and the SAME scoring metric. The three models are:

  1. Logistic Regression  — linear baseline with L2 regularisation.
     Fast, interpretable, calibrated probabilities by maximum likelihood.
  2. Random Forest         — bagged trees, low variance, the project's
     incumbent model in every prior module.
  3. Gradient Boosting     — sequential trees fit on residuals. Often
     the strongest tabular learner of the three; provides the most
     interesting comparison vs RF.

The output: a CV table (mean ± std per model), a single test-set
evaluation per model, a bar-chart visualisation, and a final justified
model selection.

Discipline:
  - All three live in the SAME `Pipeline(ColumnTransformer + classifier)`.
  - Cross-validation runs on the Pipeline, so the ColumnTransformer
    re-fits inside every CV fold on that fold's training rows only.
  - Test set is sealed throughout CV; evaluated exactly once per model.
"""
from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
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
CV_BARCHART_PATH = os.path.join(PLOTS_DIR, "model_comparison_cv.png")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_comparison_model.pkl")

CV_SPLITS = 5
SCORING = "f1"


# ----------------------------------------------------------------------
# The common preprocessing pipeline (identical for every model)
# ----------------------------------------------------------------------
def _preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ])


# ----------------------------------------------------------------------
# Three candidate models
# ----------------------------------------------------------------------
def _build_pipeline(classifier_label: str) -> Pipeline:
    """Build a Pipeline for one of the three named classifiers."""
    classifiers = {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, solver="lbfgs"
        ),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if classifier_label not in classifiers:
        raise ValueError(f"Unknown classifier label: {classifier_label!r}")
    return Pipeline(steps=[
        ("preprocessor", _preprocessor()),
        ("classifier", classifiers[classifier_label]),
    ])


# ----------------------------------------------------------------------
# Evaluation helpers
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
    if x != x:
        return "  n/a  "
    return f"{x * 100:6.2f}%"


def _print_cv_table(cv_results: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 70)
    print("Part 3: Cross-Validation results (5-fold StratifiedKFold, scoring='f1')")
    print("=" * 70)
    header = f"{'Model':<22s}{'CV Mean':>14s}{'CV Std':>14s}"
    print(header)
    print("-" * 70)
    for label, c in cv_results.items():
        print(f"{label:<22s}{_fmt_pct(c['mean']):>14s}{_fmt_pct(c['std']):>14s}")
    print("=" * 70)


def _print_combined_table(eval_rows, cv_results) -> None:
    print("\n" + "=" * 92)
    print("Part 4: Combined comparison — CV + Test (same X, same metric for all models)")
    print("=" * 92)
    header = (
        f"{'Model':<22s}{'CV Mean':>11s}{'CV Std':>11s}{'Test F1':>11s}"
        f"{'Test Prec(1)':>14s}{'Test Recall(1)':>16s}{'Test Acc':>10s}"
    )
    print(header)
    print("-" * 92)
    for r in eval_rows:
        cv = cv_results[r["label"]]
        print(
            f"{r['label']:<22s}"
            f"{_fmt_pct(cv['mean']):>11s}"
            f"{_fmt_pct(cv['std']):>11s}"
            f"{_fmt_pct(r['f1_1']):>11s}"
            f"{_fmt_pct(r['precision_1']):>14s}"
            f"{_fmt_pct(r['recall_1']):>16s}"
            f"{_fmt_pct(r['accuracy']):>10s}"
        )
    print("=" * 92)


def _print_confusion_matrices(eval_rows) -> None:
    print("\nConfusion matrices (rows = actual, cols = predicted [class 0, class 1]):")
    for r in eval_rows:
        cm = r["confusion_matrix"]
        print(f"\n  {r['label']}:")
        print(f"    actual=0: predicted_0={cm[0][0]:>4d}  predicted_1={cm[0][1]:>4d}")
        print(f"    actual=1: predicted_0={cm[1][0]:>4d}  predicted_1={cm[1][1]:>4d}")
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")


def _plot_cv_barchart(cv_results: Dict[str, Dict[str, float]], path: str) -> None:
    """CV mean ± std bar chart for the three models."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping bar chart.")
        return

    labels = list(cv_results.keys())
    means = [cv_results[l]["mean"] * 100 for l in labels]
    stds = [cv_results[l]["std"] * 100 for l in labels]

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  edgecolor="black", linewidth=0.6, alpha=0.9)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.15 + 0.5,
                f"{mean:.2f}% ± {std:.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CV mean F1 (fraud class, %)")
    ax.set_title("Model comparison — 5-fold StratifiedKFold CV  (error bars = std)")
    ymax = max(m + s for m, s in zip(means, stds)) + max(stds) * 0.6 + 2
    ax.set_ylim(0, max(ymax, 5))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  CV bar chart saved -> {path}")


# ----------------------------------------------------------------------
# Part 3 / 4 / 5 interpretation
# ----------------------------------------------------------------------
def _interpret_cv(cv_results: Dict[str, Dict[str, float]]) -> str:
    """Identify highest-mean and lowest-std and return the best candidate label."""
    by_mean = sorted(cv_results.items(), key=lambda kv: -kv[1]["mean"])
    by_std = sorted(cv_results.items(), key=lambda kv: kv[1]["std"])

    best_mean_label, best_mean_stats = by_mean[0]
    lowest_std_label, lowest_std_stats = by_std[0]

    print("\n--- Part 3 interpretation ---")
    print(
        f"  Highest CV mean       : {best_mean_label} "
        f"({_fmt_pct(best_mean_stats['mean']).strip()})"
    )
    print(
        f"  Most stable (low std) : {lowest_std_label} "
        f"({_fmt_pct(lowest_std_stats['std']).strip()})"
    )
    print(
        "  High variance (large CV std) implies the model's performance depends\n"
        "  strongly on which fold the data falls into — i.e., it's sensitive to\n"
        "  the specific subset of training rows. Low variance means the model\n"
        "  generalises consistently across folds. Under a tied or near-tied mean,\n"
        "  the lower-variance model is the more trustworthy production choice."
    )

    # Tie-breaker rule for selection: highest mean wins, but if two means are
    # within 1pp of each other, prefer the lower-variance candidate.
    by_mean_sorted = sorted(cv_results.items(), key=lambda kv: -kv[1]["mean"])
    top1_label, top1 = by_mean_sorted[0]
    top2_label, top2 = by_mean_sorted[1]
    if abs(top1["mean"] - top2["mean"]) < 0.01:
        winner = top1_label if top1["std"] <= top2["std"] else top2_label
        rule = (
            f"top two CV means within 1pp ({top1_label}={_fmt_pct(top1['mean']).strip()} vs "
            f"{top2_label}={_fmt_pct(top2['mean']).strip()}) — tie-broken by lower std: {winner}"
        )
    else:
        winner = top1_label
        rule = f"highest CV mean wins: {winner} ({_fmt_pct(top1['mean']).strip()})"

    print(f"\n  Selected candidate (CV-based)  : {winner}")
    print(f"  Selection rule                  : {rule}")
    return winner


def _print_part5_answers(eval_rows, cv_results, selected: str) -> None:
    print("\n--- Part 5: Comparative Analysis answers ---")

    print(
        "  1) Why is cross-validation better than a single train/test split for model comparison?\n"
        "     A single split is a sample of size 1 from the distribution of possible splits. The\n"
        "     metric you read is one random draw, and on small / imbalanced data the variance\n"
        "     across splits is large. CV computes that metric over k different holdouts; the\n"
        "     mean is closer to the true expected performance and the std tells you how much\n"
        "     a single split could mislead you. On 200 test rows here, a single-split metric\n"
        "     can swing several pp just from which 18 fraud cases happen to be in the test set."
    )

    print(
        "  2) Why must the test set not be used to select the best model repeatedly?\n"
        "     Because every selection decision implicitly fits the model family to the test set.\n"
        "     If you pick the best of 3 (or 30) models by their test scores, you've effectively\n"
        "     run a hyperparameter search where the hyperparameter is 'which model class', and\n"
        "     the test score is no longer an unbiased estimate of generalisation — it's an\n"
        "     overfit-on-the-test-set artifact. The correct workflow is: pick by CV, evaluate\n"
        "     the chosen model ONCE on the test set, report that single number."
    )

    print(
        "  3) If two models differ by 0.01 in score but one has lower variance, which would you\n"
        "     choose and why?\n"
        "     Lower variance, every time. A 0.01-point difference is well inside the noise of\n"
        "     5-fold CV on a 200-sample test set; the higher-mean model is not statistically\n"
        "     distinguishable from the lower-mean one. The lower-variance model is the more\n"
        "     reliable production choice — its CV std tells you the expected swing in performance\n"
        "     across deployments, which is exactly the quantity an SRE / risk-management process\n"
        "     cares about."
    )

    print(
        "  4) How does model complexity affect bias and variance?\n"
        "     Higher-capacity models (deeper trees, more parameters, less regularisation) reduce\n"
        "     bias — they can fit more complex patterns — but raise variance, because they\n"
        "     latch onto patterns specific to the particular training set. Lower-capacity\n"
        "     models do the reverse: more bias (they cannot capture rare patterns), less\n"
        "     variance (they're stable across training sets). The sweet spot depends on the\n"
        "     data size relative to the signal complexity. In our CV results, a low-std model\n"
        "     suggests lower variance — but if its mean is also low, the bias is too high and\n"
        "     it underfits."
    )

    base = next(r for r in eval_rows if r["label"] == "RandomForest")
    print(
        "  5) When might a slightly lower-performing model be preferable?\n"
        "     When operational concerns matter:\n"
        "     - Interpretability: Logistic Regression has coefficient-level explanations;\n"
        "       Gradient Boosting and Random Forest don't (without SHAP / per-tree inspection).\n"
        "       A regulated industry may prefer the LR even at lower accuracy.\n"
        "     - Inference latency / cost: GB scoring is sequential and slower than LR; for\n"
        "       high-throughput fraud scoring at the payment-gateway tier, the cheaper model\n"
        "       can be the right call.\n"
        "     - Training cost / retraining cadence: if the model is re-fit nightly on growing\n"
        "       data, LR scales much better than GB.\n"
        "     - Calibration: LR produces well-calibrated probabilities out of the box; GB and\n"
        "       RF don't — they need Platt / isotonic calibration to be useful in cost-sensitive\n"
        "       threshold decisions.\n"
        f"     For FraudX, the selected model is {selected}. If a deployment constraint (latency,\n"
        "     interpretability, retraining cost) tilts the choice away from it, the answers above\n"
        "     name the legitimate reasons."
    )


def _print_final_selection(eval_rows, cv_results, selected: str) -> None:
    """Print the final justified model selection — required output."""
    selected_eval = next((r for r in eval_rows if r["label"] == selected), None)
    selected_cv = cv_results[selected]

    print("\n--- Final justified model selection (required output) ---")
    print(f"  Selected model : **{selected}**")
    print(f"  Justification  : highest 5-fold CV mean F1 = {_fmt_pct(selected_cv['mean']).strip()}, "
          f"CV std = {_fmt_pct(selected_cv['std']).strip()}.")
    if selected_eval:
        print(f"  Sealed test F1 : {_fmt_pct(selected_eval['f1_1']).strip()} "
              f"(precision={_fmt_pct(selected_eval['precision_1']).strip()}, "
              f"recall={_fmt_pct(selected_eval['recall_1']).strip()}, "
              f"acc={_fmt_pct(selected_eval['accuracy']).strip()})")
        cv_test_gap = selected_cv["mean"] - selected_eval["f1_1"]
        if abs(cv_test_gap) > 0.05:
            print(
                f"  Note: CV-vs-test gap = {_fmt_pct(cv_test_gap).strip()}. A large gap (either\n"
                "  direction) suggests the model is sensitive to the specific test draw or\n"
                "  CV variance is high. The reported CV mean remains the better estimate of\n"
                "  expected performance."
            )

    print(
        "\n  Why this beats picking by test score: the test set is now 'spent' — it's served\n"
        "  its purpose as a single unbiased check on the CV-selected candidate. Future model\n"
        "  comparisons should generate a new held-out test set (or use nested CV) to avoid\n"
        "  multiple-comparisons inflation on the held-out metric."
    )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def run_model_comparison() -> Dict[str, object]:
    print("=" * 70)
    print("Multi-Model Comparison with Cross-Validation")
    print("=" * 70)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Part 1: data preparation (implicit — same preprocessing in every pipeline).
    print("\n--- Part 1: Data preparation ---")
    print("  Stratified train/test split: train=800 rows, test=200 rows.")
    print("  Preprocessing pipeline (identical for every model):")
    print("    - numerical : SimpleImputer(median) -> StandardScaler")
    print("    - categorical: SimpleImputer(most_frequent) -> OneHotEncoder")
    print("    - ColumnTransformer glues them together; re-fit inside each CV fold.")
    print("  Why consistent preprocessing is necessary for fair comparison:")
    print(
        "    Models compared with different preprocessing aren't compared at all — the\n"
        "    difference in scores could be the model OR the preprocessing. Identical\n"
        "    preprocessing isolates the model choice as the only variable."
    )

    # Part 2: choose three models.
    model_labels = ["LogisticRegression", "RandomForest", "GradientBoosting"]
    print("\n--- Part 2: Three candidate models ---")
    for label in model_labels:
        print(f"  - {label}")

    # Part 3: Cross-validation.
    print(f"\n--- Part 3: 5-fold StratifiedKFold CV (scoring='{SCORING}') ---")
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results: Dict[str, Dict[str, float]] = {}
    for label in model_labels:
        # Build a fresh pipeline per model so each cross_val_score call starts clean.
        pipe = _build_pipeline(label)
        scores = cross_val_score(pipe, X_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1)
        cv_results[label] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {label:<22s}  CV mean F1 = {scores.mean():.4f}  std = {scores.std():.4f}")

    _print_cv_table(cv_results)
    _plot_cv_barchart(cv_results, CV_BARCHART_PATH)
    selected_label = _interpret_cv(cv_results)

    # Part 4: Test set evaluation — fit on full train, evaluate once on test.
    print("\n--- Part 4: Test-set evaluation (one shot per model) ---")
    eval_rows = []
    fitted_pipes = {}
    for label in model_labels:
        pipe = _build_pipeline(label)
        pipe.fit(X_train, y_train)
        fitted_pipes[label] = pipe
        r = _evaluate(label, pipe, X_test, y_test)
        eval_rows.append(r)

    _print_combined_table(eval_rows, cv_results)
    _print_confusion_matrices(eval_rows)
    print(
        "\n  Does test performance align with CV performance?\n"
        "    Compare each row's CV Mean vs Test F1. A small gap (< 5pp) means CV is a\n"
        "    reliable estimate of test performance. A large gap (> 5pp) means either the\n"
        "    CV variance was high (look at CV Std) or the single test draw is unusually\n"
        "    easy / hard for that model class.\n"
        "  Was any model overfitting?\n"
        "    Overfitting in CV shows up as 'CV mean >> CV std' on the training side but\n"
        "    'Test F1 << CV mean' on the test side. We don't compute training F1 here\n"
        "    because the assignment scopes Part 4 to test-vs-CV comparison only; for the\n"
        "    full overfitting analysis see PR #18 (tuning) which reports train vs test gaps."
    )

    # Part 5 + final selection.
    _print_part5_answers(eval_rows, cv_results, selected_label)
    _print_final_selection(eval_rows, cv_results, selected_label)

    # Persist the selected model for downstream use.
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(fitted_pipes[selected_label], BEST_MODEL_PATH)
    print(f"\n  Selected model artifact -> {BEST_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("Model comparison completed without errors.")
    print("=" * 70)
    return {
        "cv_results": cv_results,
        "eval_rows": eval_rows,
        "selected": selected_label,
    }


if __name__ == "__main__":
    run_model_comparison()
