"""
tuning.py

Hyperparameter tuning for the FraudX RandomForestClassifier using
RandomizedSearchCV — implementation for the Hyperparameter Tuning assignment.

Why RandomizedSearchCV instead of GridSearchCV
----------------------------------------------
A grid over 4 hyperparameters with even a modest range explodes combinatorially:
e.g. n_estimators={50,100,200,500} * max_depth={3,5,10,20,30} * min_samples_leaf=
{1,5,10,20} * max_features={"sqrt","log2"} = 4 * 5 * 4 * 2 = 160 candidates.
At 5-fold CV that's 800 model fits. With ranges any wider, the cost balloons.

RandomizedSearchCV samples a fixed `n_iter` candidates from a *distribution* over
each hyperparameter, so coverage of the search space scales with compute budget
rather than with the product of grid sizes. Empirically (Bergstra & Bengio,
2012), random search matches or beats grid search in fewer iterations because
many hyperparameters have low effective sensitivity — sampling them at random
just doesn't waste evaluations on near-identical settings.

What this module does
---------------------
1. Builds a `Pipeline(preprocessor + RandomForestClassifier)` so the
   ColumnTransformer is re-fit inside every CV fold on that fold's training
   rows only — zero leakage even across 30 * 5 = 150 fits.
2. Defines distributions (NOT grids) for `n_estimators`, `max_depth`,
   `min_samples_leaf`, and `max_features` — the four hyperparameters the
   assignment example identifies as important for Random Forest.
3. Trains a *baseline* RF (default sklearn hyperparameters) and reports
   training / test / 5-fold CV mean+std F1 so we have a non-trivial reference
   point for the "did tuning actually help?" question.
4. Runs RandomizedSearchCV with `n_iter=30`, 5-fold StratifiedKFold,
   `scoring="f1"`, and a fixed `random_state` for reproducibility. Only
   `X_train` / `y_train` ever sees the search — the test set stays sealed.
5. Reports the best params, best CV score, CV std at the best point, and
   the number of iterations explored.
6. Evaluates the tuned best estimator ONCE on the held-out test set, prints a
   side-by-side comparison of baseline vs tuned (train F1, test F1, CV mean,
   train/test gap), and writes a CSV of the full search history to
   `reports/tuning_results.csv` for the visualisation step.
7. Generates a scatter visualisation of `max_depth` (x-axis) vs CV mean F1
   (y-axis), coloured by `min_samples_leaf`, saved to
   `reports/plots/tuning_results.png`.
8. Persists the tuned best estimator to `models/tuned_fraud_model.pkl` via
   joblib.

All metrics use the same scoring metric (`f1` on the positive / fraud class)
to satisfy the assignment's "Do NOT compare different metrics" guideline.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TUNED_MODEL_PATH,
)
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline


# Fixed configuration for reproducibility.
N_ITER = 30
CV_SPLITS = 5
SCORING = "f1"  # Binary-F1 on the positive (fraud) class.


def build_search_pipeline() -> Pipeline:
    """
    Construct the Pipeline that RandomizedSearchCV will fit and search over.

    Wrapping the ColumnTransformer + RandomForestClassifier in a Pipeline is
    critical: it means each CV fold's preprocessing is fit on that fold's
    training half only. If we instead fit the ColumnTransformer once on
    X_train and then used X_train_processed inside CV, every fold would see
    a preprocessor that had learned from the other folds' validation rows
    — preprocessing leakage that biases CV scores high.
    """
    preprocessor = build_preprocessing_pipeline(CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def get_param_distributions() -> Dict[str, Any]:
    """
    Distributions (NOT fixed grids) over the four Random Forest hyperparameters
    the assignment example identifies as important.

    Reasoning per hyperparameter:

    * `n_estimators` — randint(50, 500). More trees lower variance via
      averaging but raise compute cost roughly linearly. 50 is a sensible
      lower bound (any lower and the ensemble is too small to stabilise);
      500 is an upper bound where added trees stop helping in practice.

    * `max_depth` — randint(3, 30). The single biggest bias-variance lever
      for tree ensembles. Shallow trees (3-5) systematically underfit on
      non-trivial data; very deep trees (>20) start to memorise individual
      rows on small datasets like FraudX (n=800 train).

    * `min_samples_leaf` — randint(1, 20). The smaller leaf size, the more
      individual rows can carve out their own region (overfit). Larger
      leaves force the tree to find generalisable patterns. Especially
      relevant on imbalanced data where a single minority-class row in a
      leaf can produce a degenerate split.

    * `max_features` — ["sqrt", "log2"]. Discrete categorical: how many
      features each split considers. "sqrt" is the sklearn default and
      almost always a sensible starting point; "log2" gives more aggressive
      feature subsampling, which can help when features are correlated.
    """
    return {
        "classifier__n_estimators": randint(50, 500),
        "classifier__max_depth": randint(3, 30),
        "classifier__min_samples_leaf": randint(1, 20),
        "classifier__max_features": ["sqrt", "log2"],
    }


def _evaluate_train_test(model, X_train, X_test, y_train, y_test, cv: StratifiedKFold) -> Dict[str, float]:
    """
    Compute training F1, test F1, 5-fold CV mean F1, CV std, and the
    train/test gap. The model is assumed already fit.

    `cross_val_score` is run on a fresh clone of the model on X_train, so it
    does not contaminate the already-fit model — and crucially it does NOT
    touch X_test.
    """
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_f1 = f1_score(y_train, train_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)

    cv_scores = cross_val_score(model, X_train, y_train, scoring=SCORING, cv=cv, n_jobs=-1)

    return {
        "train_f1": float(train_f1),
        "test_f1": float(test_f1),
        "cv_mean_f1": float(cv_scores.mean()),
        "cv_std_f1": float(cv_scores.std()),
        "train_test_gap": float(train_f1 - test_f1),
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _print_metric_table(rows: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 86)
    print("Baseline RF vs Tuned RF (all scores are F1 on the fraud / positive class)")
    print("=" * 86)
    header = f"{'Model':<28s}{'Train F1':>11s}{'Test F1':>11s}{'CV mean':>11s}{'CV std':>10s}{'Train-Test':>13s}"
    print(header)
    print("-" * 86)
    for label, m in rows.items():
        print(
            f"{label:<28s}"
            f"{_fmt_pct(m['train_f1']):>11s}"
            f"{_fmt_pct(m['test_f1']):>11s}"
            f"{_fmt_pct(m['cv_mean_f1']):>11s}"
            f"{_fmt_pct(m['cv_std_f1']):>10s}"
            f"{_fmt_pct(m['train_test_gap']):>13s}"
        )
    print("=" * 86)


def _interpret_improvement(baseline_m: Dict[str, float], tuned_m: Dict[str, float]) -> None:
    test_delta = tuned_m["test_f1"] - baseline_m["test_f1"]
    gap_delta = tuned_m["train_test_gap"] - baseline_m["train_test_gap"]
    cv_delta = tuned_m["cv_mean_f1"] - baseline_m["cv_mean_f1"]

    print("\n--- Tuning impact ---")
    print(f"  Test F1 change      : {_fmt_pct(test_delta):>7s}")
    print(f"  CV mean F1 change   : {_fmt_pct(cv_delta):>7s}")
    print(f"  Train-Test gap chg  : {_fmt_pct(gap_delta):>7s}  (negative = less overfit)")

    if tuned_m["test_f1"] > baseline_m["test_f1"] + 0.05:
        verdict = (
            "VERDICT: Tuning produced a meaningful improvement on test F1 (> 5 pp)."
        )
    elif tuned_m["test_f1"] > baseline_m["test_f1"]:
        verdict = (
            "VERDICT: Tuning moved test F1 in the right direction but the gain is\n"
            "  modest. Worth more iterations or expanded search space before\n"
            "  declaring a real improvement."
        )
    elif tuned_m["test_f1"] == baseline_m["test_f1"] == 0.0:
        verdict = (
            "VERDICT: Both baseline and tuned model score 0 on test F1 — the search\n"
            "  could not find hyperparameters that overcome the class imbalance.\n"
            "  Next step: add `class_weight=\"balanced\"` to the search space or\n"
            "  resample the training data (SMOTE / under-sample). Tuning alone\n"
            "  doesn't address a class-distribution problem."
        )
    else:
        verdict = (
            "VERDICT: Tuning did not improve test F1 in this run. Reasons could be:\n"
            "  search budget too small (try higher n_iter), search ranges too wide,\n"
            "  or the model class is mis-specified for this data."
        )
    print(f"  {verdict}")


def _save_tuning_results_csv(search: RandomizedSearchCV, path: str) -> None:
    """Persist the full cv_results_ table for the visualisation + reproducibility."""
    df = pd.DataFrame(search.cv_results_)
    keep_cols = [c for c in df.columns if c.startswith("param_") or c in {"mean_test_score", "std_test_score", "rank_test_score"}]
    df = df[keep_cols].sort_values("rank_test_score")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  cv_results_ table -> {path}")


def _plot_tuning_results(search: RandomizedSearchCV, path: str) -> None:
    """
    Scatter of `classifier__max_depth` vs mean CV F1, colored by
    `classifier__min_samples_leaf`. Saved as PNG.

    Reads from `search.cv_results_` so no extra fits happen.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping visualisation.")
        return

    cv = search.cv_results_
    x = cv["param_classifier__max_depth"].data.astype(float)
    y = cv["mean_test_score"]
    color = cv["param_classifier__min_samples_leaf"].data.astype(float)
    sizes = cv["param_classifier__n_estimators"].data.astype(float)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        x,
        y,
        c=color,
        cmap="viridis",
        s=(sizes / sizes.max()) * 200 + 30,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )
    cb = plt.colorbar(scatter, ax=ax)
    cb.set_label("min_samples_leaf")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("mean CV F1 (fraud class)")
    ax.set_title(
        f"RandomizedSearchCV: {len(x)} sampled candidates\n"
        "(marker size = n_estimators)"
    )
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  visualisation     -> {path}")


def run_hyperparameter_tuning() -> Dict[str, Any]:
    """
    End-to-end tuning routine.

    Returns a dict with `baseline`, `tuned`, `best_params`, `n_iter`, and
    `cv_results_csv_path` so a caller (or `main.py`) can post-process the
    output if needed.
    """
    print("=" * 70)
    print("Hyperparameter Tuning with RandomizedSearchCV")
    print("=" * 70)

    # 1. Load + clean + split BEFORE any model is constructed.
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 2. Baseline RF (sklearn defaults) for the "did tuning help?" comparison.
    print("\n--- Step 1: Fitting baseline RandomForestClassifier (sklearn defaults) ---")
    baseline = build_search_pipeline()
    baseline.fit(X_train, y_train)
    baseline_metrics = _evaluate_train_test(baseline, X_train, X_test, y_train, y_test, cv)
    print(
        f"  train F1 = {baseline_metrics['train_f1']:.4f} | "
        f"test F1 = {baseline_metrics['test_f1']:.4f} | "
        f"CV mean F1 = {baseline_metrics['cv_mean_f1']:.4f} (std {baseline_metrics['cv_std_f1']:.4f})"
    )
    if baseline_metrics["cv_mean_f1"] >= baseline_metrics["test_f1"] + 0.05:
        print("  Note: CV mean noticeably higher than test F1 — variance across folds is real.")
    if baseline_metrics["train_test_gap"] >= 0.15:
        print("  Note: large train-test gap suggests the default RF overfits this dataset.")
    elif baseline_metrics["train_test_gap"] <= -0.05:
        print("  Note: negative train-test gap is unusual — usually indicates near-zero predictions everywhere.")

    # 3. Run RandomizedSearchCV.
    print(f"\n--- Step 2: RandomizedSearchCV (n_iter={N_ITER}, cv={CV_SPLITS}, scoring='{SCORING}') ---")
    print("  All fits happen on X_train only. The test set is sealed until step 4.")
    search = RandomizedSearchCV(
        estimator=build_search_pipeline(),
        param_distributions=get_param_distributions(),
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    best_index = int(search.best_index_)
    best_cv_std = float(search.cv_results_["std_test_score"][best_index])

    print(f"  best_score_ (CV mean F1) : {search.best_score_:.4f}")
    print(f"  std at best point        : {best_cv_std:.4f}")
    print(f"  n_iter explored          : {N_ITER}")
    print("  best_params_             :")
    for k, v in search.best_params_.items():
        print(f"    - {k}: {v}")

    # 4. Single, sealed test-set evaluation of the tuned best estimator.
    print("\n--- Step 3: Evaluate tuned best estimator on held-out test set (ONCE) ---")
    tuned_metrics = _evaluate_train_test(
        search.best_estimator_, X_train, X_test, y_train, y_test, cv
    )
    print(
        f"  train F1 = {tuned_metrics['train_f1']:.4f} | "
        f"test F1 = {tuned_metrics['test_f1']:.4f} | "
        f"CV mean F1 = {tuned_metrics['cv_mean_f1']:.4f} (std {tuned_metrics['cv_std_f1']:.4f})"
    )

    # 5. Side-by-side comparison + verdict.
    _print_metric_table({
        "Baseline RF (sklearn defaults)": baseline_metrics,
        "Tuned RF (RandomizedSearchCV)": tuned_metrics,
    })
    _interpret_improvement(baseline_metrics, tuned_metrics)

    # 6. Persist artifacts.
    print("\n--- Step 4: Persist artifacts ---")
    os.makedirs(os.path.dirname(TUNED_MODEL_PATH), exist_ok=True)
    joblib.dump(search.best_estimator_, TUNED_MODEL_PATH)
    print(f"  tuned model       -> {TUNED_MODEL_PATH}")

    csv_path = os.path.join(os.path.dirname(TUNED_MODEL_PATH), "..", "reports", "tuning_results.csv")
    csv_path = os.path.normpath(csv_path)
    _save_tuning_results_csv(search, csv_path)

    plot_path = os.path.normpath(
        os.path.join(os.path.dirname(TUNED_MODEL_PATH), "..", "reports", "plots", "tuning_results.png")
    )
    _plot_tuning_results(search, plot_path)

    print("\n" + "=" * 70)
    print("Hyperparameter tuning completed without errors.")
    print("=" * 70)

    return {
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
        "best_params": dict(search.best_params_),
        "best_cv_score": float(search.best_score_),
        "best_cv_std": best_cv_std,
        "n_iter": N_ITER,
        "cv_results_csv_path": csv_path,
    }


if __name__ == "__main__":
    run_hyperparameter_tuning()
