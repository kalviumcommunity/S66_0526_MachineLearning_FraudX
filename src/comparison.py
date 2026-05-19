"""
comparison.py

End-to-end Baseline + Class-Imbalance Comparison module.

Workflow:
    1. Load raw data and clean.
    2. Train-test split BEFORE any fitting (no leakage).
    3. Build the same ColumnTransformer pipeline used in training.
    4. Fit the pipeline on X_train, transform both X_train and X_test.
    5. Fit RandomForestClassifier on the processed training data
       (the main model — mirrors what `train.py` does).
    6. Fit DummyClassifier baselines (`most_frequent`, `stratified`)
       on the PROCESSED training data. Fitting on processed features
       guarantees baseline and main model see identically shaped inputs,
       which is required for an apples-to-apples comparison.
    7. Evaluate all three classifiers on the held-out test set using
       `evaluate_detailed` — per-class metrics + balanced accuracy +
       confusion matrix.
    8. Print a side-by-side comparison table.
    9. Compute and PRINT the improvement of the main model over the
       majority-class baseline on accuracy, balanced accuracy, and
       minority-class F1. Assert that the same X_test / y_test was used
       throughout.
    10. Persist baselines to `models/baseline_*.pkl`.

Run directly:
    PYTHONPATH=. python3 src/comparison.py
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.baseline import fit_baselines, save_baselines
from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data
from src.evaluate import evaluate_detailed
from src.feature_engineering import build_preprocessing_pipeline


def _fmt_pct(x: float) -> str:
    """Render a 0-1 float as a 5-character percentage (e.g. ' 91.0%')."""
    return f"{x * 100:5.1f}%"


def _print_comparison_table(results: List[dict]) -> None:
    """
    Side-by-side metric comparison.

    All baselines + the main model are evaluated with identical metrics.
    Per-class numbers (precision/recall/F1) are reported because accuracy
    alone hides the minority-class blind spot under heavy imbalance.
    """
    print("\n" + "=" * 86)
    print("Side-by-side comparison (test set; identical X_test / y_test for all models)")
    print("=" * 86)

    header = (
        f"{'Model':<28s}"
        f"{'Acc':>8s}"
        f"{'BalAcc':>8s}"
        f"{'P0':>8s}{'R0':>8s}{'F1_0':>8s}"
        f"{'P1':>8s}{'R1':>8s}{'F1_1':>8s}"
    )
    print(header)
    print("-" * 86)

    for r in results:
        line = (
            f"{r['label']:<28s}"
            f"{_fmt_pct(r['accuracy']):>8s}"
            f"{_fmt_pct(r['balanced_accuracy']):>8s}"
            f"{_fmt_pct(r['class_0']['precision']):>8s}"
            f"{_fmt_pct(r['class_0']['recall']):>8s}"
            f"{_fmt_pct(r['class_0']['f1']):>8s}"
            f"{_fmt_pct(r['class_1']['precision']):>8s}"
            f"{_fmt_pct(r['class_1']['recall']):>8s}"
            f"{_fmt_pct(r['class_1']['f1']):>8s}"
        )
        print(line)

    print("=" * 86)
    print("Legend: P0/R0/F1_0 = precision/recall/F1 for class 0 (non-fraud);")
    print("        P1/R1/F1_1 = same for class 1 (fraud); BalAcc = balanced accuracy.")


def _print_confusion_matrices(results: List[dict]) -> None:
    print("\nConfusion matrices (rows = actual, cols = predicted [class 0, class 1]):")
    for r in results:
        cm = r["confusion_matrix"]
        print(f"\n  {r['label']}:")
        print(f"    actual=0:  predicted_0={cm[0][0]:>4d}  predicted_1={cm[0][1]:>4d}")
        print(f"    actual=1:  predicted_0={cm[1][0]:>4d}  predicted_1={cm[1][1]:>4d}")


def _report_improvement(main_result: dict, baseline_result: dict) -> Dict[str, float]:
    """
    Compute the main model's improvement over the majority-class baseline
    on three metrics that actually matter on imbalanced data.
    """
    improvements = {
        "accuracy": main_result["accuracy"] - baseline_result["accuracy"],
        "balanced_accuracy": main_result["balanced_accuracy"] - baseline_result["balanced_accuracy"],
        "class_1_f1": main_result["class_1"]["f1"] - baseline_result["class_1"]["f1"],
        "class_1_recall": main_result["class_1"]["recall"] - baseline_result["class_1"]["recall"],
    }

    print("\n--- Improvement of main model over `most_frequent` baseline ---")
    print(f"  accuracy           : {_fmt_pct(improvements['accuracy']):>8s}  (often misleading on imbalanced data)")
    print(f"  balanced_accuracy  : {_fmt_pct(improvements['balanced_accuracy']):>8s}  (treats both classes equally)")
    print(f"  recall (class 1)   : {_fmt_pct(improvements['class_1_recall']):>8s}  (fraud catch rate — what we actually want to maximize)")
    print(f"  F1 (class 1)       : {_fmt_pct(improvements['class_1_f1']):>8s}  (joint precision/recall on fraud class)")

    if improvements["class_1_f1"] <= 0 and improvements["class_1_recall"] <= 0:
        print(
            "\n  VERDICT: The main model does NOT meaningfully beat the majority-class\n"
            "  baseline on minority-class metrics. Accuracy improvement (if any) is\n"
            "  essentially noise. Future work: class weighting, resampling (SMOTE),\n"
            "  cost-sensitive thresholds, or richer features."
        )
    elif improvements["class_1_f1"] >= 0.10:
        print(
            "\n  VERDICT: The main model clearly beats the baseline on minority-class F1\n"
            "  (improvement >= 10 percentage points). The trained model is justified."
        )
    else:
        print(
            "\n  VERDICT: The main model beats the baseline on minority-class metrics but\n"
            "  the margin is modest. Worth running more experiments before declaring victory."
        )

    return improvements


def run_baseline_comparison() -> List[dict]:
    """
    End-to-end comparison routine. Returns the list of per-model evaluation
    dicts so a caller can post-process (e.g., dump to a report).
    """
    print("=" * 70)
    print("Baseline + Class-Imbalance Comparison (Assignment)")
    print("=" * 70)

    # 1. Load and clean.
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nFull-dataset class distribution:")
    print(df["is_fraud"].value_counts(normalize=True).round(4))

    # 2. Split BEFORE any fitting. `split_data` already enforces stratify + random_state.
    X_train, X_test, y_train, y_test = split_data(df)

    # 3 & 4. Build pipeline, fit on training data only, transform both sides.
    pipeline = build_preprocessing_pipeline(CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # 5. Main model — RandomForestClassifier, same hyperparams as src/train.py.
    print("\n--- Fitting main model (RandomForestClassifier) on processed training data ---")
    main_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    main_model.fit(X_train_processed, y_train)
    print(f"  Fitted RandomForestClassifier on {X_train_processed.shape[0]} samples, "
          f"{X_train_processed.shape[1]} features.")

    # 6. Baselines on processed training data.
    baselines = fit_baselines(X_train_processed, y_train)

    # 7. Evaluate all three on the IDENTICAL held-out test set.
    results = [
        evaluate_detailed(baselines["most_frequent"], X_test_processed, y_test, "baseline_most_frequent"),
        evaluate_detailed(baselines["stratified"], X_test_processed, y_test, "baseline_stratified"),
        evaluate_detailed(main_model, X_test_processed, y_test, "RandomForestClassifier"),
    ]

    # Sanity asserts: every model was evaluated on the same test size.
    sizes = [r["class_0"]["support"] + r["class_1"]["support"] for r in results]
    assert len(set(sizes)) == 1, (
        f"Models were evaluated on different test sizes: {sizes}. "
        "This would invalidate the comparison."
    )
    assert all(r["class_0"]["support"] == results[0]["class_0"]["support"] for r in results), (
        "Per-class support differs across models — they must have seen the SAME y_test."
    )

    # 8. Print comparison + confusion matrices.
    _print_comparison_table(results)
    _print_confusion_matrices(results)

    # 9. Improvement report (main model vs majority-class baseline).
    _report_improvement(results[2], results[0])

    # 10. Persist baselines.
    print("\n--- Persisting baseline artifacts ---")
    save_baselines(baselines)

    print("\n" + "=" * 70)
    print("Baseline comparison completed without errors.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_baseline_comparison()
