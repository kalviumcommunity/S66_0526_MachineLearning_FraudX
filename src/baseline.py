"""
baseline.py

Implements baseline classifiers for the Baseline + Class-Imbalance Comparison
assignment.

The FraudX dataset is heavily imbalanced (~91% non-fraud / 9% fraud), so
naive accuracy is misleading. A baseline answers the question: "what does
a trivial, no-learning predictor get on this exact test set?". The trained
RandomForestClassifier should beat it on the metrics that matter (precision /
recall / F1 on the minority fraud class), not just on accuracy.

Two complementary strategies are provided:

1. **`most_frequent`** — always predicts the training-set majority class
   (here: class 0, non-fraud). This is the canonical "if it ain't broke
   don't fix it" lower bound. It maximises accuracy on imbalanced data
   *without learning anything*, which is precisely why it's the most
   useful baseline for exposing the accuracy trap.

2. **`stratified`** — samples predictions from the training-set class prior.
   Provides a non-trivial chance baseline so that "better than always
   predicting 0" doesn't get rewarded by mistake.

Both baselines are fit on `X_train` ONLY (no leakage), in line with the
assignment's "Important Guidelines".
"""
from __future__ import annotations

import os
from typing import Dict

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import (
    BASELINE_MOST_FREQUENT_PATH,
    BASELINE_STRATIFIED_PATH,
    RANDOM_STATE,
)


def get_baseline_models() -> Dict[str, DummyClassifier]:
    """
    Construct (un-fitted) baseline classifiers.

    Returns:
        Mapping of strategy name -> unfit DummyClassifier instance.
    """
    return {
        "most_frequent": DummyClassifier(strategy="most_frequent"),
        "stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    }


def fit_baselines(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, DummyClassifier]:
    """
    Fit each baseline strategy on the TRAINING set only.

    The assignment's Important Guidelines explicitly forbid fitting baselines
    on the full dataset before splitting. This function therefore takes
    `X_train` / `y_train` as inputs — never the full `X` / `y` — and the
    caller is responsible for splitting first.

    Args:
        X_train: Training features (any shape — DummyClassifier ignores them
                 for `most_frequent`, but we still pass them for API
                 consistency and so that `stratified` learns the prior).
        y_train: Training labels.

    Returns:
        Mapping of strategy name -> fitted DummyClassifier.
    """
    print("\n--- Fitting baseline classifiers on TRAINING data only ---")
    fitted: Dict[str, DummyClassifier] = {}
    for name, model in get_baseline_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  Fitted DummyClassifier(strategy='{name}') on {len(y_train)} training samples.")
    return fitted


def save_baselines(baselines: Dict[str, DummyClassifier]) -> Dict[str, str]:
    """
    Persist fitted baselines to `models/baseline_*.pkl` via joblib.

    Returns the mapping of strategy name -> on-disk path so the comparison
    module can log it.
    """
    paths = {
        "most_frequent": BASELINE_MOST_FREQUENT_PATH,
        "stratified": BASELINE_STRATIFIED_PATH,
    }

    written: Dict[str, str] = {}
    for name, path in paths.items():
        if name not in baselines:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(baselines[name], path)
        written[name] = path
        print(f"  Saved baseline '{name}' -> {path}")
    return written


def load_baseline(name: str = "most_frequent") -> DummyClassifier:
    """
    Load a saved baseline.

    Args:
        name: Either "most_frequent" or "stratified".

    Returns:
        Fitted DummyClassifier.

    Raises:
        FileNotFoundError if the artifact isn't on disk yet.
        ValueError on an unknown strategy name.
    """
    path = {
        "most_frequent": BASELINE_MOST_FREQUENT_PATH,
        "stratified": BASELINE_STRATIFIED_PATH,
    }.get(name)
    if path is None:
        raise ValueError(f"Unknown baseline strategy: {name!r}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline {name!r} not found at {path}. "
            "Run `python3 src/comparison.py` (or `python3 main.py`) first."
        )
    return joblib.load(path)
