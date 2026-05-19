"""
deployment.py

Final-system deployment artifact builder.

Trains the capstone pipeline (RF + RandomOverSampler, selected in PR #25),
persists it via `joblib.dump` to `models/pipeline.joblib`, and writes a
sidecar metadata JSON at `models/pipeline_metadata.json` containing:
- library versions used at training time (sklearn, imblearn, numpy, python)
- test-set performance metrics (accuracy, per-class precision/recall/F1,
  confusion matrix)
- training timestamp
- the random seed and the feature schema

This module is the bridge between the experimentation phase (PRs #15-#27)
and the deployed Streamlit app. The app reads pipeline.joblib + metadata
once at startup (cached via @st.cache_resource) and uses the loaded
pipeline for all predictions.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict

import joblib
import numpy as np
import sklearn
import imblearn
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
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
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    DEPLOYMENT_METADATA_PATH,
    DEPLOYMENT_PIPELINE_PATH,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data


def build_capstone_pipeline() -> ImbPipeline:
    """The final selected pipeline from PR #25:
    Preprocessor (impute + scale + encode) -> RandomOverSampler -> RF.

    Wrapped in imblearn.Pipeline so the sampler runs INSIDE CV (leakage-safe).
    """
    num_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ])
    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def _compute_test_metrics(fitted_pipeline, X_test, y_test) -> Dict[str, Any]:
    y_pred = fitted_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy":      float(accuracy_score(y_test, y_pred)),
        "precision_1":   float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_1":      float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_1":          float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "test_set_size": int(len(y_test)),
        "test_fraud_share": float(np.mean(y_test == 1)),
    }


def export_deployment_artifacts() -> Dict[str, Any]:
    """
    Build, fit, persist the capstone pipeline + metadata.

    Returns a dict with everything written to disk so a caller can log it.
    """
    print("=" * 70)
    print("Building deployment artifacts (final-system release)")
    print("=" * 70)

    # 1. Train.
    df = clean_data(load_data(RAW_DATA_PATH))
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_capstone_pipeline()
    pipeline.fit(X_train, y_train)
    print(f"\n  Fitted capstone pipeline on {len(y_train)} training samples.")

    # 2. Evaluate on the sealed test set.
    test_metrics = _compute_test_metrics(pipeline, X_test, y_test)
    print(f"  Test perf: acc={test_metrics['accuracy']:.4f}  "
          f"P(1)={test_metrics['precision_1']:.4f}  "
          f"R(1)={test_metrics['recall_1']:.4f}  "
          f"F1(1)={test_metrics['f1_1']:.4f}")

    # 3. Persist the pipeline via joblib (the assignment specifies joblib for
    #    the final-system artifact; PRs #26/#27 used pickle for that module's
    #    explicit pickle-vs-load-and-verify story).
    os.makedirs(os.path.dirname(DEPLOYMENT_PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline, DEPLOYMENT_PIPELINE_PATH)
    pkl_size = os.path.getsize(DEPLOYMENT_PIPELINE_PATH)
    print(f"\n  joblib.dump -> {DEPLOYMENT_PIPELINE_PATH}  ({pkl_size:,} bytes)")

    # 4. Write the sidecar metadata JSON. This is the "refuse-to-load on
    #    version mismatch" hook PR #26 recommended.
    metadata = {
        "model_description": "RandomForestClassifier with RandomOverSampler oversampling, "
                              "selected in PR #25 (Final Model Selection) for FraudX. "
                              "Capstone of PRs #15-#27.",
        "pipeline_steps": [
            "preprocessor: ColumnTransformer(SimpleImputer(median)+StandardScaler | "
            "SimpleImputer(most_frequent)+OneHotEncoder(handle_unknown=ignore))",
            "sampler: RandomOverSampler(random_state=42)",
            "classifier: RandomForestClassifier(random_state=42)",
        ],
        "training": {
            "random_state": RANDOM_STATE,
            "train_size":   int(len(y_train)),
            "test_size":    int(len(y_test)),
            "train_fraud_share": float(np.mean(y_train == 1)),
            "test_fraud_share":  float(np.mean(y_test == 1)),
            "trained_at":   datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "feature_schema": {
            "numerical_features":   NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        "test_metrics": test_metrics,
        "library_versions": {
            "python":       sys.version.split()[0],
            "platform":     platform.platform(),
            "sklearn":      sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "numpy":        np.__version__,
            "pandas":       pd.__version__,
            "joblib":       joblib.__version__,
        },
    }
    with open(DEPLOYMENT_METADATA_PATH, "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"  metadata    -> {DEPLOYMENT_METADATA_PATH}")

    print("\n" + "=" * 70)
    print("Deployment artifacts built. Run `streamlit run app.py` to launch the UI.")
    print("=" * 70)

    return {"pipeline_path": DEPLOYMENT_PIPELINE_PATH,
            "metadata_path": DEPLOYMENT_METADATA_PATH,
            "metadata": metadata}


if __name__ == "__main__":
    export_deployment_artifacts()
