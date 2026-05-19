"""
model_persistence.py

Model Persistence with Pickle.

Part 1 — Train and validate a model (the capstone selection from
PR #25: RF + RandomOverSampler inside an imblearn Pipeline).
Part 2 — Save the ENTIRE preprocessing + model pipeline via
`pickle.dump()` to a `.pkl` file.
Part 3 — Invoke `src/load_and_verify.py` AS A SEPARATE PYTHON
SUBPROCESS so it imports nothing from this module's in-memory state.
The subprocess reloads the .pkl from disk via `pickle.load()`, makes
predictions on the same X_test, and writes its metrics + the
predictions array to a JSON file. This module then reads that JSON
file and asserts that the loaded-model predictions match the
original-model predictions byte-for-byte.
Part 4 — Reflection answers (serialization, pipeline-vs-model,
security, versioning) covered in `docs/MODEL_PERSISTENCE.md` and
printed at runtime.

Why subprocess and not just a separate function in this same process?
Because a same-process load is a less-strict test: the in-memory
state of this script (imports, parsed types, sklearn estimator
classes) is already populated, so `pickle.load(...)` can succeed
even if a real fresh-environment load would fail (e.g., missing
import paths, dataclass identity differences). A subprocess gives
us the closest thing to "restart the kernel" the assignment asks
for, without leaving the project's `python3 src/...` workflow.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from typing import Dict, List

import numpy as np
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
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data


PERSISTED_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "persisted_pipeline.pkl")
VERIFICATION_JSON_PATH = os.path.join(BASE_DIR, "reports", "load_and_verify.json")
LOAD_AND_VERIFY_SCRIPT = os.path.join(BASE_DIR, "src", "load_and_verify.py")


def build_capstone_pipeline() -> ImbPipeline:
    """
    Rebuilds the capstone pipeline from PR #25: RF + RandomOverSampler
    inside imblearn.Pipeline, with the same ColumnTransformer
    preprocessing used everywhere in the project.
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


def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _evaluate(label: str, predictions, y_test) -> Dict[str, object]:
    return {
        "label": label,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_1": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_1": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "predictions_hash": int(hash(predictions.tobytes())),  # byte-level fingerprint
    }


def _print_metrics(label: str, metrics: Dict[str, object]) -> None:
    cm = metrics["confusion_matrix"]
    print(
        f"  {label:<32s}  acc={_fmt_pct(metrics['accuracy']).strip():>7s}  "
        f"P(1)={_fmt_pct(metrics['precision_1']).strip():>7s}  "
        f"R(1)={_fmt_pct(metrics['recall_1']).strip():>7s}  "
        f"F1(1)={_fmt_pct(metrics['f1_1']).strip():>7s}  "
        f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}"
    )


def _print_reflection() -> None:
    print("\n" + "=" * 70)
    print("Part 4: Reflection (short answers — full versions in docs/MODEL_PERSISTENCE.md)")
    print("=" * 70)
    print(
        "  1) What is serialization?\n"
        "     Serialization = converting a live Python object (with its references,\n"
        "     attributes, weights) into a byte stream that can be written to disk\n"
        "     or sent over a network. Pickle does this for arbitrary Python objects\n"
        "     by recording the class identity + the object's __dict__ + any\n"
        "     associated state. Deserialization (pickle.load) reverses the process\n"
        "     and reconstructs the object in a new Python process."
    )
    print(
        "  2) Why is saving the entire pipeline better than saving only the model?\n"
        "     The trained model expects features that were produced by the SAME\n"
        "     preprocessor that was fitted during training (same medians for the\n"
        "     imputer, same means/variances for the scaler, same category\n"
        "     vocabulary for the encoder). If you save only the classifier and\n"
        "     re-fit the preprocessor at load time, the preprocessor's parameters\n"
        "     will differ from the training-time ones (different input data,\n"
        "     different fold of the same data, etc.) and inference predictions\n"
        "     drift silently. Saving the whole Pipeline freezes ALL these\n"
        "     parameters together so a loaded pipeline is byte-identical at\n"
        "     inference."
    )
    print(
        "  3) What security risk exists when loading Pickle files?\n"
        "     pickle.load() can execute ARBITRARY CODE during deserialisation\n"
        "     (it constructs objects by calling their __reduce__ method, which\n"
        "     can be any function the pickle author specifies). A malicious\n"
        "     .pkl file can therefore run shell commands, exfiltrate data, or\n"
        "     install backdoors when loaded. NEVER load .pkl files from\n"
        "     untrusted sources. Mitigations: signed artifacts, restricted\n"
        "     environments (containers / sandboxes), or use safer formats\n"
        "     (ONNX, JSON for simple models, joblib's memory-mapping for arrays)."
    )
    print(
        "  4) What could go wrong if library versions differ?\n"
        "     Pickle records the class IDENTITY (module path + class name) but\n"
        "     not the implementation. If the class's internal layout has changed\n"
        "     between the version that pickled the object and the version that\n"
        "     loads it (e.g., sklearn renamed an attribute, deprecated a tree\n"
        "     structure, changed the fit-result schema), deserialisation may\n"
        "     succeed silently but produce wrong predictions, OR fail loudly\n"
        "     with AttributeError. Mitigations: pin versions in requirements.txt,\n"
        "     store metadata (sklearn version, imblearn version, numpy version)\n"
        "     in a sidecar JSON file next to the .pkl, and refuse to load when\n"
        "     versions don't match."
    )


def run_model_persistence() -> Dict[str, object]:
    """End-to-end:  train -> save -> load-in-subprocess -> verify -> reflect."""
    print("=" * 70)
    print("Model Persistence with Pickle")
    print("=" * 70)

    # --- Part 1: Train + validate (the capstone selection from PR #25) ---
    print("\n--- Part 1: Train and validate (capstone: RF + RandomOverSampler) ---")
    df = clean_data(load_data(RAW_DATA_PATH))
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_capstone_pipeline()
    pipeline.fit(X_train, y_train)
    original_predictions = pipeline.predict(X_test)
    original_metrics = _evaluate("Original (in-memory)", original_predictions, y_test)
    _print_metrics("Original (in-memory)", original_metrics)

    # --- Part 2: Save with pickle.dump() ---
    print("\n--- Part 2: Save via pickle.dump() ---")
    os.makedirs(os.path.dirname(PERSISTED_PIPELINE_PATH), exist_ok=True)
    with open(PERSISTED_PIPELINE_PATH, "wb") as fp:
        pickle.dump(pipeline, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_size = os.path.getsize(PERSISTED_PIPELINE_PATH)
    print(f"  Wrote {pkl_size:,} bytes to {PERSISTED_PIPELINE_PATH}")
    print(f"  File extension: .pkl  ✓")
    print(f"  Saved: entire preprocessor + sampler + classifier pipeline  ✓")

    # --- Part 3: Load and verify in a SEPARATE PYTHON SUBPROCESS ---
    print("\n--- Part 3: Load and verify in a FRESH Python process ---")
    print(f"  Invoking: python3 {LOAD_AND_VERIFY_SCRIPT}")
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR  # so the subprocess can import src.*
    result = subprocess.run(
        [sys.executable, LOAD_AND_VERIFY_SCRIPT],
        cwd=BASE_DIR, env=env, capture_output=True, text=True,
    )
    print("  --- subprocess stdout ---")
    for line in result.stdout.splitlines():
        print(f"    {line}")
    if result.stderr.strip():
        print("  --- subprocess stderr ---")
        for line in result.stderr.splitlines():
            print(f"    {line}")
    if result.returncode != 0:
        raise RuntimeError(
            f"load_and_verify.py subprocess failed with exit code {result.returncode}"
        )

    # --- Read the verification JSON the subprocess wrote ---
    with open(VERIFICATION_JSON_PATH) as fp:
        loaded_metrics = json.load(fp)
    loaded_preds = np.array(loaded_metrics["predictions"], dtype=original_predictions.dtype)

    print("\n--- Cross-process metric comparison ---")
    _print_metrics("Original (in-memory)", original_metrics)
    _print_metrics("Loaded (subprocess)", loaded_metrics)

    # --- Byte-identical predictions (encoded assertion) ---
    print("\n--- Verification asserts ---")
    same_predictions = np.array_equal(original_predictions, loaded_preds)
    same_metrics = (
        np.isclose(original_metrics["accuracy"], loaded_metrics["accuracy"])
        and np.isclose(original_metrics["precision_1"], loaded_metrics["precision_1"])
        and np.isclose(original_metrics["recall_1"], loaded_metrics["recall_1"])
        and np.isclose(original_metrics["f1_1"], loaded_metrics["f1_1"])
    )

    print(f"  np.array_equal(orig_preds, loaded_preds)  : {same_predictions}")
    print(f"  metrics match (accuracy / P(1) / R(1) / F1(1)) : {same_metrics}")
    assert same_predictions, "FAILURE: loaded predictions differ from original predictions."
    assert same_metrics, "FAILURE: loaded metrics differ from original metrics."
    print("  [VERIFIED] Loaded pipeline produces byte-identical predictions and metrics.")

    # --- Part 4: Reflection ---
    _print_reflection()

    print("\n" + "=" * 70)
    print("Model persistence module completed without errors.")
    print("=" * 70)

    return {
        "original": original_metrics,
        "loaded": loaded_metrics,
        "predictions_match": same_predictions,
        "metrics_match": same_metrics,
        "pkl_path": PERSISTED_PIPELINE_PATH,
        "pkl_size_bytes": pkl_size,
    }


if __name__ == "__main__":
    run_model_persistence()
