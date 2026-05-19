"""
main.py

Final-system orchestrator for the FraudX project.

Runs the canonical training + inference workflow AND builds the
deployment artifacts (models/pipeline.joblib + pipeline_metadata.json)
that `app.py` reads. Every prior module's code (PRs #15-#27) is
available in `src/` for direct invocation:

    python3 src/normalization.py        # PR #15  — MinMaxScaler workflow
    python3 src/comparison.py           # PR #17  — baseline vs RF
    python3 src/tuning.py               # PR #18  — RandomizedSearchCV
    python3 src/pipeline_demo.py        # PR #19  — Pipeline integration
    python3 src/leakage_correction.py   # PR #20  — 4-leakage audit
    python3 src/imbalance_analysis.py   # PR #21  — imbalance diagnosis
    python3 src/class_weights.py        # PR #22  — class weighting
    python3 src/oversampling.py         # PR #23  — Random + SMOTE
    python3 src/model_comparison.py     # PR #24  — LR/RF/GB head-to-head
    python3 src/final_selection.py      # PR #25  — capstone selection
    python3 src/model_persistence.py    # PR #26  — pickle round-trip
    python3 src/inference_demo.py       # PR #27  — production inference

This file runs the leakage-safe training pipeline, an inference demo,
and the deployment-artifact build that the Streamlit app depends on.
"""
import pandas as pd

from src.deployment import export_deployment_artifacts
from src.predict import predict
from src.train import train_pipeline


def main():
    print("--- Starting Modular FraudX ML Pipeline ---")

    # 1. Run the training pipeline
    # This loads data, preprocesses (with MinMaxScaler from PR #15),
    # trains, evaluates, saves the canonical model + preprocessor +
    # standalone MinMaxScaler artifacts.
    print("\n[Phase 1] Training & Artifact Generation")
    train_pipeline()

    # 2. Run a sample inference using the existing predict module.
    # Demonstrates the separation between training and inference.
    print("\n[Phase 2] Inference Demonstration")
    sample_input = pd.DataFrame([{
        "amount": 120.5,
        "transaction_count": 2,
        "velocity": 0.8,
        "category": "food",
        "location": "domestic",
    }])

    prediction = predict(sample_input)
    print(f"Sample Input Prediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")

    # 3. Final-system deployment artifacts
    # Build the capstone pipeline (RF + RandomOverSampler from PR #25),
    # fit on the same train/test split, persist via joblib to
    # models/pipeline.joblib, and write a sidecar metadata JSON with
    # library versions + test metrics. These are the artifacts the
    # Streamlit app (`streamlit run app.py`) reads at startup.
    print("\n[Phase 3] Build Deployment Artifacts (for Streamlit app)")
    export_deployment_artifacts()

    print("\n--- Pipeline Completed Successfully ---")
    print("\nNext steps:")
    print("  - To launch the Streamlit app:   streamlit run app.py")
    print("  - To explore individual modules: see src/<module>.py listed in this file's docstring.")


if __name__ == "__main__":
    main()
