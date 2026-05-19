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

from src.normalization import run_normalization_pipeline
from src.predict import predict
from src.train import train_pipeline


def main():
    print("--- Starting Modular FraudX ML Pipeline ---")

    # 1. Standalone MinMaxScaler normalization demo (Assignment 5.18)
    # Demonstrates the leakage-safe split → fit → transform → verify → save
    # workflow in its most explicit form, independent of the full pipeline.
    print("\n[Phase 0] Standalone Normalization Demo (MinMaxScaler)")
    run_normalization_pipeline()

    # 2. Run the training pipeline
    # This loads data, preprocesses (MinMaxScaler + OneHotEncoder via the
    # ColumnTransformer), trains, evaluates, and saves all artifacts —
    # including a standalone MinMaxScaler at models/minmax_scaler.pkl.
    print("\n[Phase 1] Training & Artifact Generation")
    train_pipeline()

    # 3. Run a sample inference using the saved artifacts.
    # predict() loads the fitted pipeline and calls .transform() — never
    # .fit_transform() — so no leakage occurs at inference time.
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

    print("\n--- Pipeline Completed Successfully ---")
    print("\nNext steps:")
    print("  - To launch the Streamlit app:   streamlit run app.py")
    print("  - To explore individual modules: see src/<module>.py listed in this file's docstring.")



if __name__ == "__main__":
    main()
