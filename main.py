"""
main.py

Responsible for:
- Orchestrating the end-to-end machine learning pipeline
- Coordinating data flow between the isolated modules (loader, train, predict)
- Serving as the primary entry point for the project
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


if __name__ == "__main__":
    main()
