"""
main.py

Responsible for:
- Orchestrating the end-to-end machine learning pipeline
- Coordinating data flow between the isolated modules (loader, train, predict)
- Serving as the primary entry point for the project
"""
import pandas as pd

from src.oversampling import run_oversampling_analysis
from src.predict import predict
from src.train import train_pipeline


def main():
    print("--- Starting Modular FraudX ML Pipeline ---")

    # 1. Run the training pipeline
    # This loads data, preprocesses, trains, evaluates, and saves artifacts.
    print("\n[Phase 1] Training & Artifact Generation")
    train_pipeline()

    # 2. Run a sample inference
    # This demonstrates the separation: predict loads its own artifacts.
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

    # 3. Oversampling for imbalanced classification
    # Trains Baseline RF + RandomOverSampler+RF + SMOTE+RF on the same split,
    # cross-validates each via imblearn.Pipeline (no leakage), reports the
    # full comparison table and a business recommendation.
    print("\n[Phase 3] Oversampling for Imbalanced Classification")
    run_oversampling_analysis()

    print("\n--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
