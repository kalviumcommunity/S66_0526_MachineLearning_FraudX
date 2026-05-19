"""
main.py

Responsible for:
- Orchestrating the end-to-end machine learning pipeline
- Coordinating data flow between the isolated modules (loader, train, predict)
- Serving as the primary entry point for the project
"""
import pandas as pd

from src.inference_demo import run_inference_demo
from src.predict import predict
from src.train import train_pipeline


def main():
    print("--- Starting Modular FraudX ML Pipeline ---")

    # 1. Run the training pipeline
    # This loads data, preprocesses, trains, evaluates, and saves artifacts.
    print("\n[Phase 1] Training & Artifact Generation")
    train_pipeline()

    # 2. Run a sample inference using the project's existing predict module.
    print("\n[Phase 2] Inference Demonstration (existing predict module)")
    sample_input = pd.DataFrame([{
        "amount": 120.5,
        "transaction_count": 2,
        "velocity": 0.8,
        "category": "food",
        "location": "domestic",
    }])

    prediction = predict(sample_input)
    print(f"Sample Input Prediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")

    # 3. Production-inference demo on the persisted .pkl artifact.
    # Loads pickle.load(...), scores 5 hand-crafted new transactions
    # (predict + predict_proba), re-verifies test-set perf, asserts
    # no .fit() ran during inference. Writes a CSV of new predictions.
    print("\n[Phase 3] Production Inference on Persisted Model")
    run_inference_demo()

    print("\n--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
