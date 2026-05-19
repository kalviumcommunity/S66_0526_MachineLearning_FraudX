"""
main.py

Responsible for:
- Orchestrating the end-to-end machine learning pipeline
- Coordinating data flow between the isolated modules (loader, train, predict)
- Serving as the primary entry point for the project
"""
import pandas as pd

from src.model_comparison import run_model_comparison
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

    # 3. Multi-model comparison with cross-validation.
    # Trains Logistic Regression / Random Forest / Gradient Boosting on
    # the SAME preprocessing pipeline, runs 5-fold StratifiedKFold CV with
    # consistent scoring, evaluates each model on the SAME test set, and
    # produces a justified final selection.
    print("\n[Phase 3] Multi-Model Comparison with Cross-Validation")
    run_model_comparison()

    print("\n--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
