"""
main.py

Responsible for:
- Orchestrating the end-to-end machine learning pipeline
- Coordinating data flow between the isolated modules (loader, train, predict)
- Serving as the primary entry point for the project
"""
import pandas as pd

from src.final_selection import run_final_selection
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

    # 3. Final model selection (capstone of the model-comparison work).
    # Trains the 6 candidates from PRs #22, #23, #24 on the SAME split,
    # produces the comparison table, applies the use-case-aligned selection
    # rule (recall primary, F1 / CV-std / interpretability tie-breakers),
    # and persists the winner.
    print("\n[Phase 3] Final Model Selection and Use-Case Alignment")
    run_final_selection()

    print("\n--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()
