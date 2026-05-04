"""
leakage_demo.py

This script demonstrates Target Leakage in a machine learning pipeline.
It compares a 'cheating' model (with leaked information) against a 'honest' model.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_data
from src.config import RAW_DATA_PATH, TARGET_COLUMN, RANDOM_STATE

def run_leakage_demo():
    print("--- 🛡️ Data Leakage Demonstration ---")
    
    # 1. Load the original dataset
    df = load_data(RAW_DATA_PATH)
    
    # 2. CREATE LEAKAGE: Adding a 'transaction_processed_status' feature
    # This feature is derived directly from the target. 
    # In real life, this status is only updated AFTER a fraud investigation.
    df['investigation_flag'] = df[TARGET_COLUMN].apply(lambda x: 1 if x == 1 else 0)
    
    print("\n[Scenario 1] INCORRECT Workflow (Target Leakage)")
    print("Feature 'investigation_flag' is included. This information is only available AFTER the fraud occurs.")
    
    X_leaky = df.drop(columns=[TARGET_COLUMN])
    y_leaky = df[TARGET_COLUMN]
    
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X_leaky, y_leaky, test_size=0.2, random_state=RANDOM_STATE, stratify=y_leaky
    )
    
    # We only use numerical columns for simplicity in this demo to avoid complex encoding
    # or we can just use the investigation_flag alone to prove the point
    leaky_model = RandomForestClassifier(random_state=RANDOM_STATE)
    leaky_model.fit(X_train_l[['investigation_flag']], y_train_l)
    
    y_pred_l = leaky_model.predict(X_test_l[['investigation_flag']])
    leaky_acc = accuracy_score(y_test_l, y_pred_l)
    leaky_f1 = f1_score(y_test_l, y_pred_l)
    
    print(f"Leaky Model Accuracy: {leaky_acc:.4f}")
    print(f"Leaky Model F1 Score: {leaky_f1:.4f}")
    print("Reason: The model 'cheated' by using a feature that is a proxy for the target.")

    print("\n[Scenario 2] CORRECT Workflow (No Leakage)")
    print("Removing 'investigation_flag' and using only valid predictors available at transaction time.")
    
    # Use valid features: amount, transaction_count, velocity
    valid_features = ['amount', 'transaction_count', 'velocity']
    X_honest = df[valid_features]
    y_honest = df[TARGET_COLUMN]
    
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_honest, y_honest, test_size=0.2, random_state=RANDOM_STATE, stratify=y_honest
    )
    
    honest_model = RandomForestClassifier(random_state=RANDOM_STATE)
    honest_model.fit(X_train_h, y_train_h)
    
    y_pred_h = honest_model.predict(X_test_h)
    honest_acc = accuracy_score(y_test_h, y_pred_h)
    honest_f1 = f1_score(y_test_h, y_pred_h)
    
    print(f"Honest Model Accuracy: {honest_acc:.4f}")
    print(f"Honest Model F1 Score: {honest_f1:.4f}")
    print("Reason: The model learns from actual patterns, leading to more realistic performance.")

    print("\n--- Summary ---")
    print(f"Performance Drop (Accuracy): {leaky_acc - honest_acc:.4f}")
    print("The first version was invalid because it used future information.")

if __name__ == "__main__":
    run_leakage_demo()
