import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

def analyze_scenarios():
    scenarios = [
        {
            "id": 1,
            "title": "Fintech Fraud Detection",
            "type": "Binary Classification",
            "target": "Fraudulent (Yes/No)",
            "metrics": "Precision, Recall, F1-Score, ROC-AUC",
            "justification": "Target is a discrete category (two mutually exclusive classes). Imbalance requires F1 and Recall over Accuracy."
        },
        {
            "id": 2,
            "title": "Real Estate Price Estimation",
            "type": "Regression",
            "target": "Sale Price (Continuous numerical value)",
            "metrics": "MAE, RMSE, R²",
            "justification": "Target is a continuous dollar amount. Errors should be measured in magnitude."
        },
        {
            "id": 3,
            "title": "Movie Genre Tagging",
            "type": "Multi-Label Classification",
            "target": "Set of applicable genres",
            "metrics": "Hamming Loss, Subset Accuracy, Micro/Macro F1",
            "justification": "Classes are not mutually exclusive. A movie can have multiple genres simultaneously."
        },
        {
            "id": 4,
            "title": "Retail Units Sold Prediction",
            "type": "Count Regression",
            "target": "Number of units sold (Non-negative integer)",
            "metrics": "MAE, RMSE, Poisson Deviance",
            "justification": "Target is a discrete count value, representing magnitude."
        },
        {
            "id": 5,
            "title": "Hospital Disease Categorization",
            "type": "Multi-Class Classification",
            "target": "Disease category (Viral, Bacterial, Autoimmune)",
            "metrics": "Macro F1, Confusion Matrix",
            "justification": "Target is a discrete category from three mutually exclusive options."
        }
    ]

    print("--- Scenario Analysis ---")
    for s in scenarios:
        print(f"\nScenario {s['id']}: {s['title']}")
        print(f"Problem Type: {s['type']}")
        print(f"Target Variable: {s['target']}")
        print(f"Metric Selection: {s['metrics']}")
        print(f"Justification: {s['justification']}")
    print("\n" + "="*50 + "\n")


def demonstrate_classification():
    print("--- Classification Demonstration ---")
    # Generate synthetic binary classification data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.9, 0.1], random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train simple model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate with correct metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("Model: Logistic Regression")
    print(f"Accuracy:  {acc:.4f} (Misleading due to class imbalance)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\n" + "="*50 + "\n")


def demonstrate_regression():
    print("--- Regression Demonstration ---")
    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=15.0, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train simple model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate with correct metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Model: Linear Regression")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    analyze_scenarios()
    demonstrate_classification()
    demonstrate_regression()
