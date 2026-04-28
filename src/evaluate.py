from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the model performance using multiple metrics.
    """
    predictions = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions)
    }
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
        
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))
    
    return metrics
