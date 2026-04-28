from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute evaluation metrics on test data.
    
    Args:
        model: Trained model artifact.
        X_test: Processed test features.
        y_test: Test labels.
        
    Returns:
        dict: Dictionary of performance metrics.
    """
    predictions = model.predict(X_test)
    
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions)
    }
