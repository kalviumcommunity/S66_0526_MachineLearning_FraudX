"""
predict.py

Responsible for:
- Loading saved artifacts (model and pipeline)
- Applying transformations to new data using transform() (NOT fit_transform)
- Generating predictions without refitting any components
"""
import pandas as pd
import sys
from src.persistence import load_artifacts
from src.config import MODEL_PATH, PIPELINE_PATH

def predict(new_data: pd.DataFrame) -> pd.Series:
    """
    Generate predictions on new data using saved artifacts.
    
    Args:
        new_data: Raw input data (DataFrame).
        
    Returns:
        pd.Series: Model predictions.
    """
    # Load artifacts
    model, pipeline = load_artifacts(MODEL_PATH, PIPELINE_PATH)
    
    # Transform raw data using the fitted pipeline (strictly transform, no fit)
    processed_data = pipeline.transform(new_data)
    
    # Generate predictions
    predictions = model.predict(processed_data)
    
    return predictions

if __name__ == "__main__":
    # Example usage: Generate a dummy prediction for verification
    # In a real scenario, this would load from a file or API input
    dummy_data = pd.DataFrame([{
        "amount": 500.0,
        "transaction_count": 5,
        "velocity": 2.5,
        "category": "retail",
        "location": "domestic"
    }])
    
    results = predict(dummy_data)
    print(f"Prediction for dummy data: {results}")
