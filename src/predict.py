import pandas as pd
from src.persistence import load_artifacts

def predict(new_data: pd.DataFrame, model, pipeline) -> pd.Series:
    """
    Generate predictions on new data using saved artifacts.
    
    Args:
        new_data: Raw input data (DataFrame).
        model: Loaded model object.
        pipeline: Loaded preprocessing pipeline.
        
    Returns:
        pd.Series: Model predictions.
    """
    # Transform raw data using the fitted pipeline
    processed_data = pipeline.transform(new_data)
    
    # Generate predictions
    predictions = model.predict(processed_data)
    
    return predictions
