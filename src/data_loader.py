"""
data_loader.py

Responsible for:
- Loading raw transaction data from storage
- Handling basic file-level validations and errors
- Returning a clean DataFrame for the pipeline to consume
"""
import pandas as pd
import os
from src.config import RAW_DATA_PATH

def load_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw transaction data from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the dataset is empty.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError(f"Loaded dataset from {filepath} is empty.")
        
    return df
