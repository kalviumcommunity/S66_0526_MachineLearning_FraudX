import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN

def load_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw transaction data from a CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        # For demonstration purposes, generate synthetic data if file is missing
        generate_synthetic_data(filepath)
        
    return pd.read_csv(filepath)

def generate_synthetic_data(filepath: str, n_samples: int = 1000) -> None:
    """
    Helper function to create a synthetic fraud dataset if none exists.
    
    Args:
        filepath: Path to save the synthetic CSV.
        n_samples: Number of rows to generate.
    """
    np.random.seed(RANDOM_STATE)
    data = {
        "amount": np.random.exponential(scale=100, size=n_samples),
        "transaction_count": np.random.randint(1, 50, size=n_samples),
        "velocity": np.random.uniform(0, 10, size=n_samples),
        "category": np.random.choice(["retail", "online", "food", "travel"], size=n_samples),
        "location": np.random.choice(["domestic", "international"], size=n_samples),
        "is_fraud": np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle basic data cleaning like missing values.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Simple imputation: fill numerical with median, categorical with mode
    # For simplicity in this assignment, we use fillna(0) as in the previous version
    return df.fillna(0)

def split_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN, 
               test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> tuple:
    """
    Split the data into training and testing sets.
    
    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target variable.
        test_size: Proportion of the test set.
        random_state: Seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
