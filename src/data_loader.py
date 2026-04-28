import pandas as pd
import numpy as np
import os
from src.config import RAW_DATA_PATH, RANDOM_STATE

def load_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw transaction data from a CSV file.
    If the file doesn't exist, generate synthetic data for demonstration.
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Generating synthetic data...")
        generate_synthetic_data(filepath)
        
    return pd.read_csv(filepath)

def generate_synthetic_data(filepath: str, n_samples: int = 1000) -> None:
    """
    Helper function to create a synthetic fraud dataset.
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
    print(f"Synthetic data saved to {filepath}")
