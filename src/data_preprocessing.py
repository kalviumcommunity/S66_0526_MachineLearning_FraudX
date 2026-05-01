"""
data_preprocessing.py

Responsible for:
- Handling missing values and basic data cleaning
- Splitting datasets into training and testing subsets
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, ALL_FEATURES

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle basic data cleaning like missing values.
    """
    return df.fillna(0)

def separate_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN, 
                             feature_columns: list = ALL_FEATURES) -> tuple:
    """
    Explicitly separate features and target with validation.
    
    Args:
        df: Input DataFrame.
        target_column: Name of the target variable.
        feature_columns: List of valid feature columns.
        
    Returns:
        tuple: (X, y)
    """
    # 1. Validation: Target exists
    assert target_column in df.columns, f"Target '{target_column}' not found in dataset."
    
    # 2. Validation: Target not in features (Prevent Leakage)
    assert target_column not in feature_columns, "Target variable leaked into feature list!"
    
    # 3. Separation
    X = df[feature_columns]
    y = df[target_column]
    
    # 4. Verification
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y

def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, 
               random_state: int = RANDOM_STATE) -> tuple:
    """
    Split the data into training and testing sets.
    """
    X, y = separate_features_target(df)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
