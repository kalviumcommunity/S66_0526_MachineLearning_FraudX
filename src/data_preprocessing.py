"""
data_preprocessing.py

Responsible for:
- Handling missing values and basic data cleaning
- Splitting datasets into training and testing subsets
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle basic data cleaning like missing values.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Simple imputation for raw cleaning
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
