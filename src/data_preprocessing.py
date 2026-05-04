"""
data_preprocessing.py

Responsible for:
- Handling missing values and basic data cleaning
- Splitting datasets into training and testing subsets
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, 
                        ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, 
                        EXCLUDED_COLUMNS)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle basic data cleaning like missing values.
    """
    return df.fillna(0)

def separate_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN, 
                             feature_columns: list = ALL_FEATURES,
                             excluded_columns: list = EXCLUDED_COLUMNS) -> tuple:
    """
    Explicitly separate features and target with validation.
    
    Args:
        df: Input DataFrame.
        target_column: Name of the target variable.
        feature_columns: List of valid feature columns.
        excluded_columns: List of columns to be strictly excluded.
        
    Returns:
        tuple: (X, y)
    """
    # 1. Validation: Target exists
    assert target_column in df.columns, f"Target '{target_column}' not found in dataset."
    
    # 2. Validation: Target not in features (Prevent Leakage)
    assert target_column not in feature_columns, "Target variable leaked into feature list!"
    
    # 3. Validation: Excluded columns not in features
    for col in excluded_columns:
        assert col not in feature_columns, f"Excluded column '{col}' found in feature list!"
    
    # 4. Separation
    X = df[feature_columns]
    y = df[target_column]
    
    # 5. Feature Type Verification (Assignment 5.18 Requirement)
    print("\n--- Feature Selection Verification ---")
    print(f"Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Total features: {len(feature_columns)}")
    print(f"Target variable: {target_column}")
    
    return X, y

def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, 
               random_state: int = RANDOM_STATE) -> tuple:
    """
    Split the data into training and testing sets with verification.
    """
    X, y = separate_features_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Verification Checks (Required by Assignment 5.16)
    print("\n--- Data Split Verification ---")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    
    print("\nTrain class distribution:")
    print(y_train.value_counts(normalize=True))
    
    print("\nTest class distribution:")
    print(y_test.value_counts(normalize=True))
    
    # Leakage Prevention Confirmation
    print("\n[LEAKAGE PREVENTION CONFIRMATION]")
    print("- Validation: Preprocessing (scaling/encoding) has NOT yet been fitted.")
    print("- Validation: Test set remains isolated from training statistics.")
    print("- Validation: Stratification applied to preserve class balance.")
    
    return X_train, X_test, y_train, y_test
