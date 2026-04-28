import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic preprocessing: handle missing values and encode categories.
    """
    # Simple filling of missing values
    df = df.fillna(0)
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=["category", "location"], drop_first=True)
    
    return df

def split_data(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataframe into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
