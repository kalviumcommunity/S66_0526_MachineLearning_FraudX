"""
feature_engineering.py

Responsible for:
- Creating transformation pipelines for numerical and categorical features
- Handling imputation, scaling, and encoding consistently
- Ensuring preprocessing can be saved and reused for inference
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Construct an sklearn ColumnTransformer for feature encoding and scaling.
    Following professional ML structure with imputation, scaling, and encoding.
    
    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
        
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    # Numerical pipeline: Impute missing with median and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing with most frequent and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor
