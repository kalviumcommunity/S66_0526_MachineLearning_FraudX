"""
feature_engineering.py

Responsible for:
- Creating transformation pipelines for numerical and categorical features
- Handling imputation, scaling (MinMaxScaler), and encoding consistently
- Ensuring preprocessing can be saved and reused for inference

Scaling Choice: MinMaxScaler (Assignment 5.18)
- Bounds numerical features to [0, 1].
- Chosen because the project is designed to be model-agnostic; future
  experimentation with scale-sensitive models (kNN, SVM, Logistic Regression,
  Neural Networks) benefits from features sharing a common bounded range.
- Preserves the shape of the original distribution (unlike z-score scaling
  which assumes Gaussian-like data), which is appropriate given that
  `transaction_count` and `velocity` are nearly symmetric and `amount` is
  right-skewed but cleanly bounded after split-aware fitting.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Construct an sklearn ColumnTransformer for feature encoding and scaling.

    Numerical features are imputed (median) and scaled to [0, 1] with
    MinMaxScaler. Categorical features are imputed (most frequent) and
    one-hot encoded. Categorical features are explicitly LEFT UNSCALED.

    Args:
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.

    Returns:
        ColumnTransformer: Preprocessing pipeline ready to be fit on
        TRAINING DATA ONLY (no leakage).
    """
    # Numerical pipeline: impute with median, then scale to [0, 1].
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ])

    # Categorical pipeline: impute with most frequent, then one-hot encode.
    # No scaling — one-hot encoded binary flags must not be rescaled.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine: numerical features get scaled, categorical features stay as flags.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor
