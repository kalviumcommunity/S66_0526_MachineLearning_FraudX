"""
train.py

Responsible for:
- Orchestrating the training process
- Calling the data loader and preprocessing utilities
- Fitting the model and saving artifacts (including the standalone
  MinMaxScaler artifact required by Assignment 5.18)
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.config import (CATEGORICAL_FEATURES, MINMAX_SCALER_PATH, MODEL_PATH,
                        NUMERICAL_FEATURES, PIPELINE_PATH, RANDOM_STATE,
                        RAW_DATA_PATH)
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data
from src.evaluate import evaluate_model
from src.feature_engineering import build_preprocessing_pipeline
from src.persistence import save_artifacts


def _verify_minmax_ranges(pipeline, X_train, X_test) -> None:
    """
    Confirm that after the fitted ColumnTransformer is applied, the numerical
    columns lie in the expected [0, 1] interval on the training set.

    The fitted MinMaxScaler is the second step in the 'num' Pipeline of the
    ColumnTransformer, so we reach into the structure to extract it for an
    auditable, per-column report. This step is explicitly required by the
    assignment ("Minimum values in training data ≈ 0 / Maximum values in
    training data ≈ 1").
    """
    num_pipeline = pipeline.named_transformers_["num"]
    scaler = num_pipeline.named_steps["scaler"]

    # Apply the imputer+scaler to the raw numerical columns so we read the
    # scaled values in their original column order.
    train_scaled = num_pipeline.transform(X_train[NUMERICAL_FEATURES])
    test_scaled = num_pipeline.transform(X_test[NUMERICAL_FEATURES])

    print("\n--- [Verification] MinMaxScaler ranges ---")
    print("Training set (must be ≈ [0, 1]):")
    for i, col in enumerate(NUMERICAL_FEATURES):
        mn, mx = train_scaled[:, i].min(), train_scaled[:, i].max()
        print(f"  {col:<20s}  min={mn:.6f}  max={mx:.6f}")
        assert np.isclose(mn, 0.0, atol=1e-9), f"Training min for {col} is {mn}, expected ~0"
        assert np.isclose(mx, 1.0, atol=1e-9), f"Training max for {col} is {mx}, expected ~1"

    print("Test set (may exceed [0, 1] if extremer than training extremes — expected):")
    for i, col in enumerate(NUMERICAL_FEATURES):
        mn, mx = test_scaled[:, i].min(), test_scaled[:, i].max()
        print(f"  {col:<20s}  min={mn:.6f}  max={mx:.6f}")

    print(f"Fitted scaler params — data_min_: {scaler.data_min_}, data_max_: {scaler.data_max_}")
    print("[VERIFIED] All training numerical columns scaled to ≈ [0, 1].")


def train_pipeline():
    """
    Complete training workflow from data loading to artifact persistence.

    Order of operations (leakage-safe):
        1. Load raw data.
        2. Clean.
        3. Train-test split (BEFORE any scaling).
        4. Fit ColumnTransformer (MinMaxScaler + OneHotEncoder) on X_train.
        5. Transform X_train and X_test.
        6. Verify min ≈ 0, max ≈ 1 on training scaled values.
        7. Fit RandomForestClassifier on processed training data.
        8. Evaluate on test set.
        9. Persist model, full preprocessing pipeline, AND standalone
           MinMaxScaler (so the assignment's graders can load and inspect
           the scaler in isolation).
    """
    # 1. Load Data
    df_raw = load_data(RAW_DATA_PATH)

    # 2. Clean and Split (split happens BEFORE preprocessing)
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # 3. Feature Engineering — fit only on train, transform both
    pipeline = build_preprocessing_pipeline(CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # 4. Verification step — required by the assignment
    _verify_minmax_ranges(pipeline, X_train, X_test)

    # 5. Training
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)

    # 6. Evaluation
    metrics = evaluate_model(model, X_test_processed, y_test)
    print(f"\nTraining Complete. Accuracy: {metrics['accuracy']:.4f}  "
          f"F1: {metrics['f1']:.4f}  Precision: {metrics['precision']:.4f}  "
          f"Recall: {metrics['recall']:.4f}")

    # 7. Persistence — model + full preprocessing pipeline
    save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
    print(f"Model artifact -> {MODEL_PATH}")
    print(f"Pipeline artifact -> {PIPELINE_PATH}")

    # 8. Persistence — standalone MinMaxScaler (Assignment 5.18 requirement)
    minmax_scaler = pipeline.named_transformers_["num"].named_steps["scaler"]
    joblib.dump(minmax_scaler, MINMAX_SCALER_PATH)
    print(f"Standalone MinMaxScaler -> {MINMAX_SCALER_PATH}")


if __name__ == "__main__":
    train_pipeline()
