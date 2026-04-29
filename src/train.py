"""
train.py

Responsible for:
- Orchestrating the training process
- Calling the data loader and preprocessing utilities
- Fitting the model and saving artifacts
"""
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_data
from src.data_preprocessing import clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.evaluate import evaluate_model
from src.persistence import save_artifacts
from src.config import (RAW_DATA_PATH, MODEL_PATH, PIPELINE_PATH, 
                        RANDOM_STATE, CATEGORICAL_COLS, NUMERICAL_COLS)

def train_pipeline():
    """
    Complete training workflow from data loading to artifact persistence.
    """
    # 1. Load Data
    df_raw = load_data(RAW_DATA_PATH)
    
    # 2. Clean and Split
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    # 3. Feature Engineering (Fit only on train)
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # 4. Training
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)
    
    # 5. Evaluation
    metrics = evaluate_model(model, X_test_processed, y_test)
    print(f"Training Complete. Accuracy: {metrics['accuracy']:.4f}")
    
    # 6. Persistence
    save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
    print(f"Artifacts saved to {MODEL_PATH} and {PIPELINE_PATH}")

if __name__ == "__main__":
    train_pipeline()
