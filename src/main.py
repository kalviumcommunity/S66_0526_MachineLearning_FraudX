from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts
from src.config import (RAW_DATA_PATH, MODEL_PATH, PIPELINE_PATH, 
                        TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, 
                        CATEGORICAL_COLS, NUMERICAL_COLS)

def main():
    print("--- Starting Modular FraudX ML Pipeline ---\n")
    
    # 1. Ingestion
    print("Step 1: Loading raw data...")
    df_raw = load_data(RAW_DATA_PATH)
    print(f"Loaded {len(df_raw)} samples.")
    
    # 2. Cleaning
    print("Step 2: Cleaning data...")
    df_clean = clean_data(df_raw)
    
    # 3. Splitting
    print("Step 3: Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(
        df_clean, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )
    
    # 4. Feature Engineering (Pipeline construction)
    print("Step 4: Building and fitting preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # 5. Training
    print("Step 5: Training model...")
    model = train_model(X_train_processed, y_train, RANDOM_STATE)
    
    # 6. Evaluation
    print("Step 6: Evaluating performance...")
    metrics = evaluate_model(model, X_test_processed, y_test)
    
    # Display results (orchestration layer responsibility)
    print("\nModel Metrics:")
    for name, value in metrics.items():
        print(f" - {name.capitalize()}: {value:.4f}")
    
    # 7. Persistence
    print(f"\nStep 7: Saving artifacts to models/ directory...")
    save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
    
    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
