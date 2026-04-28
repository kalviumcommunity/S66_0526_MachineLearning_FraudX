from src.data_loader import load_data
from src.preprocessing import preprocess_data, split_data
from src.model import train_model, save_model
from src.evaluate import evaluate_model
from src.config import RAW_DATA_PATH, MODEL_PATH

def main():
    print("--- Starting FraudGuard ML Pipeline ---\n")
    
    # 1. Load Data
    print("Step 1: Loading Data...")
    df = load_data(RAW_DATA_PATH)
    print(f"Loaded {len(df)} samples.")
    
    # 2. Preprocess Data
    print("\nStep 2: Preprocessing Data...")
    df_processed = preprocess_data(df)
    
    # 3. Split Data
    print("Step 3: Splitting into Train/Test sets...")
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # 4. Train Model
    print("\nStep 4: Training Model...")
    model = train_model(X_train, y_train)
    
    # 5. Evaluate Model
    print("\nStep 5: Evaluating Model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # 6. Save Model
    print("\nStep 6: Saving Model...")
    save_model(model, MODEL_PATH)
    
    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
