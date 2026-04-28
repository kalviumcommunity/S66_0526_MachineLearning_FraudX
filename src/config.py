import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "fraud_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_fraud_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "is_fraud"

# Features to use
NUMERICAL_COLS = ["amount", "transaction_count", "velocity"]
CATEGORICAL_COLS = ["category", "location"]
