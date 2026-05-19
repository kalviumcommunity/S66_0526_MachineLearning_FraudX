"""
config.py

Responsible for:
- Centralizing all constants and configuration variables
- Storing file paths for raw and processed data
- Managing model and pipeline serialization paths
- Defining hyperparameters and random seeds
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "fraud_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_fraud_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
# Standalone fitted MinMaxScaler artifact (Assignment 5.18 requirement)
MINMAX_SCALER_PATH = os.path.join(BASE_DIR, "models", "minmax_scaler.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target Definition
TARGET_COLUMN = "is_fraud"

# Numerical Features (Explicitly Declared — Assignment 5.18 Requirement)
NUMERICAL_FEATURES = ["amount", "transaction_count", "velocity"]

# Categorical Features
CATEGORICAL_FEATURES = ["category", "location"]

# Excluded Columns
EXCLUDED_COLUMNS = []  # No identifiers like CustomerID in this small dataset

# Derived
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Validation: Ensure target is never in the feature list
assert TARGET_COLUMN not in ALL_FEATURES, "Configuration Error: Target variable leaked into ALL_FEATURES list!"
