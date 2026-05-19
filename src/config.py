"""
config.py

Responsible for:
- Centralizing all constants and configuration variables
- Storing file paths for raw and processed data
- Managing model and pipeline serialization paths
- Defining hyperparameters and random seeds

This is the consolidated configuration for the full FraudX system —
every prior module's path constants are gathered here.
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "fraud_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_fraud_data.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Model + preprocessor artifacts (existing project artifacts)
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# MinMaxScaler artifact (PR #15)
MINMAX_SCALER_PATH = os.path.join(BASE_DIR, "models", "minmax_scaler.pkl")

# Baseline model artifacts (PR #17)
BASELINE_MOST_FREQUENT_PATH = os.path.join(BASE_DIR, "models", "baseline_most_frequent.pkl")
BASELINE_STRATIFIED_PATH = os.path.join(BASE_DIR, "models", "baseline_stratified.pkl")

# Hyperparameter-tuned model artifact (PR #18)
TUNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "tuned_fraud_model.pkl")

# Final selection capstone artifact (PR #25)
FINAL_SELECTED_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_selected_model.pkl")

# Production-deployment artifacts (PRs #26, #27 + final system)
PERSISTED_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "persisted_pipeline.pkl")
DEPLOYMENT_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "pipeline.joblib")
DEPLOYMENT_METADATA_PATH = os.path.join(BASE_DIR, "models", "pipeline_metadata.json")

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target Definition
TARGET_COLUMN = "is_fraud"

# Numerical Features
NUMERICAL_FEATURES = ["amount", "transaction_count", "velocity"]

# Categorical Features
CATEGORICAL_FEATURES = ["category", "location"]

# Excluded Columns
EXCLUDED_COLUMNS = []  # No identifiers like CustomerID in this small dataset

# Derived
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Validation: Ensure target is never in the feature list
assert TARGET_COLUMN not in ALL_FEATURES, "Configuration Error: Target variable leaked into ALL_FEATURES list!"

# Feature-value ranges for the Streamlit app's input widgets (final system).
# Derived from the training-set distributions; used by app.py to constrain
# the number_input widgets to realistic ranges.
FEATURE_VALUE_RANGES = {
    "amount":            {"min": 0.0,  "max": 1000.0, "default": 100.0,  "step": 1.0},
    "transaction_count": {"min": 1,    "max": 100,    "default": 5,      "step": 1},
    "velocity":          {"min": 0.0,  "max": 10.0,   "default": 1.0,    "step": 0.1},
}

# Allowed categorical values (also used by Streamlit selectbox widgets).
CATEGORY_OPTIONS = ["food", "online", "retail", "travel"]
LOCATION_OPTIONS = ["domestic", "international"]
