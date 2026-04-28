# FraudGuard: Modular ML Pipeline

FraudGuard is a professional, modular machine learning project designed to detect fraudulent transactions. This project demonstrates clean project structuring, function-based design, and disciplined import practices in a machine learning workflow.

## Project Structure

```text
project_root/
├── data/
│   ├── raw/            # Original, immutable datasets
│   └── processed/      # Cleaned and transformed datasets
├── models/             # Serialized model artifacts (.pkl files)
├── reports/            # Output metrics and evaluation logs
├── src/                # Source code directory
│   ├── __init__.py
│   ├── config.py       # Centralized configuration and paths
│   ├── data_loader.py  # Data ingestion logic
│   ├── preprocessing.py# Data cleaning and splitting
│   ├── model.py        # Model training and persistence
│   ├── evaluate.py     # Evaluation metrics and reporting
│   └── main.py         # Orchestration script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. **Clone the repository** (or navigate to this directory).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python src/main.py
   ```

## Key Engineering Principles Applied

- **Single Responsibility Principle**: Each module handles a specific part of the ML lifecycle (ingestion, preprocessing, training, evaluation).
- **Function-Based Design**: Logic is encapsulated within functions that have clear input/output contracts.
- **Centralized Configuration**: All file paths and hyperparameters are managed in `src/config.py`.
- **Reproducibility**: Random states are fixed throughout the pipeline to ensure consistent results.
- **Separation of Concerns**: The orchestration layer (`main.py`) is decoupled from the core logic modules.

## Scenario-Based Reflection: Circular Imports

If a teammate adds preprocessing logic inside `model.py` and imports `model.py` inside `preprocessing.py`, it creates a circular import error. This happens because Python cannot resolve the dependency chain where Module A depends on Module B and vice versa.

To fix this, we ensure **Dependency Direction**:
1. Keep preprocessing logic strictly in `preprocessing.py`.
2. `model.py` should only receive processed data; it should not call preprocessing functions directly.
3. The orchestration layer (`main.py`) handles the flow by passing data from the preprocessing module to the model module.
