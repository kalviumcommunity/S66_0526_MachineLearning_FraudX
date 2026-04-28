# FraudX: Modular Machine Learning Project

FraudX is a professional machine learning system designed for detecting fraudulent transactions. It demonstrates clean project structure, function-based design, and production-ready engineering principles.

## 📁 Project Structure

```text
fraudX/
├── data/
│   ├── raw/            # Original, immutable datasets
│   └── processed/      # Cleaned and transformed datasets
├── models/             # Serialized artifacts (model and preprocessor)
├── reports/            # Output metrics and evaluation logs
├── src/                # Source code directory
│   ├── __init__.py
│   ├── config.py       # Centralized configuration and paths
│   ├── data_preprocessing.py # Loading, cleaning, and splitting
│   ├── feature_engineering.py # Encoding and scaling pipelines
│   ├── train.py        # Model training logic
│   ├── evaluate.py     # Performance evaluation
│   ├── persistence.py  # Artifact saving and loading
│   ├── predict.py      # Inference logic
│   └── main.py         # Orchestration script
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline**:
   ```bash
   export PYTHONPATH=.
   python src/main.py
   ```

## 🏗️ Engineering Principles

- **Modular Design**: Every stage of the ML lifecycle is isolated into its own module.
- **Function Contracts**: All functions use type hints and descriptive docstrings.
- **No Hidden State**: Configuration is centralized in `config.py` and passed explicitly.
- **Reproducibility**: Random seeds are controlled via `RANDOM_STATE` in configuration.
- **Persistence**: Both the model and the preprocessing pipeline are saved for consistent inference.
