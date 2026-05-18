# FraudX: Modular Machine Learning Project

FraudX is a professional machine learning system designed for detecting fraudulent transactions. It demonstrates clean project structure, function-based design, and production-ready engineering principles.

## 📁 Project Structure

```text
fraudX/
├── data/
│   ├── raw/                          # Original, immutable datasets
│   └── processed/                    # Cleaned and transformed datasets
├── docs/
│   └── BASELINE.md                   # Baseline + class-imbalance write-up
├── models/                           # Serialized artifacts
│   ├── fraud_model.pkl               # Trained RandomForestClassifier
│   ├── preprocessor.pkl              # Fitted ColumnTransformer
│   ├── baseline_most_frequent.pkl    # DummyClassifier baseline (majority class)
│   └── baseline_stratified.pkl       # DummyClassifier baseline (stratified sampling)
├── reports/                          # Output metrics, plots, evaluation logs
├── src/                              # Source code directory
│   ├── __init__.py
│   ├── config.py                     # Centralized configuration and paths
│   ├── data_loader.py                # CSV loading with validation
│   ├── data_preprocessing.py         # Cleaning + train-test split (no leakage)
│   ├── feature_engineering.py        # ColumnTransformer (scaler + encoder)
│   ├── baseline.py                   # DummyClassifier baselines (most_frequent + stratified)
│   ├── comparison.py                 # Side-by-side baseline vs main-model comparison
│   ├── train.py                      # Model training and artifact persistence
│   ├── evaluate.py                   # Performance evaluation (incl. per-class metrics)
│   ├── persistence.py                # Artifact saving and loading
│   ├── predict.py                    # Inference logic (transform-only, no refit)
│   ├── leakage_demo.py               # Target leakage demonstration
│   └── eda.py                        # Exploratory plots
├── main.py                           # Orchestration entry point
├── requirements.txt                  # Project dependencies
└── README.md                         # Documentation
```

## 🚀 Project Setup Instructions

This project requires **Python 3.9+**. Follow these steps to set up a reproducible environment.

### 1. Create a Virtual Environment
Isolate the project dependencies by creating a virtual environment:
```bash
python3 -m venv venv
```

### 2. Activate the Environment
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```
- **Windows**:
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Pinned Dependencies
Install the exact versions of the required ML libraries:
```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline
Execute the full workflow (ingestion, preprocessing, training, evaluation, and baseline comparison):
```bash
export PYTHONPATH=.
python3 main.py
```

To run just the baseline comparison (fits baselines + main model, prints per-class metrics, saves baseline artifacts):
```bash
export PYTHONPATH=.
python3 src/comparison.py
```

### 5. Verification
To verify the setup, you can check the installed versions:
```bash
pip list
```

### 6. Deactivate
Exit the environment when finished:
```bash
deactivate
```

## 🏗️ Engineering Principles

- **Modular Design**: Every stage of the ML lifecycle is isolated into its own module.
- **Function Contracts**: All functions use type hints and descriptive docstrings.
- **No Hidden State**: Configuration is centralized in `config.py` and passed explicitly.
- **Reproducibility**: Random seeds are controlled via `RANDOM_STATE` in configuration.
- **Persistence**: Both the model and the preprocessing pipeline are saved for consistent inference.

## 📂 Repository Structure Explanation

- **`data/`**: Separated into `raw/` for immutable ground-truth data and `processed/` for cleaned features. This ensures that the original data is never accidentally modified.
- **`src/`**: Contains the production-ready source code. Each module has a single responsibility (e.g., `train.py` only handles fitting).
- **`models/`**: Dedicated storage for serialized model and preprocessing artifacts, keeping them separate from source code.
- **`notebooks/`**: Reserved for exploration, visualization, and EDA. Production logic is strictly kept in `src/`.
- **`reports/`**: Stores evaluation outputs like metric logs and plots, facilitating experiment comparison.
- **`logs/`**: Tracks pipeline execution and experiment history to support reproducibility.

## 🔄 Data Flow Mapping

The project follows a unidirectional data flow to prevent leakage and ensure maintainability:

1. **Ingestion**: Raw data is loaded from `data/raw/` via `data_preprocessing.py`.
2. **Cleaning**: Initial cleaning (handling missing values) is performed, and data is split into train/test sets.
3. **Feature Engineering**: `feature_engineering.py` constructs a `ColumnTransformer` pipeline.
4. **Training**: `train.py` fits the model on the preprocessed training features.
5. **Evaluation**: `evaluate.py` assesses the model on the test set and outputs metrics to `reports/`.
6. **Persistence**: `persistence.py` saves the fitted model and pipeline to `models/`.
7. **Prediction**: `predict.py` loads artifacts from `models/` and transforms new data for inference.

## 🧠 Design Justification

### Separation of Raw and Processed Data
Raw data is treated as immutable. By keeping it separate from processed data, we ensure that any feature engineering step can be discarded or re-run without risking the loss of the original source of truth. This is critical for data auditing and reproducibility.

### Separation of Notebooks and Source Code
Notebooks are excellent for exploration but poor for version control and testing. By moving stable logic into modular Python files in `src/`, we make the codebase testable, reusable, and ready for deployment.

### Artifact Management outside Source
Models and pipelines are binaries that change every time we re-train. Keeping them in a dedicated `models/` folder prevents the `src/` directory from being bloated with non-code artifacts and allows for better versioning of model files.

### Centralized Configuration (`config.py`)
Hardcoding paths and hyperparameters across multiple files creates a maintenance nightmare. `config.py` acts as a single point of truth, making it easy to move data locations or update seeds without hunting through the entire codebase.

## 📊 Feature Type Definition

A disciplined approach to feature selection was used to categorize variables based on their conceptual meaning and prediction-time availability.

### 🎯 Target Variable
- **Column Name**: `is_fraud`
- **Type**: Binary Classification (Supervised)
- **Business Meaning**: Indicates whether a transaction is fraudulent (1) or legitimate (0). Detecting this helps the business prevent financial loss and secure customer trust.

### 🔢 Numerical Features
These features represent measurable quantities where arithmetic relationships carry significant meaning.

| Feature Name | Reason for Numerical Type | Scaling Strategy |
| :--- | :--- | :--- |
| `amount` | Represents currency magnitude; continuous value. | `StandardScaler` (Z-score normalization) |
| `transaction_count` | Discrete integer count of recent activity. | `StandardScaler` |
| `velocity` | Calculated frequency ratio; continuous value. | `StandardScaler` |

- **Scaling Justification**: All numerical features are scaled to a mean of 0 and variance of 1 to ensure that features with larger ranges (like `amount`) do not dominate the distance-based calculations of the model.

### 🗂️ Categorical Features
These features represent discrete labels or groups with no inherent mathematical magnitude.

| Feature Name | Category Type | Reason for Categorical Type | Encoding Strategy |
| :--- | :--- | :--- | :--- |
| `category` | Nominal | Represents merchant types (travel, food, etc.). No natural order. | One-Hot Encoding (drop first) |
| `location` | Nominal | Represents geographic context (domestic/intl). No natural order. | One-Hot Encoding (drop first) |

- **Encoding Justification**: One-Hot Encoding is used to transform labels into binary flags, allowing the model to interpret categories without assuming any artificial rank or order.

### 🚫 Excluded Columns
- **Identifiers**: No `CustomerID` or `TransactionID` are used, as they lead to overfitting (memorizing specific rows).
- **Post-Outcome Variables**: Any data derived after the fraud decision (e.g., `investigation_notes`) is excluded to prevent target leakage.

### 🛡️ Edge Case Handling
- **Binary Columns**: While `is_fraud` is the target, any binary features (like `location` once encoded) are treated as categorical flags.
- **High-Cardinality**: Features like `ZipCode` or `Address` are not present; if they were, they would be handled via target encoding or clustering to prevent dimensionality explosion.
- **Timestamps**: No raw timestamps are used; any temporal information is pre-calculated as `velocity` before model ingestion.

### 🧪 Validation Discipline
- **Explicit Selection**: Features are manually defined in `config.py`, never auto-detected.
- **Leakage Assertion**: The code enforces `assert TARGET_COLUMN not in ALL_FEATURES`.
- **Exclusion Check**: Excluded columns are strictly validated to ensure they never enter the training pipeline.

## 🔍 Feature Distribution Analysis

Before modeling, a systematic inspection of feature distributions was performed to identify skewness, outliers, and class imbalances.

### 📈 Numerical Feature Behavior

| Feature Name | Skewness | Observations | Recommended Transformation |
| :--- | :--- | :--- | :--- |
| `amount` | 1.87 (High) | Strongly right-skewed with a long tail. Most transactions are small, with a few very large ones. | **Log Transformation** or Robust Scaling recommended to stabilize variance. |
| `transaction_count` | 0.02 (Low) | Near-perfect symmetric distribution. Well-behaved across its range (1-49). | Standard Scaling is sufficient. |
| `velocity` | -0.003 (Low) | Near-perfect symmetric distribution. Values are evenly spread between 0 and 10. | Standard Scaling is sufficient. |

### 📊 Categorical Feature Inspection

- **`category`**: Distribution is balanced across all 4 levels (*retail, travel, food, online*), each representing ~25% of the data. No rare levels detected.
- **`location`**: Almost perfectly balanced between *domestic* and *international* (~50% each). No inconsistent labeling (e.g., case typos) was found.

### 🎯 Target-Based Comparison Insights

Initial boxplot analysis across target classes (`is_fraud`) suggests:
- **Predictive Signal**: `amount` shows a slightly different distribution for fraud cases, indicating it will be a strong predictor.
- **Weak Signal**: `velocity` and `transaction_count` show significant overlap across classes, suggesting they may provide weaker individual signal but useful interaction effects.

### 🛡️ Inspection Discipline
- **No Data Leakage**: All inspection and visualization were performed on the raw training data.
- **No Test Set Contamination**: Preprocessing decisions (like log transformation for `amount`) are identified here but will be fitted *only* on the training split during the pipeline execution.

## 📊 Data Splitting Strategy

A rigorous data splitting protocol is implemented to ensure that the model evaluation is honest, reproducible, and reflective of real-world performance.

### ⚙️ Split Configuration
- **Split Ratio**: 80% Training | 20% Testing
- **Random State**: `42` (Ensures reproducibility across environments)
- **Stratification**: **Applied** (`stratify=y`)
- **Strategy Type**: Random Stratified Split (Appropriate for non-temporal classification)

### ⚖️ Strategy Justification
1. **Sufficient Learning Capacity**: The 80% training allocation provides enough examples for the `RandomForestClassifier` to identify the non-linear boundaries between legitimate and fraudulent transactions.
2. **Statistical Significance**: The 20% test set is large enough to provide stable performance metrics, ensuring that our accuracy and recall scores are not due to random chance.
3. **Preserving Class Balance**: Fraud datasets are typically imbalanced. By using **Stratified Splitting**, we guarantee that the 10% fraud rate in the original data is preserved in both the training and testing sets, preventing evaluation bias.
4. **Real-World Simulation**: The test set acts as a proxy for unseen future data. By isolating it before any preprocessing, we simulate a production scenario where the model must handle data it has never encountered.

### 🚫 Leakage Prevention Measures
- **Split-First Policy**: The `train_test_split` is executed **before** any feature engineering (scaling, encoding, or imputation).
- **Fitting Discipline**: Preprocessing pipelines are `fit()` only on the training set and merely `transform()` the test set. This prevents "future information" (like the global mean or variance) from leaking into the training process.
- **Validation Prints**: The pipeline explicitly prints shapes and class distributions at runtime to verify the integrity of the split.

---
*This strategy ensures that when we say the model is 95% accurate, it is a measurement of learning, not memorization.*

## 🛡️ Data Leakage Demonstration

As part of the engineering discipline, we conducted a controlled experiment to demonstrate the impact of **Target Leakage** and the importance of guarding the prediction boundary.

### 🧪 Experiment Setup
We compared two versions of the model:
1. **Leaky Version**: Included a feature (`investigation_flag`) that is only available *after* a fraud investigation is complete.
2. **Honest Version**: Used only valid predictors available at the *moment of transaction* (`amount`, `velocity`, `transaction_count`).

### 📊 Performance Comparison
| Metric | Leaky Model (Invalid) | Honest Model (Valid) | Impact of Leakage |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 100% | 91.0% | +9.0% (Artificial) |
| **F1-Score** | 1.00 | 0.00 | +1.00 (Artificial) |

### 🔍 Analysis & Reflection
- **Why the Leaky Model Failed**: The model achieved perfect scores not because it learned to detect fraud, but because it "cheated" by looking at the outcome (the investigation flag). In a real-world deployment, this flag would be missing for all new transactions, rendering the model useless.
- **The Prediction Moment Test**: We verified that `investigation_flag` fails the "Prediction Moment Test" because it does not exist at the exact second a transaction is processed.
- **Discipline**: By removing target-derived features and splitting data before any preprocessing, we ensure that our evaluation metrics reflect actual predictive power rather than hindsight bias.

---
*Run the demonstration yourself using:* `python3 src/leakage_demo.py`

## ⚖️ Numerical Feature Scaling

Numerical features in the FraudX dataset exist on different scales (e.g., `amount` can be in the hundreds, while `velocity` is a small ratio). To ensure stable optimization and consistent feature contribution, we implement standardization using `StandardScaler`.

### 🛠️ Implementation Details
- **Features Scaled**: `amount`, `transaction_count`, `velocity`.
- **Method**: `StandardScaler` (Standardization).
- **Transformation Formula**: $z = (x - \mu) / \sigma$ (resulting in Mean=0, Std=1).

### 🧠 Strategic Justification
1. **Model Choice**: We are using a `RandomForestClassifier`. While tree-based models are scale-invariant, we apply scaling to:
    - Maintain numerical stability in the preprocessing pipeline.
    - Ensure compatibility if we decide to switch to distance-based models (like SVM or kNN) or linear models (like Logistic Regression) in the future.
    - Provide a standardized range for feature importance comparisons.
2. **Leakage Prevention**:
    - **Split-First Policy**: Scaling is applied *only* after the train-test split.
    - **Fit-Transform Discipline**: The `StandardScaler` is `fit()` exclusively on the training set. The test set is transformed using the parameters (mean and variance) learned from the training data, ensuring no information from the test set leaks into the training process.
3. **Artifact Persistence**: The fitted scaler is part of the `ColumnTransformer` saved in `models/preprocessor.pkl`. This ensures that during inference, new data is scaled using the exact same parameters used during training.

### 🚫 Categorical Handling
Categorical features (`category`, `location`) are **not scaled**. They are processed via `OneHotEncoder`, which transforms them into binary flags (0 or 1). Scaling these binary flags would distort their logical meaning.

### 📊 Verification
After scaling, the training features exhibit a mean of approximately 0 and a standard deviation of 1, confirming a successful transformation.

## 🧪 Baseline Comparison and Class-Imbalance Evaluation

The FraudX dataset is heavily imbalanced (~91% non-fraud / ~9% fraud). On data like this, accuracy is structurally misleading: a model that always predicts non-fraud achieves ~91% accuracy without learning anything. To make this trap visible at runtime, the project ships a baseline comparison harness in [`src/comparison.py`](src/comparison.py) that fits two `sklearn.dummy.DummyClassifier` baselines on the same training data as the main `RandomForestClassifier` and evaluates everything on the same held-out test set with per-class metrics.

The long-form rationale, leakage discipline, scenario-question answers, and worked numbers live in [`docs/BASELINE.md`](docs/BASELINE.md). The short version follows.

### 🛠️ What gets compared

| Model                     | Strategy                                                       |
| ------------------------- | -------------------------------------------------------------- |
| `baseline_most_frequent`  | `DummyClassifier(strategy="most_frequent")` — predicts class 0 always. The canonical "is my model actually meaningful?" lower bound. |
| `baseline_stratified`     | `DummyClassifier(strategy="stratified", random_state=42)` — samples predictions from the training class prior. A non-trivial chance baseline. |
| `RandomForestClassifier`  | The trained main model (same hyperparameters as `src/train.py`). |

All three are fit on the **same** `X_train` / `y_train` and evaluated on the **same** `X_test` / `y_test`, with identical metrics. Identical-support assertions in [`src/comparison.py`](src/comparison.py) guarantee the test set wasn't accidentally modified between runs.

### 📐 Metrics

Per the assignment's "Important Guidelines" ("Report per-class metrics if working on imbalanced classification"), [`src/evaluate.py::evaluate_detailed`](src/evaluate.py) returns:
- Overall accuracy and **balanced accuracy** (accuracy averaged over the two classes — does NOT collapse to majority-class behaviour).
- Per-class precision / recall / F1 / support for class 0 (non-fraud) AND class 1 (fraud).
- Raw confusion matrix.

### 📊 Headline result (real numbers from this repo)

Test set: 200 samples (182 class 0 / 18 class 1).

| Model                       | Accuracy | Balanced Acc. | F1 (class 1, fraud) | Recall (class 1) |
| --------------------------- | -------: | ------------: | ------------------: | ---------------: |
| `baseline_most_frequent`    |   91.0 % |        50.0 % |               0.0 % |            0.0 % |
| `baseline_stratified`       |   83.0 % |        45.6 % |               0.0 % |            0.0 % |
| `RandomForestClassifier`    |   91.0 % |        50.0 % |               0.0 % |            0.0 % |

**Reading**: the trained model and the `most_frequent` baseline get the same 91% accuracy because they both predict non-fraud for every sample. The trained model does **not** yet meaningfully beat the baseline on the minority-class metrics that matter for fraud detection. This is exactly the failure mode this module is designed to surface; future iterations (`class_weight="balanced"`, SMOTE, threshold tuning) should be measured against this same baseline harness.

### 🚫 Leakage Prevention

The same discipline applied to the normalisation module applies here:

- `train_test_split` runs **before** any classifier is constructed.
- Baselines are `.fit()` on `X_train` / `y_train` only; never on the full dataset.
- Identical `X_test` / `y_test` is used for the baselines and the main model (asserted at runtime).
- Heuristic rules (here, the trained majority-class frequency) are derived from training data exclusively.

### 📦 Persistence

Two fitted baseline artifacts are saved by `src/comparison.py`:
- `models/baseline_most_frequent.pkl`
- `models/baseline_stratified.pkl`

Loaded for inspection via:
```python
from src.baseline import load_baseline
clf = load_baseline("most_frequent")
```

### 🏃 How to run

```bash
export PYTHONPATH=.
python3 src/comparison.py     # just the comparison
# OR
python3 main.py               # full pipeline (Phase 3 runs the comparison)
```
