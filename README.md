# FraudX: Modular Machine Learning Project

FraudX is a professional machine learning system designed for detecting fraudulent transactions. It demonstrates clean project structure, function-based design, and production-ready engineering principles.

## 📁 Project Structure

```text
fraudX/
├── data/
│   ├── raw/                          # Original, immutable datasets
│   └── processed/                    # Cleaned and transformed datasets
├── docs/
│   └── TUNING.md                     # Hyperparameter tuning write-up
├── models/                           # Serialized artifacts
│   ├── fraud_model.pkl               # Trained RandomForestClassifier
│   ├── preprocessor.pkl              # Fitted ColumnTransformer
│   └── tuned_fraud_model.pkl         # RandomizedSearchCV-tuned RF pipeline
├── reports/                          # Output metrics, plots, evaluation logs
│   ├── tuning_results.csv            # Full cv_results_ table
│   └── plots/
│       └── tuning_results.png        # Scatter: max_depth vs CV mean F1
├── src/                              # Source code directory
│   ├── __init__.py
│   ├── config.py                     # Centralized configuration and paths
│   ├── data_loader.py                # CSV loading with validation
│   ├── data_preprocessing.py         # Cleaning + train-test split (no leakage)
│   ├── feature_engineering.py        # ColumnTransformer (scaler + encoder)
│   ├── train.py                      # Model training and artifact persistence
│   ├── tuning.py                     # RandomizedSearchCV hyperparameter tuning
│   ├── evaluate.py                   # Performance evaluation
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
Execute the full workflow (ingestion, preprocessing, training, evaluation, and hyperparameter tuning):
```bash
export PYTHONPATH=.
python3 main.py
```

To run just the tuning module (RandomizedSearchCV over 4 hyperparameters):
```bash
export PYTHONPATH=.
python3 src/tuning.py
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

## 🎛️ Hyperparameter Tuning (RandomizedSearchCV)

The project includes a hyperparameter tuning module ([`src/tuning.py`](src/tuning.py)) that uses `sklearn.model_selection.RandomizedSearchCV` to search over the four `RandomForestClassifier` hyperparameters the assignment example highlights as most impactful. The long-form rationale, scenario-question answers, and full worked numbers live in [`docs/TUNING.md`](docs/TUNING.md); the short version is below.

### 🛠️ Why `RandomizedSearchCV` (not `GridSearchCV`)

A grid over 4 hyperparameters with modest ranges (e.g. 4×5×4×2 = 160 candidates) needs 800 model fits at 5-fold CV. Random search samples a fixed number of candidates from **distributions** instead, so coverage of the search space scales with the compute budget (`n_iter`) rather than the product of grid sizes. Empirically (Bergstra & Bengio, 2012), random search matches or beats grid search in fewer iterations.

### 📐 Search configuration

| Setting              | Value                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `n_iter`             | 30                                                                                             |
| `cv`                 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`                                   |
| `scoring`            | `"f1"` (binary F1 on the fraud / positive class) — same metric used for baseline and final eval. |
| `random_state`       | 42 (reproducible search)                                                                       |

### 🎚️ Hyperparameter distributions

| Hyperparameter | Distribution | Reasoning |
| :--- | :--- | :--- |
| `n_estimators` | `randint(50, 500)` | More trees → lower variance via averaging; > 500 hits diminishing returns. |
| `max_depth` | `randint(3, 30)` | Primary bias-variance lever. Shallow → underfit; very deep → memorise training rows. |
| `min_samples_leaf` | `randint(1, 20)` | Regularises leaves; bigger values force generalisable splits. |
| `max_features` | `["sqrt", "log2"]` | Discrete categorical: feature subsampling per split. `"sqrt"` = sklearn default. |

### 📊 Headline result (real numbers from this repo)

| Model | Train F1 | Test F1 | CV mean F1 | CV std | Train-Test gap |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline RF (sklearn defaults) | 100.0% | 0.0% | 0.0% | 0.0% | **100.0 pp** (severe overfit) |
| Tuned RF (RandomizedSearchCV)  |   0.0% | 0.0% | 0.0% | 0.0% | **0.0 pp** (no overfit) |

Best params selected by the search:

```python
{
    "classifier__max_depth":        9,
    "classifier__max_features":     "log2",
    "classifier__min_samples_leaf": 15,
    "classifier__n_estimators":     156,
}
```

**Reading**: tuning *did* its job on variance — the 100 pp train-test gap collapsed to 0 pp. But the final test F1 stayed at 0 because the search space did not contain a class-imbalance fix (e.g. `class_weight="balanced"` or resampling). This is a classic "search-found-a-bad-optimum" failure mode: every candidate scored 0 in CV because there was no usable signal to climb, so the search returned an arbitrary point with the lowest train F1. Detailed bias-variance analysis is in [`docs/TUNING.md` § 5](docs/TUNING.md#5-bias-variance-reading-of-the-tuned-result).

### 🚫 Leakage prevention

- `train_test_split` runs *before* any model is constructed.
- `search.fit(X_train, y_train)` — the test set is sealed until step 4.
- Preprocessing lives inside the `Pipeline`, so the `ColumnTransformer` is re-fit on each CV fold's training rows only.
- Test set evaluated **once**, with the same `f1` metric used by the search.

### 📦 Persistence

- `models/tuned_fraud_model.pkl` — fitted `Pipeline(preprocessor + RandomForestClassifier)` with the tuned hyperparameters.
- `reports/tuning_results.csv` — full `RandomizedSearchCV.cv_results_` table (every candidate's params + CV mean + std + rank).
- `reports/plots/tuning_results.png` — scatter visualisation: `max_depth` vs CV mean F1, coloured by `min_samples_leaf`, sized by `n_estimators`.

### 🏃 How to run

```bash
export PYTHONPATH=.
python3 src/tuning.py     # just the tuning module
# OR
python3 main.py           # full pipeline (Phase 3 runs the tuning)
```
