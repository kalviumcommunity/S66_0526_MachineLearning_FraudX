# Feature Normalization with `MinMaxScaler`

This document is the long-form design write-up for the FraudX feature
normalization step. The short summary lives in the main
[`README.md`](../README.md#%EF%B8%8F-numerical-feature-normalization-minmaxscaler);
this file goes deeper into the *why*.

## 1. What "normalization" means here

In the machine-learning sense, normalization is the act of rescaling each
numerical feature so that they all sit on a comparable range. Without this
step, features with naturally large magnitudes (e.g. transaction `amount`
in dollars) dominate distance- and gradient-based learners over features
with smaller ranges (e.g. `velocity` ratios). Tree-based models like our
`RandomForestClassifier` are insensitive to scale, but we still normalize
because the project is designed to be **model-agnostic** — we want to be
able to swap in kNN, SVM, Logistic Regression, or a small neural network
without re-engineering preprocessing.

`MinMaxScaler` performs an **affine rescaling**:

```
x_scaled = (x - x_min_train) / (x_max_train - x_min_train)
```

After fitting, the training set's minimum maps to `0` and its maximum maps
to `1`. The distribution shape is preserved exactly — only the axis is
remapped. There is no assumption that the data is Gaussian (which would be
required for `StandardScaler` to be ideal).

## 2. Why `MinMaxScaler` and not `StandardScaler`?

| Criterion                                | `MinMaxScaler` ([0, 1])                                                              | `StandardScaler` (z-score)                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Output range                             | **Bounded** to `[0, 1]`                                                              | Unbounded; each feature has mean 0 / std 1, but different empirical ranges                |
| Distribution assumption                  | **None** — purely linear rescaling                                                   | Implicitly favors approximately Gaussian features                                          |
| Sensitivity to outliers                  | Sensitive (extremes set the bounds)                                                  | Moderately sensitive (extremes shift mean / inflate std)                                  |
| Best fit for                             | Distance / margin / gradient-based models on **bounded** inputs                       | Linear models on **Gaussian-like** features                                                |
| Interpretability of scaled values        | High — every feature is in the same `[0, 1]` range                                    | Lower — different features have different empirical ranges around 0                       |

Three reasons drove the choice:

1. **Bounded inputs.** For a fraud-detection product that may eventually
   evolve into a real-time scoring service, having every input in a known
   `[0, 1]` range simplifies monitoring (drift detection, anomaly alerts)
   and downstream consumption.
2. **Model-agnosticism.** A `MinMaxScaler` provides reasonable inputs to
   every model family we might swap in next sprint. A z-score works well
   for parametric / Gaussian-leaning learners but is not optimal for kNN
   or neural networks that benefit from bounded inputs.
3. **Distribution preservation.** Our `transaction_count` and `velocity`
   features are nearly symmetric and `amount` is right-skewed.
   `MinMaxScaler` keeps each feature's shape intact, which is what we
   want until we deliberately decide to apply a non-linear transform
   (e.g. `log1p`).

## 3. Why fitting on the training set only is critical

The single most important rule in this whole assignment is:

> **`fit()` runs on training data only. The test set sees `transform()` and nothing else.**

A `MinMaxScaler` learns two parameters per feature: `data_min_` and
`data_max_`. Those parameters are statistics of the data. If we compute
them across the entire dataset (i.e. `fit_transform()` on `X` before
splitting), we leak information about the test set into the training
process — specifically, the test set's minimum and maximum values
implicitly contribute to the scaling parameters that the model trains on.

The result is **data leakage** (specifically a flavor sometimes called
"feature distribution leakage" or "preprocessing leakage" — see §5 for
the scenario answer). Two consequences:

- **Optimistic test scores.** The reported test accuracy / F1 will look
  better than what the model would actually achieve on truly unseen
  production data, because the model was indirectly tuned to the test
  set's value range.
- **Untrustworthy comparisons.** Two model candidates trained this way
  can no longer be honestly compared on held-out data — both are
  contaminated.

Why re-fitting on inference data is **also wrong** for a different reason:
during inference we may receive a single new transaction. There is no
"min" or "max" to learn from one sample, and even if there were many,
re-fitting means the model sees a different scale than it was trained on
— predictions silently degrade. At inference we **must** call
`.transform()` on the **saved, fitted** scaler.

## 4. Where everything happens in this codebase

| Step                                   | File                                                                                  | Function / Call                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Explicit feature definitions           | [`src/config.py`](../src/config.py)                                                   | `NUMERICAL_FEATURES`, `CATEGORICAL_FEATURES`, `TARGET_COLUMN`              |
| Train-test split (before scaling)      | [`src/data_preprocessing.py`](../src/data_preprocessing.py)                           | `split_data` (calls `train_test_split` with `stratify=y`, `random_state=42`) |
| `fit()` on TRAINING data only          | [`src/feature_engineering.py`](../src/feature_engineering.py) (inside `ColumnTransformer`) and [`src/normalization.py`](../src/normalization.py) | `MinMaxScaler.fit(X_train[NUMERICAL_FEATURES])`                            |
| `transform()` on TEST data             | [`src/normalization.py`](../src/normalization.py) and [`src/train.py`](../src/train.py) | `scaler.transform(X_test[NUMERICAL_FEATURES])`                             |
| Verification (min ≈ 0, max ≈ 1)        | [`src/normalization.py`](../src/normalization.py) and [`src/train.py`](../src/train.py) | `verify_scaled_ranges` / `_verify_minmax_ranges`                            |
| Standalone scaler artifact persistence | [`src/normalization.py`](../src/normalization.py) and [`src/train.py`](../src/train.py) | `joblib.dump(scaler, "models/minmax_scaler.pkl")`                          |
| Inference-time scaling (no refit)      | [`src/predict.py`](../src/predict.py) and `demo_inference_scaling` in `normalization.py` | `pipeline.transform(new_data)` / `scaler.transform(...)`                   |

## 5. Scenario answer (required by Part 2 video)

> **Question:** "If you accidentally fit `MinMaxScaler` on the entire
> dataset before splitting, what type of data leakage occurs and why would
> it make your evaluation unreliable?"

**Answer.** This is **preprocessing leakage** (a form of data leakage in
which test-set statistics influence the training process). When
`fit_transform()` is called on the full dataset before the train-test
split, the scaler computes `data_min_` and `data_max_` over *both* train
and test rows. Two specific things go wrong:

1. The **scaling parameters themselves carry test-set information** — the
   global minimum or maximum may come from a row that ends up in the test
   set. Those parameters are then used to scale the training set, so the
   model is trained on inputs that have been quietly tuned to the test
   set's range.
2. **Reported test metrics are optimistic.** The model performs slightly
   better on this contaminated test set than it would on truly unseen
   production data, because the test set is no longer a true held-out
   sample — its extremes have already influenced training. Picking
   between two model candidates this way is therefore unreliable: you
   may select the model that overfits the contamination, not the model
   that generalizes best.

In summary: leakage produces an evaluation that **does not reflect
real-world performance**, and that is exactly the property a test set is
meant to provide.

## 6. Outlier handling decision

- **Findings (from EDA).** `amount` is right-skewed (skewness ≈ 1.87)
  with a long tail; `transaction_count` and `velocity` are nearly
  symmetric.
- **Action taken.** Outliers in `amount` were **left in place**. They
  were not capped, winsorized, or log-transformed.
- **Why this is acceptable for fraud detection.** Large transactions
  often *are* the signal — capping or removing them would discard
  predictive information.
- **Why `MinMaxScaler` is still appropriate here.** The scaler is
  fit on training data only, so the resulting `[0, 1]` range simply
  encodes "the most extreme transaction we saw during training." A
  larger value in the test set or in production will scale slightly
  above `1.0`, which is acceptable for a `RandomForestClassifier`
  (insensitive to scale) and tolerable for the downstream scale-sensitive
  models we may experiment with later. If outlier-driven tail effects
  become a problem for a future model family, a `log1p(amount)` step
  (fit on train only, applied before scaling) is the next iteration.

## 7. How scaling is handled at prediction time

The fitted `MinMaxScaler` is part of two persisted artifacts:

- `models/preprocessor.pkl` — the full `ColumnTransformer`, used by
  [`src/predict.py`](../src/predict.py).
- `models/minmax_scaler.pkl` — the **standalone** fitted scaler,
  provided so the assignment graders can load and inspect just the
  scaler in isolation.

At inference time:

1. Load the artifact with `joblib.load(...)`.
2. Apply `.transform()` to the new sample's numerical columns. **Never
   `.fit_transform()`.**
3. Pass the scaled features (along with the OneHot-encoded categorical
   features from the same persisted pipeline) into the model.

A working demonstration is in `demo_inference_scaling` inside
[`src/normalization.py`](../src/normalization.py) and in
[`src/predict.py`](../src/predict.py).

---

## Quick-reference checklist

- [x] `NUMERICAL_FEATURES`, `CATEGORICAL_FEATURES`, `TARGET_COLUMN` declared explicitly in `src/config.py`
- [x] `train_test_split` happens **before** any scaler is instantiated
- [x] `random_state=42`, `stratify=y` for reproducibility and class balance
- [x] `MinMaxScaler.fit()` runs on training data only
- [x] `MinMaxScaler.transform()` is used (never `fit_transform()`) on test data and new data
- [x] Categorical features are left unscaled
- [x] Min ≈ 0 and Max ≈ 1 verification on training scaled features (asserted, not just printed)
- [x] Standalone fitted scaler persisted to `models/minmax_scaler.pkl` via `joblib`
- [x] Loading-for-prediction demonstration in `normalization.py` and `predict.py`
- [x] Outlier inspection results documented and a deliberate decision recorded
- [x] README explains why `MinMaxScaler` was chosen over `StandardScaler`, leakage prevention, and inference-time scaling
