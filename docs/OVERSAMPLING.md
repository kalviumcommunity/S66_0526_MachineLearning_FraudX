# Oversampling for Imbalanced Classification (Random + SMOTE)

This document is the long-form write-up for the FraudX oversampling
module. The short summary lives in
[`README.md`](../README.md#-oversampling-for-imbalanced-classification);
this file supplies the worked numbers, the Part 5 scenario answers,
the recall-precision trade-off analysis, and the business
recommendation.

## 1. What oversampling does

Oversampling rebalances the training set by **adding minority-class
rows**. Two flavours:

| Technique | What it does | Trade-off |
| :--- | :--- | :--- |
| **RandomOverSampler** | Randomly duplicates minority rows until both classes have the same count. | Simplest possible. Doesn't add new information — just gives the model more *exposure* to existing minority rows. Risks overfitting on those exact rows. |
| **SMOTE** | For each minority row, picks a random one of its k=5 nearest minority neighbours and creates a new row on the line segment between them. | Synthesises rows the model has not literally seen, increasing minority diversity. The new rows are still constrained to the convex hull of the original minority class, so SMOTE cannot manufacture truly novel signal (e.g., from a region of feature space where no minority example exists). |

Both differ from class weighting (PR #22):

| Lever | Changes... | Effect |
| :--- | :--- | :--- |
| Class weights | the **loss / impurity** | Each minority error counts ~10× more, but the data itself is unchanged. |
| Oversampling | the **training data** | Class 0 / class 1 ratio is now 50 / 50 (or whatever `sampling_strategy` is set to). The loss treats every row equally as before. |

In principle they can be combined. This module evaluates resampling on
its own; class weighting was already evaluated in PR #22.

## 2. Why oversampling must live INSIDE the cross-validation loop

(Mandatory: Part 4 of the assignment.)

The naive workflow:

```python
# THIS IS WRONG.
X_train_os, y_train_os = SMOTE().fit_resample(X_train, y_train)
scores = cross_val_score(model, X_train_os, y_train_os, cv=5)
```

is *broken* because each CV fold's "validation" rows now contain
synthetic minority rows generated from minority neighbours that are in
the same fold's training set. The model has effectively seen its
validation rows during training. The reported CV score is
**leakage-inflated**.

The correct workflow:

```python
from imblearn.pipeline import Pipeline   # NOT sklearn.pipeline.Pipeline
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer(...)),
    ("sampler", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42)),
])
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

Inside `cross_val_score`:

| Step | What runs | Sees |
| :--- | :--- | :--- |
| Clone pipeline | `sklearn.base.clone(pipeline)` — fresh, never-fitted everything | nothing |
| Fit preprocessor | on the **fold's training rows** | fold training only |
| Fit sampler | on the **fold's processed training rows** | fold training only |
| Fit classifier | on the **resampled fold training rows** | fold training only |
| Predict + score | on the **fold's validation rows** — sampler is BYPASSED | validation rows are transformed (not resampled, not retrained on) |

The key invariant: the resampler is skipped at `predict` time. sklearn's
Pipeline can't do this because it has no notion of an estimator that
changes row count, but imblearn's Pipeline does. **This is the only
leakage-safe way to combine a sampler with CV.**

## 3. Leakage discipline (this module enforces it)

[`src/oversampling.py`](../src/oversampling.py):

- `train_test_split` runs **before** any sampler is instantiated.
- Both `RandomOverSampler` and `SMOTE` are wrapped in
  `imblearn.pipeline.Pipeline` along with the preprocessor and the
  classifier. CV is run on the whole pipeline; the sampler re-fits in
  every fold on that fold's training rows only.
- Test set is sealed. `pipeline.predict(X_test)` bypasses the sampler
  entirely.
- Identical scoring metric (`f1` on the positive / fraud class) used
  for baseline + Random OS + SMOTE, so the comparison is honest.

## 4. Worked numbers — FraudX (real run)

Training set after resampling (required output):

| Stage                        | class 0 (legit) | class 1 (fraud) | total |
| :--------------------------- | --------------: | --------------: | ----: |
| Before (original train)      |             727 |              73 |   800 |
| After RandomOverSampler      |             727 |             727 | 1,454 |
| After SMOTE                  |             727 |             727 | 1,454 |

Test-set evaluation (same X_test, y_test, same metric for all three):

| Model                       | Accuracy | Precision (1) | Recall (1) | F1 (1) | CV mean F1 | CV std |
| :-------------------------- | -------: | ------------: | ---------: | -----: | ---------: | -----: |
| Baseline RF                 |  91.00 % |        0.00 % |     0.00 % | 0.00 % |     0.00 % | 0.00 % |
| RandomOverSampler + RF      |  89.00 % |       16.67 % |     5.56 % | 8.33 % |     2.50 % | 5.00 % |
| SMOTE + RF                  |  83.00 % |        5.56 % |     5.56 % | 5.56 % |     6.18 % | 8.70 % |

### 4.1 Confusion matrices

**Baseline RF** (matches the majority-class baseline exactly):

|              | predicted 0 | predicted 1 |
| ------------ | ----------: | ----------: |
| **actual 0** | TN = 182    | FP = 0      |
| **actual 1** | FN = 18     | TP = 0      |

**RandomOverSampler + RF**:

|              | predicted 0 | predicted 1 |
| ------------ | ----------: | ----------: |
| **actual 0** | TN = 177    | FP = 5      |
| **actual 1** | FN = 17     | TP = 1      |

**SMOTE + RF**:

|              | predicted 0 | predicted 1 |
| ------------ | ----------: | ----------: |
| **actual 0** | TN = 165    | FP = 17     |
| **actual 1** | FN = 17     | TP = 1      |

## 5. Recall-precision trade-off

This is the **first module in the project where a trained model
actually catches fraud cases**. Both Random OS and SMOTE flip exactly
1 of 18 test-set fraud cases to a true positive (TP = 1).

The accuracy drop is the *cost*: 91 % → 89 % (Random OS) → 83 % (SMOTE).
Each false positive subtracts 0.5 pp from accuracy in a 200-sample
test set, and SMOTE produces 17 false positives vs Random OS's 5. The
classifier learned to predict class 1 *more aggressively* on SMOTE,
because the synthetic minority rows enlarged the minority class's
feature footprint enough to influence more split decisions.

Concretely:

- **Random OS**: 1 TP / 5 FP → precision = 1/6 ≈ 16.67 %. The
  duplicated minority rows are exact copies, so the model only
  predicts class 1 in regions where the original minority rows live.
- **SMOTE**: 1 TP / 17 FP → precision = 1/18 ≈ 5.56 %. The synthetic
  rows live on line segments between minority neighbours, which
  enlarges the minority footprint into nearby regions of feature
  space and produces more false positives.

For both: **recall is identical (5.56 %)**, but **precision differs
3×** in favour of Random OS. This is the canonical
oversampling trade-off — and SMOTE's CV mean F1 (6.18 %) is meanwhile
the *highest* of the three, because the synthetic diversity helps
*on the CV folds* even if it costs precision on this specific test
draw.

The right operating point depends on the business cost ratio
between false negatives and false positives. See §7.

## 6. Part 5 — Scenario answers (mandatory)

### 6.1 Why must oversampling never be applied before train/test split?

Because oversampling treats the rows it operates on as the "training
set". If you `SMOTE().fit_resample(X, y)` on the full dataset, then
split, the test set ends up containing:

- **synthetic rows** that the SMOTE algorithm interpolated between
  pairs of (possibly future-test) minority neighbours — they were
  literally constructed from data that should have been held out;
- **duplicated rows** (for `RandomOverSampler`) that are copies of
  rows the model trained on — so the model has memorised them.

The held-out test metric is then inflated and unreliable. Splitting
first guarantees the test set contains only original data the model
has never seen.

### 6.2 How does SMOTE differ from random oversampling?

| | RandomOverSampler | SMOTE |
| :--- | :--- | :--- |
| Strategy | Duplicates random minority rows | Interpolates new minority rows between a row and its k nearest neighbours |
| Adds new information | No (duplicates) | A little (synthesises rows the model hasn't literally seen) |
| Risk of overfitting | High (model can memorise duplicated rows) | Lower (synthesis adds diversity), but bounded by the existing minority distribution |
| Computational cost | O(n) | O(n log n) for the k-NN search per minority row |
| Best for | Quick sanity check; when minority class is already diverse | Datasets where minority rows are sparse but reasonably clustered in feature space |

In this run, SMOTE produced higher CV mean F1 (6.18 % vs 2.50 %)
because the synthetic rows added diversity that helped generalisation.
But on the single test draw, SMOTE traded that diversity for more
false positives, ending up with slightly lower test F1 than Random OS.

### 6.3 Why might precision decrease after oversampling?

Because oversampling **biases the model toward predicting class 1
more eagerly**. Mechanically:

- Before oversampling, the model trains on 727 class-0 rows and 73
  class-1 rows. The split-impurity at each node favours class-0
  predictions because they're 10× more common.
- After oversampling, the model trains on 727 class-0 rows and 727
  class-1 rows. The impurity criterion now treats both classes
  symmetrically. Splits that isolate minority rows become attractive,
  and the model's leaves are more likely to predict class 1.

The cost is **false positives**: rows the model previously correctly
classified as class 0 now get predicted as class 1. Each FP lowers
precision (= TP / (TP + FP)). On this run, going from Random OS
(5 FP) to SMOTE (17 FP) tracks the increased eagerness directly.

### 6.4 When would class weights be preferred over oversampling?

Class weights are preferred when:

- **Dataset is large** and resampling would double or triple training
  time without proportional benefit.
- **Memory is tight** — oversampling literally multiplies the row
  count; class weighting doesn't.
- **The minority class is already diverse** — SMOTE's synthesis adds
  little when the minority class already covers its natural region of
  feature space.
- **You want to keep training-set diagnostics interpretable** —
  oversampled training sets contain duplicates / synthetic rows, so
  diagnostics like "training accuracy" or "training feature
  histograms" get distorted in subtle ways.

Oversampling is preferred when:

- The minority class is *very* sparse and the model needs more
  examples to fit useful decision boundaries.
- The learner doesn't easily expose a class-weight knob (some
  off-the-shelf libraries don't).
- You want to also explore SMOTE / ADASYN variants that synthesise
  new examples — that capability only exists on the resampling side.

For FraudX specifically, **resampling produced visible recall gains
where class weighting (PR #22) did not** — likely because changing
the *data* exposed the trees to enough minority instances to find
useful splits, where reweighting the *loss* alone wasn't enough on
73 examples.

### 6.5 Does oversampling create new information? Why or why not?

**Strictly: no.**

- **RandomOverSampler** literally duplicates existing rows. Each
  duplicate adds zero new information — the same feature values
  appear multiple times. What it changes is the model's *exposure*:
  splits get evaluated against more minority rows, which can produce
  different impurity decisions.
- **SMOTE** creates rows on line segments between existing minority
  neighbours. Those rows didn't literally exist in the original data,
  but they're constrained to the convex hull of the original minority
  class. SMOTE adds *diversity* within that hull, not *signal* from
  outside it.

If the true fraud distribution has modes in regions where no original
fraud examples live, neither resampler will manufacture those rows.
The fundamental limit of oversampling is the support of the original
minority class. Truly novel fraud patterns require new labelled data
or domain knowledge encoded as features — not oversampling.

## 7. Final recommendation (business perspective)

For a fraud-detection deployment, the FN/FP cost ratio is typically
high (50:1 or higher). At that ratio, both Random OS and SMOTE
*outperform* the baseline despite their precision costs.

Concretely on this run:

- Baseline: catches 0 of 18 fraud cases → all 18 FN go through to
  production → ~$X cost (where X depends on average transaction
  amount).
- Random OS: catches 1 of 18 → 5 FP (~5 minutes of customer service
  each = ~25 min total).
- SMOTE: catches 1 of 18 → 17 FP (~85 min of customer service total).

If `c_FN / c_FP ≥ 25`, both Random OS and SMOTE are net positive
versus the baseline. Random OS is the better starting point — same
recall as SMOTE but 3× higher precision — but SMOTE's higher CV mean
F1 (6.18 % vs 2.50 %) suggests it generalises better; the test-set
gap may close on a different test sample.

**RECOMMENDATION**: ship the RandomOverSampler+RF configuration at
the default 0.5 threshold. It improves minority-class recall over the
baseline without sacrificing F1, which is the right joint metric on
imbalanced data. Operate at the default 0.5 threshold to start; tune
the threshold with `predict_proba` from the saved
`models/smote_fraud_model.pkl` once a business cost ratio
(c_FN / c_FP) is set.

The natural next iteration combines this module's RandomOS with PR
#18's tuning search (let `RandomizedSearchCV` choose between
RandomOS and SMOTE alongside the RF hyperparameters), against a
stated `c_FN/c_FP` cost.

## 8. How to run

```bash
pip install imbalanced-learn==0.12.4
export PYTHONPATH=.
python3 src/oversampling.py    # just the analysis
# OR
python3 main.py                 # full pipeline (Phase 3 runs the analysis)
```

Artifacts produced:

- `models/smote_fraud_model.pkl` — fitted
  `imblearn.Pipeline(preprocessor + SMOTE + RandomForestClassifier)`.
  Reusable at inference (the sampler is skipped automatically on
  `predict()` / `predict_proba()`):
  ```python
  import joblib
  pipeline = joblib.load("models/smote_fraud_model.pkl")
  proba = pipeline.predict_proba(new_data_df)[:, 1]   # for threshold tuning
  ```
- `reports/plots/oversampling_confusion_matrices.png` — 3-panel
  confusion-matrix heatmap (Baseline | Random OS | SMOTE).

## 9. How this complements the rest of the project

- [PR #15](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/15)
  — `MinMaxScaler` normalisation (preprocessing concerns).
- [PR #17](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/17)
  — baseline-vs-RF comparison harness. Diagnosed RF == baseline.
- [PR #18](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/18)
  — `RandomizedSearchCV`. The natural home for combining
  oversampling + RF hyperparameter search.
- [PR #19](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/19)
  — canonical Pipeline + ColumnTransformer pattern.
- [PR #20](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/20)
  — four-leakage audit + Pipeline correction.
- [PR #21](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/21)
  — class imbalance analysis (severity, PR-AUC, ROC-AUC).
- [PR #22](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/22)
  — class weighting. Re-weighted the loss; didn't lift recall.
- **THIS PR** — resampling. Re-weights the *data*. **First module
  where the trained model catches fraud cases** (TP = 1 of 18 for
  both Random OS and SMOTE), with a visible accuracy ↓ / recall ↑
  trade-off and a business recommendation.

The arc: diagnose imbalance → try the loss-side fix (class weights) →
try the data-side fix (oversampling). Class weighting didn't move
the needle; oversampling did. Both are now available as composable
levers in subsequent iterations.
