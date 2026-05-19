# Scikit-Learn Pipeline Integration

This document is the long-form write-up for the FraudX pipeline-integration
step. The short summary lives in
[`README.md`](../README.md#-scikit-learn-pipeline-integration);
this file goes deeper into the *why* and supplies the worked numbers,
the mandatory reflection answers, and the scenario answers.

## 1. What a Pipeline is and why it matters

A `sklearn.pipeline.Pipeline` is a single object that chains together a
sequence of fitted-and-transforming steps (the preprocessing) plus a
final estimator (the model). It exposes the standard
`.fit()` / `.predict()` / `.score()` API of any sklearn estimator,
which means downstream tooling (`cross_val_score`, `GridSearchCV`,
`RandomizedSearchCV`) can hand a fresh slice of data to the *entire*
pipeline and the preprocessing automatically re-fits on that slice.

The thing the Pipeline buys you is *not* convenience. It's *correctness*.
A pipelined workflow makes data leakage essentially impossible by
construction:

- During cross-validation, sklearn clones the pipeline for every fold.
  The `ColumnTransformer` inside the clone has never been fit on
  anything. When `clone.fit(X_train_fold, y_train_fold)` runs, the
  preprocessing learns its parameters from that fold's training rows
  *only*. The validation rows are then `.transform()`-ed with those
  parameters and never seen by the fit. There is no path for leakage.
- During inference, you call `loaded_pipeline.predict(new_sample)`. The
  preprocessing inside the pipeline applies `.transform()` (never
  `.fit_transform()`) using parameters learned at training time.
  Predictions automatically include the preprocessing.

The "manual" alternative — call `scaler.fit_transform(X)`, then
`train_test_split`, then `cross_val_score(model, X_train_processed, y_train)`
— breaks both properties.

## 2. What `ColumnTransformer` adds

Real-world tabular data mixes feature types. The FraudX dataset has
three numerical features (`amount`, `transaction_count`, `velocity`)
and two categorical features (`category`, `location`). They need
different preprocessing: scaling for the first set, one-hot encoding
for the second.

`ColumnTransformer` is the canonical way to handle this. It takes a list
of `(name, transformer, columns)` triples and, when fit, runs each
transformer on its declared subset of columns in parallel, then
horizontally concatenates the outputs into a single feature matrix.
Two important properties:

1. The transformers are **independent** — the numerical pipeline never
   sees the categorical columns and vice versa. That means imputing
   numeric missing values with the median doesn't accidentally try to
   compute a median over a categorical string column.
2. The transformers are **fit together** as a single object — when
   the parent pipeline is `.fit()`, the entire `ColumnTransformer`
   refits, and when it's `.transform()`-ed, the entire ColumnTransformer
   transforms. There is no per-column manual bookkeeping for the
   caller to mess up.

In this module the `ColumnTransformer` is the FIRST step of the
Pipeline:

```python
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, NUMERICAL_FEATURES),
    ("cat", cat_pipeline, CATEGORICAL_FEATURES),
])
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])
```

## 3. The manual workflow — what goes wrong

[`src/pipeline_demo.py::_approach_a_manual_with_leakage`](../src/pipeline_demo.py)
implements the bad pattern explicitly. Two bugs that should never
appear in real code:

```python
# Bug 1: fit the scaler on X_train + X_test combined.
scaler = StandardScaler()
X_num_scaled_full = scaler.fit_transform(X_full[NUMERICAL_FEATURES])

# Bug 2: same with the encoder.
encoder = OneHotEncoder(handle_unknown="ignore", drop="first", ...)
X_cat_encoded_full = encoder.fit_transform(X_full[CATEGORICAL_FEATURES])
```

By the time `train_test_split` runs, the scaler's mean/variance and the
encoder's category vocabulary already incorporate information from
*every row of the dataset*, including the rows that are about to become
the held-out test set. The held-out evaluation is no longer an honest
test of generalisation: the model is trained on inputs whose normalisation
parameters were tuned on the test set.

Then the manual workflow runs `cross_val_score(model, X_train_processed,
y_train, cv=5)` — and the same problem extends to CV: every fold's
"validation" rows have *already* been seen by the scaler/encoder during
their fit. The reported CV score is inflated.

In other datasets the inflation is dramatic (especially with rare
categories, sparse text features, or strong outliers). On this small,
imbalanced FraudX dataset the underlying class-distribution problem
dominates and the leakage signal happens to be near zero — but the
**structural bug is still present**, and on a less degenerate dataset
it would matter a lot.

## 4. The proper workflow — what makes it safe

[`src/pipeline_demo.py::_approach_b_pipeline`](../src/pipeline_demo.py)
puts everything inside a single Pipeline:

```python
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([...])),
    ("classifier", RandomForestClassifier(...)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")

pipeline.fit(X_train, y_train)
test_f1 = f1_score(y_test, pipeline.predict(X_test))
```

Three properties that make this safe:

1. `cross_val_score` clones the pipeline for every fold. The clone's
   preprocessor has never seen any data. `clone.fit(...)` runs on that
   fold's training subset only. The validation subset is then
   `.transform()`-ed but never touched by `fit`.
2. The test set is `.transform()`-ed *after* `pipeline.fit(X_train,
   y_train)`, using parameters learned from `X_train` alone. The test
   set has never participated in any `.fit_*()` call.
3. The fitted pipeline is a single picklable object. Loading it at
   inference time and calling `loaded.predict(new_sample)` automatically
   re-applies the preprocessing using the SAME parameters used at training.

## 5. Cross-validation inside the Pipeline — the mechanics

For each of the 5 folds in `StratifiedKFold`:

| Step                        | What runs                                                                 | Sees                                  |
| --------------------------- | ------------------------------------------------------------------------- | ------------------------------------- |
| Clone pipeline              | `sklearn.base.clone(pipeline)` — fresh, never-fitted ColumnTransformer + RandomForestClassifier | nothing                               |
| Fit preprocessing on fold   | `clone.named_steps["preprocessor"].fit(X_train_fold, y_train_fold)`        | **fold training rows only**           |
| Fit classifier on fold      | `clone.named_steps["classifier"].fit(X_train_fold_transformed, y_train_fold)` | fold training rows only           |
| Predict + score on fold val | `clone.predict(X_val_fold)` then `f1_score(...)`                          | fold validation rows are transformed (not re-fit) and scored |
| Discard clone               | clone is thrown away; next fold gets a brand-new clone                    |                                       |

`cross_val_score` aggregates the per-fold scores into a mean ± std.
Stratified folding preserves the ~91/9 class balance in each split so
F1 is well-defined for every fold.

## 6. Worked example — actual FraudX numbers

Test set: 200 samples (182 class 0 / 18 class 1).

| Approach                | Train F1 | Test F1 | CV mean F1 | CV std |
| ----------------------- | -------: | ------: | ---------: | -----: |
| Manual (with leakage)   |   100.0% |    0.0% |       0.0% |   0.0% |
| Pipeline (no leakage)   |   100.0% |    0.0% |       0.0% |   0.0% |

On this specific dataset both approaches collapse to the same numeric
result because the default RandomForestClassifier cannot learn the
minority class under the existing 91/9 imbalance (see PR #17 for the
full baseline-vs-RF analysis). The CV F1 is zero in both cases because
the model predicts class 0 for every fold-validation row. The leakage
in Approach A has nothing to act on — but the structural bug is still
present, and on a less degenerate dataset Approach A's CV would be
visibly inflated above Approach B's.

The important lesson is correctness rather than headline numbers.
Approach B is the right workflow even when the metric difference is
zero, because:

- it survives a switch to a more balanced dataset without rewriting,
- it composes with `GridSearchCV` / `RandomizedSearchCV` correctly out
  of the box (see PR #18),
- it produces a single picklable artifact that handles inference safely.

## 7. Reflection (mandatory in PR description)

### 7.1 Why is preprocessing outside a pipeline risky?

Because nothing forces the preprocessor's fit to be restricted to
training rows. The most natural way to write the workflow — fit
once, transform later — incorporates the test set's statistics into the
fit. The result is **data leakage**: the model is trained on inputs that
were normalised using information from the test set, the held-out
metric becomes optimistic, and the gap from research metric to
production metric grows without anyone noticing.

### 7.2 How does Pipeline improve reproducibility?

The fitted pipeline is a single picklable object that contains every
preprocessing parameter the model needs. Loading it at any future
date — different machine, different sklearn version pinned the same
way, different developer — yields identical predictions for the same
input. You cannot accidentally drop a step, swap a step ordering, or
forget to apply the encoder. Combined with `random_state` and pinned
package versions, the pipeline makes the workflow byte-for-byte
reproducible.

### 7.3 Why is `ColumnTransformer` important for mixed data types?

Numerical and categorical features require fundamentally different
preprocessing (scaling vs encoding, with different imputation strategies).
Doing them by hand requires manually slicing the DataFrame, applying
each transformer to its slice, then stitching the outputs back together
in the right column order. Every manual step is a place for a bug —
column-order swaps, silent dtype changes, forgetting to apply the
imputer before the encoder, dropping the index. `ColumnTransformer`
declares the mapping once and the rest is automatic: each transformer
fits and transforms on its declared columns, outputs are concatenated
correctly, the whole thing participates in CV cleanly, and the fitted
object remembers everything for inference.

### 7.4 What could go wrong if encoding is done before train/test split?

Two specific problems:

1. The encoder's category vocabulary will include categories that exist
   only in the test set. The model is then trained on inputs whose
   *dimensionality* was decided with the test set in scope. At inference
   on a truly-unseen sample with a different category, the model
   behaves differently from how it behaved during evaluation — the
   reported metric is optimistic.
2. For high-cardinality categoricals, the encoder may learn rare
   categories that exist only in the test split. Once that one-hot
   column is part of the training feature matrix, the model can pick
   up *spurious* signal from the all-zero column. The reported metric
   is again optimistic, and the model fails to generalise.

Both problems disappear if the encoder is fit inside the Pipeline,
because then the fold's encoder only ever sees the fold's training rows.

## 8. Scenario answers (for Part 2 video)

> *"You preprocess your dataset manually: fit the scaler on the entire
> dataset, encode categories before splitting, then split into train/test
> and train a model. You report 92% test accuracy. Later, when deployed,
> the model performs at 78%."*

### 8.1 What mistake likely occurred in preprocessing?

Both transformers were fit on the **entire dataset before the train/test
split**. The scaler's mean/variance and the encoder's category
vocabulary incorporated information from the rows that were about to
become the test set. The test split is therefore not a true held-out
set — its statistics already participated in training. This is
preprocessing leakage; specifically, the worst form of it (fitting
across the split boundary) rather than the milder form (forgetting to
re-fit per CV fold).

### 8.2 How did data leakage inflate test accuracy?

The model trains on inputs whose normalisation parameters and category
vocabulary were tuned with the test set's statistics in scope. Two
concrete inflation mechanisms:

1. **Distribution shift goes away.** Normally the test set has a
   slightly different feature distribution than the training set —
   that's the point of holding it out. Fitting the scaler globally
   forces train and test inputs onto the same scale, hiding the
   distribution gap the model would have to handle in production.
2. **Rare-category awareness.** Categories that exist only in the
   test set get their own one-hot columns. The training rows carry
   zeros in those columns, and the model implicitly learns "if this
   rare-category column is 1, behave differently." In production,
   genuinely-unseen categories follow `handle_unknown="ignore"` (zeros
   across the board) — a regime the model never trained on. The
   reported 92% test accuracy doesn't reflect that.

### 8.3 How would using a Pipeline prevent this issue?

A Pipeline forces the preprocessing fit to happen as part of the model
fit, inside the `pipeline.fit(X_train, y_train)` call. Because
`X_train` is the result of the prior `train_test_split`, the
preprocessing's `.fit(...)` cannot see test rows — they simply don't
exist in the array being passed. The same restriction holds during
cross-validation: sklearn clones the pipeline per fold, so each fold's
preprocessing fits on that fold's training subset only. The test set
is `.transform()`-ed (never `.fit_transform()`-ed) once at the end of
the workflow. Leakage becomes impossible by construction, not by
discipline.

### 8.4 Why must preprocessing be fitted only on training data?

Because the test set is meant to be a stand-in for production data —
data the model has not yet seen. If preprocessing parameters are
derived from the test set, the model is *implicitly* trained on
properties of the test set, and the test metric stops reflecting
production performance. The 92→78 gap in the scenario is the textbook
symptom: the model looked great in evaluation because evaluation
quietly cheated, and the production drop is the unbiased measurement
finally arriving.

The principle generalises beyond scaling. Anything that learns from
data — imputers, encoders, feature-selection masks, target-leakage
detectors, calibrators — must learn from the training set only. The
test set transforms with those parameters and contributes nothing back.

### 8.5 What role does cross-validation play in detecting such leakage?

Cross-validation provides multiple held-out splits, so a leakage-induced
inflation shows up as a *characteristic* gap between CV mean and final
test score, *and* between CV with proper pipeline vs CV with
pre-transformed inputs. Specifically:

- If your CV (without a pipeline) reports 92% and your test reports
  92%, you'll trust the model. In production it drops to 78%.
- If your CV (with a pipeline) reports 78% and your test reports 78%,
  you correctly anticipate production behaviour.

So CV *with a pipeline* is the diagnostic. CV *without* a pipeline
reproduces the same leakage as the global preprocessing and gives a
falsely-comforting number. The asymmetry — pipelined CV always tells
the truth, manual CV can be tricked — is exactly why the workflow in
[`src/pipeline_demo.py`](../src/pipeline_demo.py) keeps both side by
side and computes the gap explicitly.

## 9. How to run

```bash
export PYTHONPATH=.
python3 src/pipeline_demo.py    # just the demo
# OR
python3 main.py                 # full pipeline (Phase 3 runs the demo)
```

Artifacts produced:

- `models/sklearn_pipeline.pkl` — fitted `Pipeline(ColumnTransformer + RandomForestClassifier)`. Reusable for inference via:
  ```python
  import joblib
  pipeline = joblib.load("models/sklearn_pipeline.pkl")
  predictions = pipeline.predict(new_data_df)   # preprocessing applied automatically
  ```
