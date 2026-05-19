# Data Leakage Detection and Pipeline Correction

This document is the long-form write-up for the FraudX leakage-detection
step. The short summary lives in
[`README.md`](../README.md#-data-leakage-detection-and-pipeline-correction);
this file goes deeper into the four leakage types, supplies the worked
numbers, provides the assignment's mandatory "Explain the Leakage" and
"Final Conclusion" sections, and answers the five scenario questions.

## 1. What data leakage is

Data leakage is any path by which information from outside the training
set enters the training process. The classic flavour — the one this
module is built around — is **preprocessing leakage**: fitting a
transformer (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`,
`SelectKBest`, …) on data that *includes test rows* before the train /
test split is honoured.

Why it matters: the model is then trained on inputs that were
constructed with the test set's statistics in scope. The held-out
evaluation is no longer a true held-out evaluation, the reported metric
is optimistic, and the production-vs-research metric gap widens silently.

The assignment's "Evaluation Focus" lists the things we need to
demonstrate:
- Conceptual understanding of leakage.
- Proper ML workflow discipline.
- Correct use of `Pipeline` and `ColumnTransformer`.
- Cross-validation integrity.
- Ability to reason about generalisation.

[`src/leakage_correction.py`](../src/leakage_correction.py) runs an
incorrect workflow with FOUR layered leakage types and a correct
Pipeline replacement on the same train / test split, then compares
them with identical metrics.

## 2. The four leakage types we layer

The incorrect workflow stacks four independent leakage paths so the
audit story is complete. Each line below shows the offending call, the
file:line, and what specifically goes wrong.

### 2.1 Scaler leakage

```python
scaler = StandardScaler()
X_num_scaled_full = scaler.fit_transform(X_full[NUMERICAL_FEATURES])
```

`X_full` is the *concatenation* of `X_train` and `X_test`. `fit_transform`
computes mean and variance across both, so the scaler's parameters
encode information about the rows that are about to become the held-out
test set. The training rows are then normalised with those contaminated
parameters.

### 2.2 Imputer leakage

```python
imputer = SimpleImputer(strategy="median")
X_num_imputed_full = imputer.fit_transform(X_num_scaled_full)
```

The median is computed across the full dataset. Even when the data has
no `NaN`s (as is the case for FraudX), the **pattern is structurally
wrong**. On a project with sparse data, the imputed values plugged into
training rows would have been derived from the test rows' empirical
distribution.

### 2.3 Encoder leakage

```python
encoder = OneHotEncoder(handle_unknown="ignore", drop="first", ...)
X_cat_encoded_full = encoder.fit_transform(X_full[CATEGORICAL_FEATURES])
```

The encoder's category vocabulary now includes any category appearing
*only* in the test split. The training-row feature space (number and
ordering of one-hot columns) is now a function of the test set.
On high-cardinality categoricals this becomes catastrophic; on FraudX's
low-cardinality `category` / `location` it still alters the train-side
feature layout.

### 2.4 Feature-selection leakage (the most pernicious)

```python
selector = SelectKBest(score_func=f_classif, k=K_BEST_FEATURES)
X_selected_full = selector.fit_transform(X_processed_full, y_full)
```

`SelectKBest(f_classif)` ranks features by ANOVA F-score *against
y_full*. The retained feature subset is implicitly tuned to the labels
of the rows that will become the test set. This one matters most for
three reasons:

1. The leakage signal is **invisible** without a controlled comparison
   — the chosen feature list looks defensible at code-review time.
2. The offending line *looks virtuous* ("feature selection"!).
3. The training feature subset is now a function of test-set labels,
   which is a leakage path that survives even an otherwise-careful
   downstream Pipeline.

In our run, `SelectKBest` on the full dataset retained features at
indices `[0, 1, 3, 4, 5, 6]` (6 of 7). The Pipeline-side selector,
fitting on training rows only, retained
`['num__amount', 'num__transaction_count', 'cat__category_online',
'cat__category_retail', 'cat__category_travel',
'cat__location_international']`. They are *not* identical in general,
and that gap is the visible signature of feature-selection leakage.

## 3. Why CV is broken in the incorrect workflow

The incorrect workflow calls:

```python
cv_scores = cross_val_score(model, X_train_processed_selected, y_train, cv=5)
```

But `X_train_processed_selected` has already been touched by *four*
fits over the full dataset. Each fold's "validation" rows therefore
participated in the preprocessing parameters and the feature selection.
The folds are no longer independent — the model already had partial
visibility into them before CV started.

## 4. The correct workflow

A single `Pipeline` containing:

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(median)),
                          ("scaler", StandardScaler())]), NUMERICAL_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(most_frequent)),
                          ("onehot",  OneHotEncoder(...))]), CATEGORICAL_FEATURES),
    ])),
    ("selector", SelectKBest(f_classif, k=K_BEST_FEATURES)),
    ("classifier", RandomForestClassifier(random_state=42)),
])
```

For each of the 5 folds in `StratifiedKFold(5, shuffle=True,
random_state=42)`:

| Step | What runs | Sees |
| :--- | :--- | :--- |
| Clone pipeline | `sklearn.base.clone(pipeline)` | nothing |
| Preprocessor fit on fold | `clone.named_steps["preprocessor"].fit(X_train_fold)` | **fold training rows only** |
| Selector fit on fold | `clone.named_steps["selector"].fit(X_train_fold_processed, y_train_fold)` | fold training rows only — including the labels for ranking |
| Classifier fit on fold | `clone.named_steps["classifier"].fit(X_train_fold_selected, y_train_fold)` | fold training rows only |
| Predict + score | `clone.predict(X_val_fold)` then `f1_score(...)` | fold validation rows are transformed (never re-fit) and scored |
| Discard | clone is thrown away | — |

Each fold's preprocessing, feature ranking, and classifier are *all*
fit from scratch on the fold's training subset. Folds are independent.
The CV score is honest.

## 5. Worked numbers (this repo, real run)

Test set: 200 samples (182 class 0 / 18 class 1).
Scoring metric: F1 on the fraud (positive) class — identical for both
workflows.

| Workflow                       | Train F1 | Test F1 | CV mean F1 | CV std |
| :----------------------------- | -------: | ------: | ---------: | -----: |
| Incorrect (4 leakage types)    |   100.0% |    8.0% | **4.71 %** |  5.76% |
| Correct (Pipeline)             |   100.0% |    8.0% | **0.00 %** |  0.00% |

**The leakage signal is 4.71 percentage points of CV F1.** That is the
gap between "we have learning signal" (what the incorrect workflow would
make you believe) and "we have nothing on this minority class" (the
honest truth that the Pipeline workflow surfaces). 4.71pp is small in
absolute terms but matters disproportionately on a metric whose ceiling
is determined by class imbalance — a non-zero CV is the difference
between "ship it" and "keep iterating", and the incorrect workflow
would push the team toward the wrong call.

Both workflows happen to score test F1 = 8.0% (1 of 18 minority cases
correctly identified by chance after `SelectKBest` selected 6 features).
The leakage doesn't change the *final test score* in this particular
random split, but it absolutely changes the *CV mean*, which is what
practitioners watch during iteration.

## 6. Explain the Leakage (Part 1 Step 3 — required, 4-6 lines)

- **What caused the leakage?** Four transformers (`StandardScaler`,
  `SimpleImputer`, `OneHotEncoder`, `SelectKBest`) were all
  `fit_transform`-ed on the concatenation of `X_train` and `X_test`
  before the split boundary was honoured. The training rows the model
  saw were therefore normalised, imputed, encoded, and feature-selected
  using information drawn from the test rows (and the test labels in
  the case of `SelectKBest`).
- **Why are the metrics inflated?** The CV folds inside the incorrect
  workflow's `cross_val_score` are no longer independent. Each fold's
  "validation" rows already participated in the four pre-fits, so the
  classifier inside each fold was trained on inputs that quietly
  encoded properties of its own evaluation rows. The CV mean (4.71%) is
  consequently above the honest CV mean (0.00%) that the Pipeline
  workflow reports.
- **How did test distribution influence training data?** `StandardScaler`
  learned its mean and variance from the full distribution (so training
  rows were rescaled using statistics that include the test rows);
  `OneHotEncoder` admitted any test-only category into the training
  feature space; `SelectKBest` ranked features by their ANOVA F-score
  against the full label vector and retained a feature subset implicitly
  tuned to the test labels.
- **Why is this evaluation unreliable for deployment?** In production
  the model sees data the preprocessing has never been fit on. The
  4.71% CV signal disappears because it was an artefact of the
  contaminated training-time setup, not a real generalisation property.
  Shipping the leaky workflow means promising a CV-derived metric and
  delivering production performance that drops toward the honest 0%.

## 7. Final Conclusion (required in PR)

The incorrect workflow's CV F1 is **4.71 percentage points higher** than
the Pipeline workflow's, and that gap is structural — it survives
re-runs with `random_state=42` because the leakage paths are
deterministic. The test F1 happens to coincide at 8.0% in this
particular split, which is itself instructive: leakage does *not*
always inflate the held-out score by a visible margin; it inflates the
*workflow-internal* metric (CV) that the team uses to make iteration
decisions, while the held-out test score can lag because it's a single
small sample. In a production setting, where the team's iteration
loop is "watch CV go up, ship when it does", the leaky workflow would
have driven the team to ship a model the held-out test set already
knows is no better than majority-class prediction.

The Pipeline workflow is unconditionally preferred. Even when the
absolute numbers happen to be tiny because of an underlying class
imbalance (see [PR #17](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/17)
for the imbalance discussion), the Pipeline workflow reports the right
number, while the leaky workflow reports a misleading one. The cost of
correctness here is essentially zero — the Pipeline is no more code
than the manual workflow, and it gains a single picklable artifact
that handles inference safely.

## 8. Scenario answers (for the video — required)

### 8.1 Why must preprocessing be fitted only on training data?

Because the test set is meant to stand in for production data — data
the model has not seen. If a preprocessing step (scaler / imputer /
encoder / feature selector / target calibrator / anything else with a
`fit()`) ever learns from test rows, the model is trained on inputs
whose construction implicitly used the held-out set, and the
held-out metric stops being an unbiased estimate of production
performance. In particular, *any* transformer that learns parameters
from its input (means, medians, variances, vocabularies, importance
scores, scaling bounds) must derive those parameters from training
rows only. The transform is then applied to the test rows with the
parameters fixed.

### 8.2 How does cross-validation behave differently when using a pipeline?

Without a Pipeline, `cross_val_score(model, X_train_processed, y_train,
cv=5)` runs on data that has *already* been processed once, before CV
started. Each fold's validation rows have already passed through every
`fit_transform` step. The folds are not independent and the CV mean is
optimistic.

With a Pipeline, `cross_val_score(pipeline, X_train, y_train, cv=5)`
hands the *whole pipeline* to sklearn's CV machinery. Sklearn clones
the pipeline for each fold; the clone's preprocessing, feature
selection, and classifier have never been fit. They all fit on that
fold's training subset only. Validation rows are then `.transform()`-ed
through the clone but never participate in any `.fit_*()` call. The
folds are independent; the CV mean is honest.

### 8.3 Why can scaling before GridSearchCV cause leakage?

Because `GridSearchCV` uses cross-validation internally to score every
hyperparameter candidate. If you scale BEFORE handing data to
`GridSearchCV`, every fold's "validation" rows have already been
normalised with statistics computed from the *full* training set —
which includes those validation rows. The grid search then picks the
hyperparameter setting that performs best on a CV that is itself
leaky, so the chosen "best_params" are tuned to the leakage as much as
to the actual data. Wrap the scaler inside a Pipeline and hand the
Pipeline to `GridSearchCV` — the scaler will then re-fit inside every
CV fold on that fold's training rows only.

### 8.4 Why does leakage often go unnoticed?

Three reasons:

1. **The metric goes up, not down.** Leakage *inflates* CV / test
   scores, and "scores went up" is what engineering teams celebrate.
   No alarm fires when a metric improves.
2. **The offending code looks defensible.** `scaler.fit_transform(X)`
   and `SelectKBest(...).fit(X, y)` look like good engineering. They
   only become bugs in the *order* they're called relative to
   `train_test_split` — an ordering decision that's invisible to
   anyone reading a single function.
3. **There's no error.** All four leakage types in this module produce
   valid, fittable, deployable objects. They run cleanly, they
   produce predictions, they pass type checkers. The bug shows up
   only as a *gap between research and production* — visible at deploy
   time, not at PR time. Without a controlled comparison (which this
   module is) the gap is easy to attribute to "production data
   drift" rather than to "the workflow lied to us".

### 8.5 Why is a slightly lower but honest score better than an inflated one?

Because the honest score is *the* signal you actually use to make
shipping decisions. A 0.00% honest CV says "this model is no better
than majority-class prediction; do not ship" — and the team correctly
keeps iterating. A 4.71% inflated CV says "we have something; ship it"
— and the team ships a model that immediately degrades in production
to the honest 0%. The inflated score doesn't just mislead the team
once; it costs the *next* iteration too, because the next experiment's
delta is measured against a baseline that was wrong. Honest numbers
compound; inflated numbers debt-finance future mistakes.

In addition: an honest CV makes hyperparameter tuning trustworthy
(`GridSearchCV` picks for real signal), makes ablation studies
meaningful (you can attribute a delta to a specific change rather
than to leakage variance), and makes production monitoring
interpretable (you know what number to alarm on). Leakage breaks all
of that.

## 9. How to run

```bash
export PYTHONPATH=.
python3 src/leakage_correction.py    # just the demo
# OR
python3 main.py                      # full pipeline (Phase 3 runs the demo)
```

Artifacts produced:

- `models/leakage_correction_pipeline.pkl` — fitted `Pipeline(preprocessor + SelectKBest + RandomForestClassifier)`. Reusable for inference via:
  ```python
  import joblib
  pipeline = joblib.load("models/leakage_correction_pipeline.pkl")
  predictions = pipeline.predict(new_data_df)
  ```

## 10. How this complements other modules

- [PR #15](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/15)
  switched the project to `MinMaxScaler` and introduced split-before-scale
  discipline. This module audits why that discipline matters.
- [PR #17](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/17)
  introduced baseline-vs-RF comparison with per-class metrics. The
  honest 0% CV in this module's correct workflow is the same number
  the baseline-comparison module reports for the minority class.
- [PR #18](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/18)
  ran `RandomizedSearchCV` over a Pipeline. This module's "scaling
  before `GridSearchCV` causes leakage" answer is the conceptual
  underpinning of that PR's design.
- [PR #19](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/19)
  introduced the Pipeline / ColumnTransformer pattern with one
  leakage type and a focus on the *pattern*. This module is the
  audit / accountability layer that stacks four leakage types and
  produces the assignment-specific reflection + scenario answers.
