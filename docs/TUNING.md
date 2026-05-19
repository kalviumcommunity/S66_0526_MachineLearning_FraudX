# Hyperparameter Tuning with `RandomizedSearchCV`

This document is the long-form write-up for the FraudX hyperparameter-tuning
step. The short summary lives in
[`README.md`](../README.md#-hyperparameter-tuning-randomizedsearchcv);
this file goes deeper into the *why* and supplies the worked numbers.

## 1. Why `RandomizedSearchCV` instead of `GridSearchCV`

A modest 4-hyperparameter grid explodes combinatorially. Even with conservative
ranges:

```
n_estimators  ∈ {50, 100, 200, 500}     # 4 values
max_depth     ∈ {3, 5, 10, 20, 30}      # 5 values
min_samples_leaf ∈ {1, 5, 10, 20}        # 4 values
max_features  ∈ {"sqrt", "log2"}         # 2 values
```

That is `4 * 5 * 4 * 2 = 160 candidates`. At 5-fold CV: **800 model fits**.
If we widen any range, the cost blows up linearly per axis added.

`RandomizedSearchCV` samples a fixed `n_iter` candidates from **distributions**
over each hyperparameter. Coverage of the search space scales with the
*compute budget* (`n_iter`), not with the *product of grid sizes*. The
result, as Bergstra & Bengio (2012) showed empirically, is that random
search matches or beats grid search in fewer iterations, because many
hyperparameters have low effective sensitivity — sampling them at random
doesn't waste evaluations on near-identical settings.

For this module we use **`n_iter = 30`**. Concretely that means 30 unique
hyperparameter combinations × 5 CV folds = 150 model fits — about 5×
cheaper than the modest grid above, with substantially better coverage of
the *interesting* combinations of `max_depth` and `min_samples_leaf`.

## 2. Parameter distributions (NOT fixed grids)

| Hyperparameter             | Distribution                  | Why                                                                                                       |
| -------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------- |
| `classifier__n_estimators` | `scipy.stats.randint(50, 500)` | More trees lower variance via averaging; > 500 stops helping and gets expensive.                          |
| `classifier__max_depth`    | `scipy.stats.randint(3, 30)`  | Biggest bias-variance lever. Shallow (3-5) underfit; very deep (> 20) memorise rows on n=800 train data. |
| `classifier__min_samples_leaf` | `scipy.stats.randint(1, 20)` | Smaller leaves overfit; larger leaves force generalisable splits. Critical on imbalanced data.            |
| `classifier__max_features` | `["sqrt", "log2"]`            | Discrete categorical. `"sqrt"` = sklearn default; `"log2"` is more aggressive feature subsampling.        |

The integer distributions (`randint(low, high)`) sample uniformly between
their bounds — every value in the range is equally likely. The discrete
list for `max_features` samples uniformly between the two options. These
choices match the assignment example's "Random Forest" recommendation.

## 3. Leakage discipline

Every step below is enforced in [`src/tuning.py`](../src/tuning.py).

| Rule                                                                                          | Where it's enforced                                                                                                                       |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Train / test split runs *before* any model is constructed.                                    | `src/data_preprocessing.py::split_data` is called first; `RandomForestClassifier` and `RandomizedSearchCV` are instantiated *after*.       |
| Tuning sees `X_train` / `y_train` only — the test set is sealed until step 4.                 | `search.fit(X_train, y_train)`. There is no `search.fit(X, y)` call anywhere.                                                              |
| Preprocessing happens *inside* each CV fold.                                                  | The search runs over `Pipeline(preprocessor + classifier)`; sklearn re-fits the entire pipeline for every fold, so the `ColumnTransformer` sees only that fold's train rows. |
| Test set is evaluated exactly once.                                                           | `tuned_metrics["test_f1"] = f1_score(y_test, best_estimator.predict(X_test))` is called *after* `search.fit(...)` and never again.        |
| Identical scoring metric is used for baseline, search, and final evaluation.                  | `SCORING = "f1"` is a module-level constant; every reported score uses the same metric so cross-comparisons are honest.                    |

## 4. Worked example — actual FraudX numbers

### 4.1 Baseline `RandomForestClassifier` (sklearn defaults)

The default `RandomForestClassifier()` produces a model that perfectly
memorises the 800 training rows and then predicts the majority class on
everything it hasn't seen:

| Metric                  | Value      |
| ----------------------- | ---------- |
| Train F1 (fraud class)  | **100.0 %** |
| Test F1 (fraud class)   | **0.0 %**   |
| CV mean F1              | 0.0 %      |
| CV std F1               | 0.0 %      |
| Train–Test gap          | **100.0 pp** — extreme overfit |

The 100 pp train-test gap is the textbook overfit pattern: the model
learns the training set perfectly but generalises like a coin flip — except
even worse, because the validation/test class balance pulls it toward
predicting class 0 for everything.

### 4.2 Tuned `RandomForestClassifier` (best of 30 random samples)

`RandomizedSearchCV` selected:

```python
{
    "classifier__max_depth":         9,
    "classifier__max_features":      "log2",
    "classifier__min_samples_leaf":  15,
    "classifier__n_estimators":      156,
}
```

Resulting metrics:

| Metric                  | Value      |
| ----------------------- | ---------- |
| Train F1 (fraud class)  | **0.0 %**   |
| Test F1 (fraud class)   | **0.0 %**   |
| CV mean F1              | 0.0 %      |
| CV std F1               | 0.0 %      |
| Train–Test gap          | **0.0 pp** — no overfit |
| best_score_ (CV mean)   | 0.0000     |
| n_iter explored         | 30         |

### 4.3 Side-by-side

| Model                            | Train F1 | Test F1 | CV mean | CV std | Train–Test |
| -------------------------------- | -------: | ------: | ------: | -----: | ---------: |
| Baseline RF (sklearn defaults)   |   100.0 % |   0.0 % |   0.0 % |   0.0 % |    100.0 pp |
| Tuned RF (RandomizedSearchCV)    |     0.0 % |   0.0 % |   0.0 % |   0.0 % |      0.0 pp |

## 5. Bias-variance reading of the tuned result

Tuning **massively** reduced the variance of the model (100 pp gap →
0 pp gap) — that part of the job worked exactly as designed. The
combination of `max_depth=9` and `min_samples_leaf=15` is enough regularisation
to stop the trees from carving out individual training rows.

The problem is what's left behind: a strongly-regularised RF that has
*no* signal to learn from is biased toward predicting the majority
class everywhere. Test F1 went from 0.0 % to 0.0 % because the train-side
F1 collapse (100 % → 0 %) is what made the train-test gap close — not
because the test side moved up.

So:
- **Variance**: reduced dramatically (good).
- **Bias on the minority class**: still maximal (bad).
- **Generalisation**: unchanged in absolute terms (Test F1 stayed at 0).

This is the **classic search-found-a-bad-optimum failure mode**. The
search space did not contain a hyperparameter that addresses class
imbalance, so the cross-validation objective (binary F1 on the positive
class) reported 0 for *every* candidate — there is no signal to climb,
so the search returns an arbitrary point with the lowest train F1.
You can see this directly in `reports/tuning_results.csv`: all 30
candidates rank `rank_test_score = 1` because they all score `mean_test_score = 0.0`.

## 6. Was the search budget enough? Was the search "smart"?

- **Search efficiency**: `n_iter=30` × 5-fold CV = 150 fits is sound for
  a 4-axis search. It would be wrong to crank `n_iter` higher hoping
  for a different result here: every candidate scored 0, so the issue
  is the *shape of the loss surface*, not the *number of probes on it*.

- **Search quality**: The chosen distributions cover sensible ranges
  for each hyperparameter individually. What they do **not** cover is
  the *class-distribution* lever. Two well-known ways to add it:

  1. Add `classifier__class_weight: [None, "balanced"]` as a fifth
     hyperparameter. This re-weights the loss inside each tree to give
     minority-class errors more importance.
  2. Resample the training set before search (SMOTE / random
     under-sampling), keeping the test set untouched.

  Either of these would produce a non-degenerate CV loss surface, which
  is what `RandomizedSearchCV` actually needs to do its job. **Better
  search is not the answer here; better search space is.**

## 7. Visualisation

[`reports/plots/tuning_results.png`](../reports/plots/tuning_results.png) is a
scatter of all 30 sampled candidates:

- x-axis = `max_depth`
- y-axis = mean CV F1 (fraud class)
- color  = `min_samples_leaf`
- marker size = `n_estimators`

In this run every point sits at y = 0 because the search space did not
contain a class-imbalance fix. Once we add `class_weight` (or resample),
the y-axis will spread out and we'll be able to read off how each
hyperparameter affects CV F1.

## 8. Scenario answers (for Part 2 video)

> *Scenario: RandomizedSearchCV on Random Forest with `n_iter=15`, wide ranges,
> 5-fold CV. Results: best CV score 0.91, train acc 0.99, test acc 0.74,
> CV std 0.06.*

### 8.1 What does the gap between training and test accuracy suggest?

A train accuracy of 0.99 with a test accuracy of 0.74 is a **25-point
train-test gap** — strong evidence of **overfitting** (high variance).
The model has memorised patterns in the training data that don't
generalise to the held-out test set. This is the high-variance side of
the bias-variance trade-off: the model is too complex for the underlying
signal, so small fluctuations in the training data are being treated as
information.

### 8.2 Why might CV score still appear high?

Because **`best_score_` is the CV score of the *best* candidate over all
`n_iter` runs**, not a typical candidate. With wide ranges and only 15
random samples, it's likely that a single candidate got lucky with a
favourable fold split. A CV std of 0.06 on top of that means the CV
estimate itself fluctuates by ±6 percentage points across folds — so a
"best CV score" of 0.91 could easily mean a true generalisation in the
0.85–0.97 range, and the test result of 0.74 sits clearly outside that.
That gap is also a red flag for the search itself being unstable: the
"best" candidate may simply be a lucky draw, not the genuinely-best
configuration.

### 8.3 Is `n_iter = 15` sufficient for this search space? Why or why not?

**No**, almost certainly not. With wide ranges over many hyperparameters,
15 samples leaves enormous gaps in the search space. A useful rule of
thumb: `n_iter` should grow at least linearly with the number of
hyperparameters when ranges are wide. For 4–5 hyperparameters with wide
ranges, `n_iter = 30–60` is a more defensible starting point; for 7+
hyperparameters or extremely wide ranges, you'd want 100+. With only 15,
you're more likely to overfit the *search* — picking a lucky single
configuration — than to find a genuinely best one.

### 8.4 How would you refine the tuning strategy to improve generalisation?

Three concrete moves:

1. **Use repeated stratified K-fold** (e.g. `RepeatedStratifiedKFold` with
   3 repeats of 5 splits) so `best_score_` averages over more splits and
   `cv_std` becomes a more reliable estimate of true variance.
2. **Increase `n_iter` to 50–100** to cover more of the search space.
3. **Narrow the search space using prior knowledge after a first pass.**
   The first pass identifies the region that produces good CV scores;
   the second pass zooms in around it with tighter distributions.
4. **Add explicit regularisation hyperparameters** (e.g.
   `min_samples_leaf` with `randint(1, 50)` rather than just `(1, 20)`)
   so the search can actually push back on the observed 25-point
   train-test gap. Without that, the search has no lever for "be less
   overfit".

### 8.5 Would narrowing parameter ranges help? Explain.

Yes — narrower ranges effectively **increase the density of probes per
unit of hyperparameter space**, so with the same `n_iter` you cover the
likely-good region more thoroughly. But narrowing matters only if you
have *evidence* about where the good region lies. The right workflow is
"wide first pass → narrow second pass":

- First pass: wide ranges, modest `n_iter`. Inspect which combinations
  produced the top decile of CV scores.
- Second pass: narrow each range around the centroid of those top
  candidates, and increase `n_iter` (or switch to `GridSearchCV` if the
  remaining grid is small enough to afford).

Narrowing without that evidence — just picking smaller ranges by feel —
risks excluding the actual optimum and locking in a sub-optimal model
with high confidence. That's worse than the original wide search.

## 9. How to run

```bash
export PYTHONPATH=.
python3 src/tuning.py          # just the tuning module
# OR
python3 main.py                # full pipeline (Phase 1 train → Phase 2 inference → Phase 3 tuning)
```

Artifacts produced:

- `models/tuned_fraud_model.pkl` — fitted `Pipeline(preprocessor + RandomForestClassifier)` with the tuned hyperparameters.
- `reports/tuning_results.csv` — full `RandomizedSearchCV.cv_results_` table.
- `reports/plots/tuning_results.png` — scatter of `max_depth` vs CV mean F1.

## 10. Quick-reference checklist

- [x] Proper train/test split (`random_state=42`, `stratify=y`) runs before any tuning.
- [x] Test data not used during tuning (`search.fit(X_train, y_train)` only).
- [x] Pipeline used so preprocessing happens inside each CV fold (no leakage).
- [x] Reasoning for chosen hyperparameters documented (§ 2).
- [x] Per-class metrics + train/test/CV breakdown reported for both baseline and tuned model.
- [x] Same scoring metric (`f1` on the fraud class) used throughout.
- [x] Honest "did tuning help?" verdict — not a metric-laundering exercise.
