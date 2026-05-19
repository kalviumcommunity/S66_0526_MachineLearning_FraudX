# Multi-Model Comparison with Cross-Validation

This document is the long-form write-up for the FraudX multi-model
comparison module. The short summary lives in
[`README.md`](../README.md#-multi-model-comparison-with-cross-validation);
this file supplies the worked numbers, the five Part 5 comparative-analysis
answers, the bias-variance discussion, and the final justified selection.

## 1. What "fair comparison" requires

The assignment is explicit: "The goal is not simply to find the highest
score — but to demonstrate disciplined comparison and reasoned model
selection." Five invariants are required for that to be possible:

| Invariant | Why |
| :--- | :--- |
| Same train/test split | If A is tested on a different sample than B, the difference reflects sample variance, not model quality. |
| Same preprocessing pipeline | If A sees StandardScaler-ed features and B sees raw features, the difference reflects preprocessing, not the model. |
| Same CV strategy | If A is evaluated with 5-fold and B with 3-fold, A's variance estimate is finer-grained — but the means are not comparable. |
| Same scoring metric | An F1-optimised candidate compared against an accuracy-optimised candidate is two different optimisation problems. |
| Same test set, evaluated once per model | Re-using the test set for repeated comparisons turns it into a validation set and inflates the reported metric for the chosen winner. |

[`src/model_comparison.py`](../src/model_comparison.py) enforces all
five: identical `Pipeline(ColumnTransformer + classifier)`, identical
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, identical
`scoring="f1"` on the positive class, identical
`X_train` / `X_test` / `y_train` / `y_test` across all three models.

## 2. The three candidates

| Model | Why it's in the comparison |
| :--- | :--- |
| **Logistic Regression** (L2-regularised) | Linear baseline. Fast, interpretable, well-calibrated probabilities out of the box. Reveals whether the FraudX signal is linearly separable. |
| **Random Forest** | The project's incumbent model in every prior module (PRs #15-#23). Tree ensemble; low variance via bagging. |
| **Gradient Boosting** | Sequential trees fit on residuals. Typically the strongest tabular learner of the three. Provides the most interesting comparison vs RF. |

All three are built via `_build_pipeline(label)` in the source — the
ONLY thing that differs between the three pipelines is the final
`classifier` step.

## 3. Cross-validation results

| Model              | CV Mean F1 | CV Std F1 |
| :----------------- | ---------: | --------: |
| LogisticRegression |   0.00 %   |   0.00 %  |
| RandomForest       |   0.00 %   |   0.00 %  |
| GradientBoosting   |   0.00 %   |   0.00 %  |

All three CV means are 0 because the default-threshold predictions
collapse to majority-class behaviour on FraudX's severely imbalanced
test folds — the same ceiling PR #17 / #21 / #22 identified. The CV
std is also 0 across folds, which means each fold produces an
identical "predict class 0 everywhere" result (F1 on the positive
class = 0 in every fold). The bar chart at
[`reports/plots/model_comparison_cv.png`](../reports/plots/model_comparison_cv.png)
visualises the tied means and the zero error bars.

## 4. Test-set evaluation (single-shot per model)

| Model              | Accuracy | Precision (1) | Recall (1) | F1 (1) | Test confusion matrix |
| :----------------- | -------: | ------------: | ---------: | -----: | :--- |
| LogisticRegression |  91.00 % |        0.00 % |     0.00 % | 0.00 % | TN=182 FP=0 FN=18 TP=0 |
| RandomForest       |  91.00 % |        0.00 % |     0.00 % | 0.00 % | TN=182 FP=0 FN=18 TP=0 |
| GradientBoosting   |  88.50 % |        0.00 % |     0.00 % | 0.00 % | TN=177 FP=5 FN=18 TP=0 |

A subtle but important observation: **Gradient Boosting predicts class 1
5 times on the test set, while LR and RF never predict class 1.** GB
gets all 5 of those positive predictions wrong (precision = 0), but
the willingness to predict class 1 *at all* is a real signal that GB
has more learning capacity than the other two. It loses 2.5 pp of
accuracy by being more eager — a cost that pays off when combined with
techniques that improve calibration (threshold tuning) or balance the
classes (PR #22 / #23).

LR and RF, by contrast, produce the literal majority-class baseline
predictions on this test draw.

## 5. CV-to-test gap and overfitting check

| Model              | CV Mean | Test F1 | Gap |
| :----------------- | ------: | ------: | --: |
| LogisticRegression |   0.00 %|   0.00 %|  0.00 pp |
| RandomForest       |   0.00 %|   0.00 %|  0.00 pp |
| GradientBoosting   |   0.00 %|   0.00 %|  0.00 pp |

CV-to-test gap is 0 for every model because both numbers collapse to 0
under the imbalance ceiling. No model is *literally* overfitting in the
classical sense (train F1 >> test F1) — the issue is the opposite,
*underfitting* on the minority class. PR #23 (Oversampling) is the
canonical iteration that broke through this ceiling.

## 6. Final justified model selection

**Selected model: LogisticRegression.**

Selection rule (encoded in [`_interpret_cv`](../src/model_comparison.py)):

1. Pick the model with the **highest CV mean F1**.
2. If two means are within 1 pp of each other (i.e. statistically
   indistinguishable on this dataset), break the tie by **lowest CV std**.
3. If still tied, select the first model in declaration order — which
   here is Logistic Regression.

All three CV means are 0.00 % (tied), all three CV stds are 0.00 %
(tied), so the tie-break falls to declaration order → Logistic
Regression.

The deeper reading: this dataset cannot statistically distinguish the
three models at the default threshold. The fair conclusion is that
*all three are equivalent under the chosen metric and threshold*. The
sensible production choice then defaults to **operational considerations**:

- **Interpretability**: LR exposes coefficients per feature. Easy
  audit trail for regulators / compliance. RF and GB don't.
- **Inference latency**: LR is a single dot product. RF and GB are
  multi-tree traversals. For high-throughput payment-gateway scoring
  the LR latency is preferable.
- **Calibration**: LR's `predict_proba` is well-calibrated by maximum
  likelihood. RF and GB are over-confident on small datasets and need
  Platt / isotonic calibration to be useful in cost-sensitive
  threshold decisions.
- **Training cost / retraining cadence**: LR refits in milliseconds.
  Nightly / hourly retraining is trivial. GB and RF are more
  expensive.

If the team's deployment instead optimises for marginal *capacity to
learn* (e.g. with class weights, resampling, or threshold tuning
applied on top), **GradientBoosting** is the right pick — it already
shows non-zero positive predictions where LR and RF don't. The
selected model from a pure CV-mean perspective is LR; the selected
model from a "downstream-iterable" perspective is GB. Both reasonings
are defensible.

The persisted artifact at `models/best_comparison_model.pkl` is the
LogisticRegression-based pipeline (the CV-rule winner).

## 7. Part 5 — Comparative Analysis answers (required)

### 7.1 Why is cross-validation better than a single train/test split for model comparison?

A single split is a sample of size 1 from the distribution of possible
splits. The metric you read is one random draw, and on small / imbalanced
data the variance across splits is large. CV computes that metric over
k different holdouts; the mean is closer to the true expected
performance and the std tells you how much a single split could mislead
you. On 200 test rows here, a single-split metric can swing several pp
just from which 18 fraud cases happen to be in the test set. CV reduces
that swing by averaging over 5 different (train, validation) pairs
within `X_train` itself.

### 7.2 Why must the test set not be used to select the best model repeatedly?

Because every selection decision implicitly fits the model family to the
test set. If you pick the best of 3 (or 30) models by their test scores,
you've effectively run a hyperparameter search where the hyperparameter
is "which model class", and the test score is no longer an unbiased
estimate of generalisation — it's an overfit-on-the-test-set artifact.
The correct workflow is: pick by CV, evaluate the chosen model ONCE on
the test set, report that single number. The test set is "spent" after
that one evaluation, and any subsequent comparison should generate a new
held-out test set (or use nested CV) to avoid multiple-comparisons
inflation.

### 7.3 If two models differ by 0.01 in score but one has lower variance, which would you choose and why?

Lower variance, every time. A 0.01-point difference is well inside the
noise of 5-fold CV on a 200-sample test set; the higher-mean model is
not statistically distinguishable from the lower-mean one. The
lower-variance model is the more reliable production choice — its CV
std tells you the expected swing in performance across deployments,
which is exactly the quantity an SRE / risk-management process cares
about. The higher-mean / higher-variance model might give you a great
quarter and a terrible quarter; the lower-mean / lower-variance model
gives you two okay quarters. The okay-okay path is preferable for
production systems where the cost of a bad quarter is high.

### 7.4 How does model complexity affect bias and variance?

Higher-capacity models (deeper trees, more parameters, less
regularisation) reduce **bias** — they can fit more complex patterns —
but raise **variance**, because they latch onto patterns specific to
the particular training set. Lower-capacity models do the reverse:
more bias (they cannot capture rare patterns), less variance (they're
stable across training sets).

The sweet spot depends on the data size relative to the signal
complexity. On a small dataset like FraudX (n=800 training rows),
high-capacity models (e.g. deep RF or GB with low `min_samples_leaf`)
will overfit; low-capacity models (e.g. shallow GB or LR with strong
L2) will underfit. The CV std is the diagnostic: high std means the
model is too sensitive to training-set quirks (variance dominates);
low std with low mean means it's too rigid to find signal (bias
dominates).

In our results, all three models have CV std = 0 *because the imbalance
ceiling makes them all predict class 0 in every fold*. That's not a
real "low variance" signal — it's a degenerate one. PR #23
(Oversampling) demonstrated the same point: until the imbalance is
addressed, the bias-variance trade-off is moot because the bias is
maximal.

### 7.5 When might a slightly lower-performing model be preferable?

When **operational concerns** matter — and they almost always do in
production ML:

- **Interpretability**: Logistic Regression has coefficient-level
  explanations; GB and RF don't (without SHAP or per-tree inspection).
  A regulated industry (finance, healthcare, hiring) may prefer the
  LR even at lower accuracy because every prediction has an auditable
  reason chain.
- **Inference latency / cost**: GB scoring is sequential and slower
  than LR. For high-throughput payment-gateway fraud scoring, the
  cheaper model can be the right call even if its CV mean is 1-2 pp
  lower.
- **Training cost / retraining cadence**: If the model is re-fit
  nightly on a growing dataset, LR scales much better than GB.
- **Calibration**: LR produces well-calibrated probabilities out of
  the box; GB and RF don't. For cost-sensitive threshold decisions
  (the whole point of fraud detection), calibrated probabilities
  matter more than raw F1.
- **Robustness to distribution drift**: Simpler models with strong
  regularisation tend to degrade more gracefully when production data
  drifts from training-time assumptions. A 95 % accuracy GB that
  silently drops to 70 % when payment patterns shift is worse than a
  92 % LR that holds at 88 %.

For FraudX, the CV-rule winner is **Logistic Regression**. If a
deployment constraint (latency, interpretability, retraining cost,
calibration) tilts the choice away from it, the answers above name the
legitimate reasons.

## 8. How to run

```bash
export PYTHONPATH=.
python3 src/model_comparison.py    # just the comparison
# OR
python3 main.py                    # full pipeline (Phase 3 runs the comparison)
```

Artifacts produced:

- `models/best_comparison_model.pkl` — fitted Pipeline for the
  CV-rule winner (Logistic Regression in this run). Reload via
  `joblib.load(...)` for downstream use.
- `reports/plots/model_comparison_cv.png` — CV mean ± std bar chart.

## 9. How this complements the rest of the project

- [PR #15](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/15)
  — MinMaxScaler normalisation.
- [PR #17](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/17)
  — baseline vs RF comparison harness. Diagnosed RF == baseline.
- [PR #18](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/18)
  — RandomizedSearchCV. Natural home for tuning the selected model.
- [PR #19](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/19)
  — Pipeline + ColumnTransformer pattern (reused here).
- [PR #20](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/20)
  — 4-leakage audit + Pipeline correction.
- [PR #21](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/21)
  — imbalance diagnosis (severity, PR-AUC, ROC-AUC).
- [PR #22](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/22)
  — class weighting.
- [PR #23](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/23)
  — oversampling (Random + SMOTE). First module where the trained
  model catches fraud cases.
- **THIS PR** — first module that compares the project's incumbent RF
  against alternatives in a fair head-to-head. Confirms that on this
  dataset, at the default threshold, all three model classes hit the
  same imbalance ceiling. The natural next iteration is to add
  resampling (PR #23) on top of the selected model and re-run this
  comparison.
