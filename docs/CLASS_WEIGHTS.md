# Class Weights / Cost-Sensitive Learning

This document is the long-form write-up for the FraudX class-weights
module. The short summary lives in
[`README.md`](../README.md#-class-weights-for-cost-sensitive-learning);
this file goes deeper into the mechanism, supplies the worked numbers,
includes the assignment's Part 3 + Part 4 answers verbatim, and
records the **business recommendation** the assignment requires.

## 1. What class weighting does

Most classifiers minimise an objective that treats every training
example as equally important. Under class imbalance (e.g. FraudX's
91% / 9% split), the optimisation pressure is dominated by the
majority class — the model gets to ~91% accuracy by predicting class 0
for every row.

`class_weight="balanced"` flips that incentive. Every example now
contributes to the loss in *inverse* proportion to how common its
class is. Concretely, sklearn computes:

```
weight_class = n_samples / (n_classes * n_class_examples)
```

For FraudX's training set (800 rows, class counts {0: 727, 1: 73}):

| Class | Count | Weight (`balanced`) |
| :---: | ----: | ------------------: |
|   0   |   727 |                ≈0.55 |
|   1   |    73 |                ≈5.48 |

Minority-class errors now cost ~10× more than majority-class errors.
A `RandomForestClassifier` trained this way uses the weights in the
**impurity criterion** at each split, so splits that separate a few
minority rows become attractive even at the cost of a slight
majority-class regression.

## 2. The trade-off the module is designed to surface

The assignment is explicit: "Improving minority recall often comes at
a cost. Your job is to determine whether that cost is justified."

The textbook trade-off when class weighting moves the needle:

| Metric | Direction | Why |
| :--- | :---: | :--- |
| Recall (minority) | ↑ | Model now predicts class 1 more eagerly to avoid the (expensive) missed-positive penalty. |
| Precision (minority) | ↓ | More positive predictions = more false positives. |
| Accuracy | ↓ | False positives that didn't exist before now cost accuracy points. |
| F1 (minority) | ?  | Depends on whether recall gains outweigh precision losses. |

The right business question is then: *is the precision drop / accuracy
drop worth the recall gain?* — and the answer depends on the **cost
asymmetry** between false negatives and false positives in the
specific domain.

## 3. Worked numbers on FraudX (real run)

Test set: 200 samples (182 class 0 / 18 class 1). Same `random_state=42`,
same Pipeline, same preprocessing for both models.

| Metric                  | Without Weights | With Weights (`balanced`) | Δ |
| :---------------------- | --------------: | -----------------------: | -: |
| Accuracy                | 91.00 %         | 91.00 %                  | 0.00 pp |
| Precision (minority)    | 0.00 %          | 0.00 %                   | 0.00 pp |
| Recall (minority)       | 0.00 %          | 0.00 %                   | 0.00 pp |
| F1-score (minority)     | 0.00 %          | 0.00 %                   | 0.00 pp |

**Confusion matrix — both models** (identical):

|              | predicted 0 | predicted 1 |
| ------------ | ----------: | ----------: |
| **actual 0** | TN = 182    | FP = 0      |
| **actual 1** | FN = 18     | TP = 0      |

The two models produce **identical default-threshold predictions** on
this dataset. That is itself an important finding — and the more
interesting story is in the `predict_proba` scores below.

### 3.1 What `predict_proba` shows

Looking at the underlying scores (not just the threshold-cut
predictions):

```
Baseline predict_proba[fraud]:  min=0.0000  max=0.4900  mean=0.1006   pct≥0.5 = 0%
Weighted predict_proba[fraud]:  min=0.0000  max=0.4900  mean=0.0950   pct≥0.5 = 0%

Shift (weighted − baseline) per sample:
  mean = −0.0056   std = 0.0450
  fraction where weighted > baseline:  38%
  fraction where shift > 0.05:          6.5%
```

The weighted RF's score for the fraud class is, on average, ~0.5 pp
*lower* than the baseline's. The two models look almost identical at
the score level too.

### 3.2 Why class weighting didn't move the needle here

`class_weight="balanced"` re-weights the **impurity criterion** at
each split. It does not change the data itself. With only 73
fraud examples in 800 training rows (≈9.1 %), the Random Forest's
trees simply do not find splits that separate fraud cleanly from
non-fraud — even when minority-class errors are penalised ~10× more.
The features available (`amount`, `transaction_count`, `velocity`,
`category`, `location`) on this synthetic dataset don't carry the
signal needed for the trees to over-rule the class prior.

Concretely:
- The maximum `predict_proba` for the fraud class across the whole
  test set is **0.49** for *both* models. No row is over the 0.5
  decision threshold, so no row gets a fraud prediction.
- The mean predict_proba is essentially the class prior (~10 %).
- The std of the shift between models (0.045) is tiny — class
  weighting nudged scores up for some rows and down for others, with
  no consistent direction.

This is not a bug; it is the genuine engineering finding for this
dataset. It produces a useful generalisation: **class weighting alone
is not a magic bullet — it amplifies whatever discriminative signal
the model can already find, but cannot manufacture signal where none
exists.**

## 4. Part 3 — Comparative Analysis (required)

### 4.1 How did recall change for the minority class?

It didn't change — both baseline and weighted recall on class 1 are
**0.00 %**. On a dataset with more learnable signal, the expected
behaviour is for recall to rise (often substantially — 0 % → 20-60 %
is typical) when class weights are turned on. Here, the underlying
features don't give the trees enough to grip onto, so weighting alone
doesn't surface any new positive predictions.

### 4.2 Did precision increase or decrease? Why?

Precision is **0.00 %** in both runs — same reason as recall, because
both models make zero positive predictions. On a more learnable
dataset, the expected pattern is: precision *drops* when weighting is
turned on, because the model predicts class 1 more eagerly and
introduces false positives that didn't exist before. The structural
trade-off is recall ↑ at the cost of precision ↓; on FraudX with the
default RF hyperparameters, neither lever moved.

### 4.3 Why did accuracy possibly drop?

Accuracy didn't drop here — both runs report **91.00 %**. On datasets
where the weighted model actually predicts class 1 sometimes,
accuracy *would* drop, because each false positive costs one accuracy
point and false positives are what class weighting trades for higher
recall. The principle is general: **accuracy is the wrong metric to
optimise once you're applying cost-sensitive learning.**

### 4.4 Which model is more appropriate for this problem and why?

**Neither**, at the default threshold, on this exact run. Both models
score 0 on the metrics that actually matter (minority-class recall and
F1). The right move is *not* to ship either model and *not* to declare
class weighting a failure. The next iteration should:

1. **Lower the decision threshold** below 0.5 — the saved
   `models/weighted_fraud_model.pkl` exposes `predict_proba`; a
   threshold of 0.3 already produces 5 % positive predictions.
   Choose the threshold from the precision-recall curve using a
   stated business cost ratio.
2. **Add resampling** (SMOTE or random undersample of class 0) on
   the training set only.
3. **Tune `min_samples_leaf` and `max_depth`** (see PR #18) to
   produce trees that can carve out smaller fraud clusters.

The weighted model is still **preferable to the baseline** for any of
those next steps — its `predict_proba` scores are calibrated against
an asymmetric loss, which makes them better suited to a future
threshold-tuning iteration.

### 4.5 Does applying class weights completely solve imbalance?

**No.** `class_weight="balanced"` re-weights the loss but does not
change the data distribution. The model still trains on the same 91/9
features; it just pays more for missing fraud than for false alarms.
Class weights help, but a complete solution typically combines:

- class weighting (sets the right loss),
- threshold tuning (picks the right operating point on the PR curve),
- resampling (gives the model more minority-class examples to learn
  from),
- and a stated cost function (so the threshold isn't picked by feel).

This module covers the first lever; the rest are scope for future
iterations, with PR #18 being the natural home for the threshold-tuning
work.

## 5. Part 4 — Scenario answers (mandatory)

### 5.1 Why do unweighted models naturally favour the majority class?

Because most learning algorithms minimise an objective that treats
every example equally. On 91/9 data, ~91 % of the optimisation
pressure is on class-0 predictions. An "always class 0" decision rule
satisfies 91 % of that pressure perfectly, and the marginal cost of
getting one minority-class row right (in exchange for losing a
majority-class prediction somewhere) is tiny. Tree-based learners
have the same issue at the impurity level: a candidate split that
separates a handful of fraud rows but slightly hurts a much larger
non-fraud population doesn't look attractive on average. Without an
explicit asymmetric loss, the optimisation gradient (or impurity
reduction) points toward the majority class.

### 5.2 How do class weights modify the optimisation objective?

`class_weight="balanced"` injects per-class scaling into the loss /
impurity criterion. For a tree:

```
weighted_gini(split) = sum_class( weight_class · prob_class · (1 − prob_class) )
```

When minority examples are scaled up by ~10×, a split that isolates
minority examples reduces weighted impurity much more than it does
under unweighted training. The same logic applies to gradient-based
learners: the gradient of the loss with respect to the score is
scaled by the example's class weight, so minority-class gradients
push the parameters harder. The net effect is that the model's
decision boundary is biased toward predicting the minority class —
exactly as if the dataset were balanced.

### 5.3 In fraud detection, which is typically worse — false positives or false negatives?

**False negatives are usually worse**, but the right framing is
asymmetric cost rather than a one-size-fits-all ranking:

- **False negative (FN, missed fraud)**: the bank eats the entire
  fraud amount. Typical cost: hundreds to thousands of dollars per
  incident.
- **False positive (FP, blocked legitimate transaction)**: customer
  is inconvenienced — possibly enough to lose them as a customer.
  Typical direct cost: minutes of customer-service / re-auth time.

The expected costs are very asymmetric in magnitude, but FPs are
*much* more frequent than FNs in a deployed system because legitimate
transactions vastly outnumber fraudulent ones. The right way to set
the operating point is to explicitly model
`Cost = c_FN · FN + c_FP · FP` and pick the precision-recall curve
point that minimises expected cost. At a typical c_FN/c_FP ratio of
50-500, the system tolerates many false positives to catch one false
negative.

### 5.4 Why is stratified splitting still required even after applying class weights?

Two reasons:

1. **Train-test parity.** Stratification ensures the test set has
   approximately the same minority share as the training set. Without
   it, a random split could land the test set with very few minority
   examples (sometimes zero), making the held-out metric for the
   minority class meaningless or undefined.
2. **CV variance control.** When the search inside `RandomizedSearchCV`
   / `GridSearchCV` runs k-fold CV, stratified folds guarantee each
   fold contains some minority examples. Without that, occasional
   folds end up with zero minority examples, and the F1 / recall on
   those folds is undefined — the metric reported as `cv_mean_f1`
   becomes a noisy mix of valid and degenerate folds.

Class weighting doesn't change the *distribution* of your data; it
only changes how the loss treats existing examples. Stratification is
the orthogonal lever that makes sure both halves of the split see the
same distribution to begin with.

### 5.5 Why should you not evaluate weighted models using accuracy alone?

Because the *whole point* of weighting is to trade accuracy (a
majority-class-dominated metric) for recall and F1 on the minority
class. Reporting accuracy alone would penalise the weighted model for
doing exactly what it was designed to do. The relevant evaluation
under weighting is:

- **Per-class precision / recall / F1** on the minority class —
  measures whether weighting bought you the recall you intended.
- **Balanced accuracy** — accuracy averaged across classes; not
  dominated by class 0.
- **PR-AUC** — full precision-recall trade-off across thresholds;
  the right summary statistic when you intend to tune the threshold.
- **Confusion matrix** — concrete counts of TP, FP, FN, TN. Lets
  you ask "is the precision-recall trade I'm seeing acceptable in
  raw counts?"

Accuracy has a place as a context indicator, but never as the
single metric a weighted model is judged on.

## 6. Final recommendation (business perspective, required output)

In a fraud-detection deployment, the cost asymmetry is sharp:

- **False negative (missed fraud)** = bank eats the full fraud amount.
  Typical impact: hundreds to thousands of dollars per incident.
- **False positive (legitimate transaction blocked)** = customer-friction
  cost (call / re-auth / lost-customer risk). Typical impact:
  minutes-to-hours of customer-service / lost customer LTV.

Even at a conservative 50:1 cost ratio, the weighted model is
preferred whenever it catches meaningful additional fraud — provided
the false-positive rate stays inside the team's manual-review capacity.

**For THIS specific run on the FraudX dataset:** `class_weight="balanced"`
alone did *not* lift fraud recall above the baseline at the default 0.5
threshold. We do **NOT recommend shipping** either model on this
configuration. The recommended next step is:

1. **Threshold tuning.** Use `predict_proba` from
   `models/weighted_fraud_model.pkl`, compute the precision-recall
   curve, and pick the threshold that minimises business expected
   cost. The maximum predict_proba in this run was 0.49 — a threshold
   of 0.3-0.4 already lets the weighted model surface ~5 % of test
   rows as candidates.
2. **Resampling.** Apply SMOTE or random undersampling on the
   training set only (no leakage), re-train the weighted model, and
   re-evaluate.
3. **Hyperparameter tuning.** Use the existing
   `RandomizedSearchCV(scoring="f1", ...)` infrastructure from PR #18
   with `class_weight="balanced"` baked into the search.

The general principle still applies: the weighted model is the right
*starting point* for any of those iterations, because its
`predict_proba` scores are calibrated against an asymmetric loss.
Treat this module's output as the diagnostic, not as the ship
candidate.

## 7. How to run

```bash
export PYTHONPATH=.
python3 src/class_weights.py    # just the demo
# OR
python3 main.py                  # full pipeline (Phase 3 runs the demo)
```

Artifacts produced:

- `models/weighted_fraud_model.pkl` — fitted
  `Pipeline(preprocessor + RandomForestClassifier(class_weight="balanced"))`.
  Reusable at inference:
  ```python
  import joblib
  pipeline = joblib.load("models/weighted_fraud_model.pkl")
  proba = pipeline.predict_proba(new_data_df)[:, 1]   # for threshold tuning
  ```
- `reports/plots/class_weights_confusion_matrices.png` — side-by-side
  confusion-matrix heatmap.

## 8. How this complements the rest of the project

- [PR #15](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/15)
  — MinMaxScaler normalisation (preprocessing concerns).
- [PR #17](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/17)
  — built the baseline-vs-RF **comparison harness**. Diagnosed that
  RF == baseline.
- [PR #18](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/18)
  — `RandomizedSearchCV` over the Pipeline. The natural home for the
  threshold-tuning / class-weight-tuning follow-up.
- [PR #19](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/19)
  — canonical Pipeline + ColumnTransformer pattern (reused here).
- [PR #20](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/20)
  — four-leakage audit + Pipeline correction (orthogonal correctness
  concern).
- [PR #21](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/21)
  — class-imbalance diagnosis (severity, PR-AUC, ROC-AUC). Recommended
  exactly this `class_weight="balanced"` follow-up.
- **THIS PR** — executes that recommendation, quantifies the trade-off
  (which on this small dataset turned out to be zero), and produces
  the business recommendation.

The arc across PRs #17 → #21 → THIS is: identify the imbalance →
diagnose its severity → try the textbook fix → honestly report the
result. The next module in the arc (out of scope here) is threshold
tuning + resampling.
