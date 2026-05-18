# Baseline Model + Class-Imbalance Comparison

This document is the long-form write-up for the FraudX baseline-comparison
step. The short summary lives in the main
[`README.md`](../README.md#-baseline-comparison-and-class-imbalance-evaluation);
this file goes deeper into the *why* and supplies the worked numbers.

## 1. What a baseline is and why it matters

A **baseline classifier** is a deliberately stupid predictor — no learned
patterns, just a fixed rule. Its job is not to be good; its job is to
**anchor evaluation**. The right question after training a model isn't "did
my accuracy go up?" — it's "did my model beat a trivial predictor by a
margin that's worth the complexity?".

On heavily imbalanced data, this matters even more, because accuracy
itself becomes misleading. If 91 % of transactions are non-fraud, a model
that *always* predicts non-fraud is right 91 % of the time — and is
completely useless for the actual task (catching fraud). Comparing against
a `most_frequent` baseline makes that trap visible in one line of output.

## 2. The two strategies in this module

| Strategy        | What it does                                              | Why we include it                                                                  |
| --------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `most_frequent` | Always predicts the majority class from training data.    | Maximises accuracy on imbalanced data **without learning anything** — the canonical "is my model meaningful?" test. |
| `stratified`    | Samples predictions according to the training class prior. | Provides a non-trivial chance baseline so a model that just predicts class 0 every time doesn't look like a "stratified-baseline-beating" model. |

Both baselines are constructed via `sklearn.dummy.DummyClassifier` and
fitted in [`src/baseline.py`](../src/baseline.py).

## 3. Leakage discipline

The assignment's "Important Guidelines" forbid:
- Fitting the baseline on the full dataset before splitting.
- Comparing different metrics between baseline and main model.
- Modifying test data to make the baseline look better.
- Deriving heuristic rules from anywhere except training data.

All four are enforced here:

| Rule                                                | Where it's enforced                                                                 |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Split before fitting                                | `src/data_preprocessing.py::split_data` runs `train_test_split` first.              |
| Baseline fit on training data only                  | `src/baseline.py::fit_baselines` accepts `X_train, y_train` — never the full `X`.   |
| Identical X_test / y_test for all classifiers       | `src/comparison.py::run_baseline_comparison` builds the test set once and asserts the same support per class across every result. |
| Identical evaluation metrics                        | `src/evaluate.py::evaluate_detailed` is called for the baselines AND the main model. |

## 4. Worked example — actual FraudX numbers

Test set size: **200 samples** (182 class 0 / 18 class 1 — preserving the
~91/9 ratio via stratified split).

| Model                       | Accuracy | Balanced Accuracy | P (class 0) | R (class 0) | F1 (class 0) | P (class 1) | R (class 1) | F1 (class 1) |
| --------------------------- | -------: | ----------------: | ----------: | ----------: | -----------: | ----------: | ----------: | -----------: |
| `baseline_most_frequent`    |   91.0 % |            50.0 % |      91.0 % |     100.0 % |       95.3 % |       0.0 % |       0.0 % |        0.0 % |
| `baseline_stratified`       |   83.0 % |            45.6 % |      90.2 % |      91.2 % |       90.7 % |       0.0 % |       0.0 % |        0.0 % |
| `RandomForestClassifier`    |   91.0 % |            50.0 % |      91.0 % |     100.0 % |       95.3 % |       0.0 % |       0.0 % |        0.0 % |

Confusion matrix for the **main model** (`RandomForestClassifier`):

|              | predicted 0 | predicted 1 |
| ------------ | ----------: | ----------: |
| **actual 0** |         182 |           0 |
| **actual 1** |          18 |           0 |

That is to say: the RandomForest predicts class 0 for every single test
sample. Its accuracy of 91.0 % is structurally identical to the
`most_frequent` baseline because **it has effectively become** the
`most_frequent` baseline.

## 5. Improvement over baseline

| Metric                                   | Δ (Main − `most_frequent`) | Reading                                                                       |
| ---------------------------------------- | -------------------------: | ----------------------------------------------------------------------------- |
| accuracy                                 |                      0.0 % | Identical — accuracy is uninformative on this data.                            |
| balanced accuracy                        |                      0.0 % | Both score 50 % — chance-level when both classes are weighted equally.        |
| recall on class 1 (fraud catch rate)     |                      0.0 % | Neither catches any fraud cases.                                              |
| F1 on class 1                            |                      0.0 % | Same.                                                                         |

**Verdict.** The current trained model does **not** meaningfully beat the
majority-class baseline on the minority-class metrics that matter for fraud
detection. Its 91 % accuracy is a class-balance artefact, not a learning
signal. This is precisely the failure mode the baseline comparison is
designed to expose.

This is genuine, useful engineering signal: it tells us the next
preprocessing iteration should focus on the imbalance — class weighting
in the `RandomForestClassifier` (`class_weight="balanced"`), resampling
(SMOTE or random under-sampling), or a cost-sensitive decision threshold.
A small synthetic dataset of 1000 rows with default hyperparameters
*should not* be expected to learn fraud signals — but we now have a
concrete, principled way to measure progress whenever those changes
land.

## 6. Scenario question answers (for Part 2 video)

> *"You are working on a churn prediction dataset where 88 % of customers do
> NOT churn and 12 % churn. Your majority-class baseline achieves: Accuracy 88 %,
> Recall (churn) 0.00, F1 (churn) 0.00. Your trained model achieves Accuracy
> 90 %, Recall (churn) 0.42, F1 (churn) 0.48."*

### 6.1 Why is the 88 % baseline accuracy misleading?

Because it's achieved by a predictor that never identifies a churning
customer. The classifier always says "won't churn"; it gets every churn
case wrong, and it still ends up with 88 % accuracy because 88 % of the
data really doesn't churn. Accuracy here is measuring the **class
prior**, not the model's ability to detect churn. For a business that
cares about predicting churn (the whole point of the project), this
classifier is useless — yet accuracy alone would call it "great".

### 6.2 Is the trained model meaningfully better than the baseline? Justify using metrics.

Yes — and the right metrics to look at are **recall and F1 on the churn
class**, not accuracy.

| Metric          | Baseline | Trained Model | Delta              |
| --------------- | -------: | ------------: | ------------------ |
| Accuracy        |     88 % |          90 % | + 2 percentage points (small) |
| Recall (churn)  |     0.00 |          0.42 | **+ 0.42 absolute** |
| F1 (churn)      |     0.00 |          0.48 | **+ 0.48 absolute** |

Accuracy moved only +2 points, but recall on the churn class went from
*zero* to 0.42 — meaning the model now actually catches 42 % of churning
customers (vs none for the baseline). F1 of 0.48 says precision and
recall on the churn class are balanced and non-trivial. **That** is what
"meaningfully better" looks like under imbalance.

### 6.3 Which metric is more important in this scenario, and why?

**Recall on the churn (minority) class** is the most important single
metric, with F1 on the churn class as a close second.

- **Recall** answers "of all the customers who actually churned, how many
  did we catch?" — which is exactly what a retention team needs to act.
- **F1** keeps recall honest: it discourages a "predict churn for
  everyone" hack that would max recall but tank precision (and waste
  retention spend on customers who weren't going to leave anyway).

Plain accuracy is the **worst** metric to optimise on imbalanced data
because it's dominated by the majority class. Even balanced accuracy is
preferable, but minority-class recall / F1 are what stakeholders actually
care about.

### 6.4 If the model had 89 % accuracy but recall of 0.05, would it still be acceptable? Explain.

**No.** A model with 89 % accuracy and 0.05 recall on the churn class is
*essentially* the majority-class baseline. Recall of 0.05 means it
correctly identifies only 5 % of customers who actually churned — i.e.
it misses 95 % of the people the business needs to retain. A 1 percentage
point improvement on accuracy over an 88 % baseline is a rounding error;
it does not justify shipping a model. The fact that overall accuracy is
higher is irrelevant — it just reflects that the model is slightly
better at predicting the majority class, which the business doesn't need
help with.

The right thing to do here is exactly what we did with the FraudX
RandomForest: name the failure honestly and iterate on the imbalance.

## 7. What this enables next

With the comparison harness in place, every future model change can be
evaluated against the same baseline using the same metrics. Concretely,
this unblocks:

1. **Class weighting**: re-run with
   `RandomForestClassifier(class_weight="balanced", ...)`, re-run
   `comparison.py`, observe the delta in class-1 recall / F1.
2. **Resampling**: SMOTE / random undersampling on the training half
   only (no leakage), re-run, observe.
3. **Threshold tuning**: lower the decision threshold from 0.5, observe
   the precision-recall trade-off.
4. **Better features / models**: same comparison harness, different
   model class.

In every case, the relevant question is "did class-1 F1 actually move?"
— not "did accuracy go up?".

## 8. How to run

```bash
export PYTHONPATH=.
python3 src/comparison.py     # just the comparison
# OR
python3 main.py               # full pipeline: train → predict → comparison
```

Artifacts produced:

- `models/baseline_most_frequent.pkl` — fitted `DummyClassifier(strategy="most_frequent")`
- `models/baseline_stratified.pkl`    — fitted `DummyClassifier(strategy="stratified", random_state=42)`

Both can be loaded for offline analysis via:

```python
from src.baseline import load_baseline
clf = load_baseline("most_frequent")  # or "stratified"
```
