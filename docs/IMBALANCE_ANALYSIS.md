# Class Imbalance Analysis

This document is the long-form write-up for the FraudX class-imbalance
diagnostic step. The short summary lives in
[`README.md`](../README.md#-class-imbalance-analysis);
this file goes deeper into severity classification, the metric battery
(including PR-AUC vs ROC-AUC), the assignment's Part 3 comparison
answers, and the four mandatory scenario answers.

## 1. What class imbalance is

A classification problem has class imbalance when one target class
appears far more often than the others. The natural failure modes:

- **Naive accuracy is misleading.** Always predicting the majority
  class gives an "accuracy" equal to the majority share, with zero
  predictive value for the minority class.
- **The training signal is dominated by the majority class.** Gradient
  / split-impurity / loss objectives weight every example equally by
  default, so the optimisation puts most of its budget into getting
  majority-class examples right and treats minority-class errors as
  rounding noise.
- **Some ranking metrics get distorted.** ROC-AUC averages over
  thresholds that include "predict everything as class 0", and on
  imbalanced data the true-negative rate dominates the ROC curve. PR-AUC
  (average precision) does not have this problem because it only looks
  at the positive class.

This module is purely **diagnostic** — it does not try to fix any of
the above. The assignment is explicit that we focus on diagnosis and
evaluation discipline first; resampling, class weighting, and threshold
tuning are out of scope here. (Where they belong in the project
roadmap is briefly noted in §10.)

## 2. Severity rubric

I classify imbalance severity by the share of the **minority** class:

| Minority share | Severity |
| :--- | :--- |
| ≥ 40 %       | mild     |
| 10 % – 40 %  | moderate |
| < 10 %       | severe   |

These thresholds are conventional rules of thumb (Provost & Fawcett,
*Data Science for Business*; Krawczyk, 2016). They are stated in the
module (`classify_imbalance_severity` in
[`src/imbalance_analysis.py`](../src/imbalance_analysis.py)) so any future
reader can disagree with the boundaries explicitly rather than
implicitly.

## 3. Worked numbers — FraudX class distribution

| Split        | n   | class 0 (legit) | class 1 (fraud) | minority share | severity |
| :----------- | --: | --------------: | --------------: | -------------: | :------- |
| full dataset | 1000 |  909 (90.9 %)  |   91 ( 9.1 %)  |        9.10 %  | **severe** |
| train (80 %) |  800 |  727 (90.9 %)  |   73 ( 9.1 %)  |        9.12 %  | severe   |
| test  (20 %) |  200 |  182 (91.0 %)  |   18 ( 9.0 %)  |        9.00 %  | severe   |

`train_test_split(stratify=y)` preserved the minority share inside one
percentage point across splits (compare 9.10 / 9.12 / 9.00 — see
scenario answer §8.4 for why this matters).

### Why this imbalance is problematic (3-4 lines, per assignment)

- A naive classifier can score **~91 % accuracy** by always predicting
  class 0 yet catch **zero** fraud cases — useless for the real-world
  objective.
- Gradient / split-impurity objectives are dominated by class-0
  examples; the optimisation does not naturally weight fraud detection
  unless we explicitly intervene (class weight, resampling, or
  threshold tuning).
- Standard ranking metrics (ROC-AUC) can look inflated because the
  true-negative rate dominates the curve when negatives are abundant.
  **PR-AUC is the more honest single-number summary in this regime.**

## 4. Majority-class baseline (`DummyClassifier(strategy="most_frequent")`)

Predicts class 0 unconditionally.

| Metric          | Value      |
| :-------------- | ---------- |
| Accuracy        | **91.00 %** |
| Precision (1)   | 0.00 %     |
| Recall (1)      | 0.00 %     |
| F1 (1)          | 0.00 %     |
| PR-AUC          | 9.00 % (= the class prior) |
| ROC-AUC         | 50.00 % (chance-level — degenerate for a constant predictor) |

Confusion matrix:

|              | predicted 0 | predicted 1 |
| ------------ | ----------: | ----------: |
| **actual 0** | TN = 182    | FP = 0      |
| **actual 1** | FN = 18     | TP = 0      |

**Why this baseline can appear strong.** 90.9 % of rows belong to the
majority class. Always predicting the majority class is correct on
every one of those rows, which buys you an accuracy that *looks* like
a competent classifier on a dashboard.

**Why it is practically useless.** Every minority-class row is
misclassified. In a fraud-detection deployment, that means every fraud
transaction goes through. The business cost of false negatives is
exactly what the project exists to reduce, and the baseline optimises
*against* it.

## 5. Standard model (`RandomForestClassifier`, sklearn defaults)

Default-threshold predictions on the test set:

| Metric          | Value       |
| :-------------- | ----------- |
| Accuracy        | **91.00 %** |
| Precision (1)   | 0.00 %      |
| Recall (1)      | 0.00 %      |
| F1 (1)          | 0.00 %      |
| **PR-AUC**      | **10.74 %** |
| **ROC-AUC**     | **46.38 %** |

Confusion matrix:

|              | predicted 0 | predicted 1 |
| ------------ | ----------: | ----------: |
| **actual 0** | TN = 182    | FP = 0      |
| **actual 1** | FN = 18     | TP = 0      |

At the **default 0.5 threshold** the RF predictions are *identical* to
the baseline's — it predicts class 0 for every test row. Accuracy,
precision, recall, and F1 are therefore all the same as the baseline's.

But the ranking-based metrics tell a more interesting story:

- **PR-AUC = 10.74 %** — slightly above the 9.00 % class prior. There
  is *some* learning signal in the ranking, even though the default
  threshold doesn't surface it. A threshold-tuning iteration could
  pick this signal up.
- **ROC-AUC = 46.38 %** — *below* chance. On this particular small,
  severely imbalanced test set the RF ranks negatives above positives
  slightly more often than chance, which is consistent with an
  underfit minority class. The ROC curve is dominated by the
  true-negative rate, so the metric doesn't penalise this very hard.

The disagreement between PR-AUC (says "tiny signal") and ROC-AUC (says
"slightly worse than chance") is itself the lesson: **on severely
imbalanced data, neither metric alone tells the whole story; PR-AUC is
the more honest single number, and looking at both is the right
discipline.**

The heatmap visualisation (`reports/plots/imbalance_confusion_matrices.png`)
makes the *identical* confusion matrices side-by-side easy to see at a
glance.

## 6. Side-by-side comparison

| Model                       | Accuracy | Precision (1) | Recall (1) | F1 (1) | PR-AUC  | ROC-AUC |
| :-------------------------- | -------: | ------------: | ---------: | -----: | ------: | ------: |
| Baseline (most_frequent)    |  91.00 % |        0.00 % |     0.00 % | 0.00 % |  9.00 % | 50.00 % |
| RandomForestClassifier      |  91.00 % |        0.00 % |     0.00 % | 0.00 % | 10.74 % | 46.38 % |

## 7. Part 3 — Comparison answers (required)

### 7.1 Does accuracy reflect minority-class performance?

**No.** Both the baseline and the model report **91.00 %** accuracy.
Both also report **0.00 %** recall on the minority class (no fraud
caught). Accuracy is a weighted average over the two classes, and with
90.9 % of rows being class 0, the average is dominated by class-0
behaviour. Accuracy in this regime answers "did the model predict the
majority class correctly?" — which the trivial baseline can also do.
It tells you nothing about the only thing the project actually cares
about (catching fraud).

### 7.2 Is recall high or low for the minority class?

**Very low — 0.00 %.** The model misses every fraud case in the test
set. Recall on class 1 is the metric that maps directly to the
business question "of all the actually-fraudulent transactions, how
many did we catch?", and the answer here is "none." Compared to the
9.10 % class share, the model is hitting zero of its fraud-detection
mandate at the default threshold.

### 7.3 Is precision high or low?

**Also 0.00 %**, but in a slightly misleading way. Precision = TP /
(TP + FP). With TP = 0 and FP = 0, the convention is to define
precision as 0 (sklearn's `zero_division=0`). The interesting question
isn't whether precision is "high" or "low" in this run — it's that
**precision and recall are jointly degenerate** because the model
makes zero positive predictions. The precision-recall trade-off only
becomes meaningful once we have at least one positive prediction (via
threshold tuning, class weighting, or resampling — out of scope here).

### 7.4 Which metric best captures the true usefulness of the model?

**PR-AUC (average precision)** is the right single-number summary
under severe imbalance. Why:

- It only depends on how the model ranks positives among the top of
  the predicted score distribution. It is not distorted by the
  abundance of true negatives.
- It can distinguish "model has *some* ranking signal" (PR-AUC = 10.74 %,
  above the 9.00 % class prior) from "model is exactly the baseline"
  (PR-AUC = 9.00 %), even when default-threshold metrics (accuracy /
  precision / recall / F1) collapse to identical numbers.
- It directly informs the threshold-tuning conversation: if PR-AUC > class
  prior, a tuned threshold can produce a working model; if PR-AUC ≈ class
  prior, no threshold will help and the search space needs to change.

F1 at the default threshold is a reasonable second choice, but only
once you have non-zero predictions. ROC-AUC is generally the worst
single number under severe imbalance — it can look unproblematic when
the model is in fact useless.

### 7.5 Does the model meaningfully outperform the majority baseline?

**No, not at the default threshold.** The two models produce
*identical* confusion matrices on this test set. Accuracy, precision,
recall, and F1 are all the same. The only metric that distinguishes
them is PR-AUC (10.74 % vs 9.00 %), which says the RF has tiny but
genuine ranking signal — not enough to ship.

That said, "meaningfully outperform" requires a definition. By the
right definition (catches at least some fraud at an acceptable
false-positive rate), the RF does **not** outperform the baseline yet.
The path forward — class weighting, threshold tuning, resampling — is
the scope of separate modules (see [PR #18](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/18)).

## 8. Scenario answers (mandatory)

### 8.1 Why does class imbalance naturally bias models toward the majority class?

Because most learning algorithms minimise an objective that treats
every training example equally. With class 0 = 90.9 % of examples,
~91 % of the optimisation pressure is on getting class-0 predictions
right. A learner that picks an "always class 0" rule satisfies 90.9 %
of that pressure perfectly, and the marginal benefit of getting a
minority-class example right is tiny relative to the marginal cost
(losing one majority-class prediction). Tree-based learners are also
biased by the impurity criterion: a split that helps the minority
class but slightly hurts a much larger majority-class population
will not look attractive on average. The result: models default to
the majority class unless we explicitly tell them not to (class
weighting, resampling, cost-sensitive thresholds).

### 8.2 Why can a model achieve high accuracy but still be useless?

Because accuracy is a class-weighted average and the weights are
the class proportions. On a 91 / 9 dataset, accuracy is 91 % the
moment you predict everything as the majority class — even if you
never catch a single minority case. The "high accuracy" reflects how
the data is distributed, not how the model performs on the task that
matters. The classic phrasing: "If 99 % of patients are healthy,
diagnosing every patient as healthy is 99 % accurate and lethal."
The right metric is whichever one maps to the cost the project
actually pays for being wrong — for fraud detection, that's recall
on the fraud class (combined with a tolerable false-positive rate),
which is what F1 and PR-AUC summarise.

### 8.3 In fraud detection, which is worse — false positives or false negatives?

**False negatives are usually worse**, but the right framing is asymmetric
cost rather than a one-size-fits-all ranking:

- **False negative (FN, missed fraud)**: a real fraudulent transaction
  passes through. The bank eats the fraud loss. On a typical
  transaction this is the entire amount of the transaction.
- **False positive (FP, blocked legitimate transaction)**: a real
  customer is inconvenienced — possibly seriously, and possibly
  enough to lose them as a customer.

The expected costs are *very* different in magnitude (a fraud loss
can be hundreds to thousands of dollars; an inconvenienced customer
costs the friction of a phone call or app re-auth) but the *frequency*
is very different too (you have many more legitimate transactions
than fraudulent ones). The right way to set the threshold is to
explicitly model `Cost = c_FN · FN + c_FP · FP` and pick the
operating point on the precision-recall curve that minimises expected
cost — which is exactly the workflow PR-AUC is designed to support.

### 8.4 Why is stratified splitting necessary in imbalanced datasets?

Because a random split can, by chance, push too many or too few
minority-class examples into one half. On this dataset (91 minority
examples in 1000 rows total), a random 80/20 split might give the
test set 12 minority examples instead of 18, or 25 instead of 18.
Two consequences:

- **CV variance explodes.** The reported metrics swing wildly across
  folds because the minority class is small and a different shuffle
  produces a different fold composition.
- **The metric you optimise during training may not match the metric
  you ship.** If training-side CV gets lucky on minority count and
  test-side gets unlucky, a model that looks good in CV will look
  worse in production.

Stratified splitting fixes this by enforcing that each split contains
the *same proportion* of each class as the full dataset. On FraudX
the stratified split landed train at 9.12 %, test at 9.00 % minority
— inside one percentage point of the full 9.10 %. With a random split
you cannot make that guarantee on a single sample.

### 8.5 When would PR-AUC be more informative than ROC-AUC?

When the positive class is rare AND the cost of false negatives
matters. The structural reason:

- **ROC-AUC** is the probability that a randomly drawn positive is
  ranked above a randomly drawn negative. On a 91 / 9 dataset, that's
  a *very* asymmetric pairing — 9 % of the candidate pairs include a
  positive, 91 % don't. The metric averages over thresholds that
  include "predict everything class 0", where the model gets a free
  pass because the true-negative rate is 100 %. ROC-AUC can stay
  near 0.9 even when the model is missing every positive.
- **PR-AUC** (average precision) is the area under the precision-recall
  curve. Precision = TP / (TP + FP). The metric only cares about
  positive predictions, so it is not free-ridden by true negatives.
  Under severe imbalance, PR-AUC ≈ class prior when the model has no
  signal and rises only when the model is genuinely ranking positives
  higher than negatives.

On this FraudX run, the comparison is concrete:
- ROC-AUC: 46.38 % — barely below chance, doesn't sound the alarm.
- PR-AUC: 10.74 % — barely above the 9.00 % class prior, makes the
  signal-vs-no-signal question quantitative.

Anywhere the positive class is < 10 % of the data — fraud, medical
diagnosis of rare conditions, manufacturing-defect detection,
intrusion detection — PR-AUC is the right primary metric.

## 9. How to run

```bash
export PYTHONPATH=.
python3 src/imbalance_analysis.py     # just the analysis
# OR
python3 main.py                       # full pipeline (Phase 3 runs the analysis)
```

Artifacts produced:

- `models/imbalance_standard_model.pkl` — fitted `Pipeline(preprocessor + RandomForestClassifier)`.
- `reports/plots/imbalance_confusion_matrices.png` — side-by-side heatmap.

## 10. How this complements the rest of the project

- [PR #15](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/15)
  switched to `MinMaxScaler` (preprocessing concerns).
- [PR #17](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/17)
  introduced the **baseline-comparison harness** (the infrastructure
  this module builds on top of).
- [PR #18](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/18)
  ran `RandomizedSearchCV` and is the natural home for the
  threshold-tuning / class-weighting follow-up that this module's
  PR-AUC analysis motivates.
- [PR #19](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/19)
  introduced the canonical `Pipeline(ColumnTransformer + classifier)`
  pattern (reused here).
- [PR #20](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/20)
  layered four leakage types and produced the Pipeline-correction
  audit — orthogonal concern, also covered.

This module's contribution is the **interpretive layer**: how to read
imbalanced-classification metrics, why accuracy lies on this data,
and which single number to track. The complementary modules supply
the infrastructure; this one supplies the discipline.
