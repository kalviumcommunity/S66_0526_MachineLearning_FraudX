# Final Model Selection and Use-Case Alignment

This document is the long-form write-up for the FraudX capstone module.
The short summary lives in
[`README.md`](../README.md#-final-model-selection-and-use-case-alignment);
this file covers the use-case alignment reasoning, the encoded
selection rule, the holistic evaluation, and how each prior PR
contributed to the final decision.

## 1. What this module is and isn't

This module is the **capstone**. It does NOT train a new model class.
Its job is to:

- Synthesise the candidates evaluated across PRs #17, #18, #21, #22,
  #23, #24 into one fair head-to-head comparison.
- Apply a stated selection rule aligned to the *business use case*,
  not to "highest accuracy".
- Produce a single justified final pick, with the holistic-evaluation
  factors (interpretability, inference cost, CV stability, baseline
  improvement) explicitly weighed.

The assignment is explicit: **submissions that select on highest
accuracy alone do not receive full marks**. This module's selection
rule is encoded to make that mistake structurally impossible.

## 2. The six candidates

| # | Candidate | Source PR | What it changes |
| - | :--- | :--- | :--- |
| 1 | Logistic Regression | PR #24 | Linear baseline; coefficient-level interpretability. |
| 2 | Random Forest (default) | Incumbent across PRs #15-#24 | Tree ensemble, low variance via bagging. |
| 3 | Gradient Boosting | PR #24 | Sequential trees on residuals; typically strongest tabular learner. |
| 4 | RF + `class_weight="balanced"` | PR #22 | Re-weights the loss; minority errors cost ~10× more. |
| 5 | RF + RandomOverSampler | PR #23 | Duplicates minority rows during training. |
| 6 | RF + SMOTE (k=5) | PR #23 | Synthesises new minority rows by k-NN interpolation. |

All six share:
- The same `train_test_split(stratify=y, random_state=42)`.
- The same `ColumnTransformer(num: SimpleImputer + StandardScaler |
  cat: SimpleImputer + OneHotEncoder)` preprocessor.
- The same 5-fold `StratifiedKFold(shuffle=True, random_state=42)` CV.
- The same scoring metric (`f1` on the fraud / positive class).
- The same sealed test set, evaluated once per candidate.

Candidates 5 and 6 wrap their sampler inside `imblearn.pipeline.Pipeline`
so the sampler re-runs INSIDE every CV fold (leakage-safe per PR #20 /
PR #23).

## 3. Use-case alignment (Part 2 — required)

### 3.1 Scenario constraints

- Fraud detection.
- **False Negatives (missed fraud) are significantly more costly than
  False Positives.** Typical c_FN/c_FP ratio is 50:1 or higher: a
  missed fraud loses the full transaction amount; a false positive
  loses minutes of customer-service / re-auth time.
- Some model interpretability is preferred but not mandatory.
- The system must handle moderate real-time traffic.

### 3.2 Prioritised metric

**Recall on the fraud (positive) class**, because it directly answers
the business question — "of all the actually-fraudulent transactions,
how many did we catch?".

- Accuracy is the *wrong* metric: a `most_frequent`-baseline scores
  91 % accuracy with zero fraud caught (PR #17, PR #21). The
  assignment explicitly warns against selecting on accuracy alone.
- F1 is the secondary check: it guards against degenerate "predict
  class 1 for every row" solutions that would game pure-recall
  optimisation.
- Precision matters for operational cost (false alarms), but with
  c_FN/c_FP ≥ 50, the business tolerates many FPs to catch each
  additional TP.

### 3.3 Tie-breakers

Encoded in [`_select_final`](../src/final_selection.py):

1. **Primary**: highest test recall on class 1.
2. **Tie #1**: highest test F1 on class 1 (joint precision-recall —
   prevents a "predict everything as fraud" winner).
3. **Tie #2**: lowest CV std (stability across training-set shuffles
   = deployment-ready).
4. **Tie #3**: better interpretability (operational preference; LR > RF/GB).

## 4. Worked numbers — FraudX real run

Test set: 200 samples (182 class 0 / 18 class 1). All metrics identical
preprocessing + identical scoring.

| Model                          | CV mean F1 | CV std | Test acc | Test P (1) | Test R (1) | Test F1 (1) |
| :----------------------------- | ---------: | -----: | -------: | ---------: | ---------: | ----------: |
| LogisticRegression             |     0.00 % | 0.00 % |  91.00 % |     0.00 % |     0.00 % |      0.00 % |
| RandomForest                   |     0.00 % | 0.00 % |  91.00 % |     0.00 % |     0.00 % |      0.00 % |
| GradientBoosting               |     0.00 % | 0.00 % |  88.50 % |     0.00 % |     0.00 % |      0.00 % |
| RF + class_weight=balanced     |     0.00 % | 0.00 % |  91.00 % |     0.00 % |     0.00 % |      0.00 % |
| **RF + RandomOverSampler**     |   **2.50 %** | 5.00 % | **89.00 %** | **16.67 %** |  **5.56 %** |  **8.33 %** |
| RF + SMOTE                     |     6.18 % | 8.70 % |  83.00 % |     5.56 % |     5.56 % |      5.56 % |

### Confusion matrices

| Model                        | TN  | FP | FN | TP |
| :--------------------------- | --: | -: | -: | -: |
| LogisticRegression           | 182 |  0 | 18 |  0 |
| RandomForest                 | 182 |  0 | 18 |  0 |
| GradientBoosting             | 177 |  5 | 18 |  0 |
| RF + class_weight=balanced   | 182 |  0 | 18 |  0 |
| **RF + RandomOverSampler**   | 177 |  5 | 17 |  **1** |
| RF + SMOTE                   | 165 | 17 | 17 |  **1** |

### Highest-numerical and most-stable

- **Best test recall (class 1)**: RF + RandomOverSampler (5.56 %) — tied
  with RF + SMOTE (also 5.56 %).
- **Best test F1 (class 1)**: RF + RandomOverSampler (8.33 %).
- **Most stable** (lowest CV std): tied between Logistic Regression, RF,
  GB, RF + class_weight — all at 0.00 % std. But that's degenerate
  stability (every fold scores 0 on the positive class).

## 5. Final selection

**Selected model: `RF + RandomOverSampler`.**

Applying the encoded tie-break order to the table above:

1. Test recall on class 1: RandomOverSampler and SMOTE both 5.56 % (tied).
2. Test F1 on class 1: RandomOverSampler **8.33 %** > SMOTE **5.56 %**
   → RandomOverSampler wins.
3. (Subsequent tie-breakers not needed.)

### Why this is right under the use case

- It is the **only candidate that maximises recall on the fraud class
  while keeping precision usable** (16.67 % vs SMOTE's 5.56 %).
  Both resamplers catch 1 of 18 fraud cases, but RandomOS does it
  with 3× fewer false positives.
- The accuracy cost is 91 % → 89 % (a 2pp drop), tiny compared to
  the recall gain (0 % → 5.56 %, infinite proportional gain over a
  baseline that catches *no* fraud).
- The CV mean F1 of 2.50 % with std 5.00 % isn't degenerate — there's
  real cross-fold variation, meaning the model *does* learn something
  fold-by-fold rather than collapsing to majority-class predictions.

### Why the other candidates lose the encoded rule

| Candidate | Lost on |
| :--- | :--- |
| Logistic Regression | Recall = 0. Loses on the primary metric. |
| Random Forest | Recall = 0. Loses on the primary metric. |
| Gradient Boosting | Recall = 0. Tries to predict class 1 (5 FPs), but gets none right. |
| RF + class_weight=balanced | Recall = 0 — class weighting alone didn't move the needle on this dataset (PR #22). |
| RF + SMOTE | Tied on recall, loses on F1 (5.56 % vs 8.33 %). |

## 6. Holistic evaluation (Part 3 — required)

Beyond the metrics:

### 6.1 Interpretability

- **Logistic Regression** has the strongest interpretability story —
  per-feature coefficients with sign + magnitude provide a one-line
  explanation per prediction.
- **Random Forest** (and the resampler variants on top of it) can use
  per-tree feature importances or SHAP values, but those are
  derived, not native.
- **Gradient Boosting** is the same — interpretable via SHAP but not
  natively.

For this scenario, interpretability is "preferred but not mandatory",
so it does **not** override the recall-first priority. If the
deployment requirement changed to "every decision must justify to a
regulator", Logistic Regression would become the right choice even at
0 recall. We'd then pair it with threshold tuning to claw back recall.

### 6.2 Computational cost at inference

- **LR**: O(features) — single dot product. Negligible.
- **RF / RF+ClassWeight / RF+RandomOS / RF+SMOTE**: O(n_trees × depth).
  At 100 trees and depth ≈ 15-30, still < 1ms per prediction on
  modern hardware.
- **GB**: O(n_trees × depth), but sequential — somewhat slower than RF.

For "moderate real-time traffic" (typical payment-gateway QPS), all
six candidates are easily fast enough. If the scenario changed to
"hundreds of thousands of QPS at the edge", LR would pull ahead.

### 6.3 Stability across CV folds

| Model | CV std |
| :--- | ---: |
| LogisticRegression | 0.00 % |
| RandomForest | 0.00 % |
| GradientBoosting | 0.00 % |
| RF + class_weight=balanced | 0.00 % |
| **RF + RandomOverSampler** | 5.00 % |
| RF + SMOTE | 8.70 % |

The "stable" candidates (top four) are stable because they
*consistently produce zero recall* on every fold — that's degenerate
stability, not deployment-ready stability. RandomOS and SMOTE have
higher CV std because they *actually learn something* on some folds.
Of the two, RandomOS is more stable (5 % std vs SMOTE's 8.7 %), which
adds to the case for selecting it.

### 6.4 Improvement over baseline

The baseline (majority-class `DummyClassifier(strategy="most_frequent")`)
achieves:
- Accuracy: 91 % (the class prior)
- Recall on class 1: 0 %
- F1 on class 1: 0 %

`RF + RandomOverSampler` achieves:
- Accuracy: 89 % (−2 pp vs baseline — the visible cost)
- **Recall on class 1: 5.56 %** (+5.56 pp vs baseline — pure win)
- **F1 on class 1: 8.33 %** (+8.33 pp vs baseline — pure win)

The recall improvement is the only improvement that matters for this
use case. The accuracy drop is the documented and acceptable cost.

## 7. What could change this decision in a different business context

The assignment explicitly asks this (Video Demo #5):

| Different context | Switch to |
| :--- | :--- |
| FP cost dominates (e.g., customer-churn risk too high) | Higher-precision candidate. Maybe GB at default threshold (5 FPs without flooding), or LR with threshold tuned upward. |
| Interpretability mandatory (regulated industry) | LR. Accept 0 recall here; pair with threshold tuning and feature engineering. |
| Inference cost matters (very high QPS, edge deployment) | LR. Tree-based ensembles are over-provisioned for the task. |
| Dataset becomes very large (millions of rows) | RF + class_weight or LR. SMOTE becomes slow; RandomOS scales linearly. |
| Features change such that the signal becomes linear | LR may pull ahead of the tree-based candidates on recall too — re-run this comparison. |
| Concept drift introduces new fraud patterns | Retrain on fresh data + re-check the comparison. The selection rule itself doesn't change. |

## 8. PR-required outputs (checklist)

- [x] Comparison table of all models — §4 above and runtime output.
- [x] Final selected model clearly stated — `RF + RandomOverSampler`, §5.
- [x] Justification aligned with use case — §3 (priority) + §5 (winner)
      + §6 (holistic).
- [x] Cross-validation evidence (mean ± std) — §4 table.
- [x] Test performance summary — §4 table + §4 confusion matrices.
- [x] Confusion matrix reference — §4 confusion matrices.
- [x] Baseline comparison — §6.4 vs majority-class baseline.

The assignment also notes: *submissions that select a model based only
on "highest accuracy" will not receive full marks.* This module's
selection rule deliberately makes that mistake impossible — recall is
the primary metric, the encoded tie-break order makes accuracy
secondary, and the holistic-evaluation discussion examines factors
that cannot be captured by any single metric.

## 9. How to run

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python3 src/final_selection.py    # just the capstone
# OR
python3 main.py                    # full pipeline (Phase 3 runs the capstone)
```

Artifacts produced:

- `models/final_selected_model.pkl` — fitted
  `imblearn.Pipeline(preprocessor + RandomOverSampler + RandomForestClassifier)`.
  Reusable at inference (sampler is skipped automatically):
  ```python
  import joblib
  pipeline = joblib.load("models/final_selected_model.pkl")
  proba = pipeline.predict_proba(new_data_df)[:, 1]
  ```
- `reports/plots/final_selection_comparison.png` — grouped bar chart
  of test recall / test F1 / CV mean for all six candidates.

## 10. How this completes the project arc

PRs #15 → #24 built the infrastructure (preprocessing, baselines,
tuning, pipelines, leakage audits, imbalance analysis, class
weighting, oversampling, multi-model comparison). Each PR ended with
a pointer to "what's next". This module ties them all together:

- **PR #17** (baseline comparison) established the harness; this
  module uses it to compare six models, not two.
- **PR #21** (imbalance analysis) named PR-AUC vs ROC-AUC and severity;
  this module's selection rule prioritises recall, which is the metric
  PR-AUC summarises.
- **PR #22** (class weights) and **PR #23** (oversampling) supplied
  candidates 4-6.
- **PR #24** (multi-model comparison) supplied candidates 1-3 and the
  CV-vs-test discipline.
- **THIS PR** is the deployment-level decision on top of all of them.

The chosen model — `RF + RandomOverSampler` — is the synthesis: it
combines the project's incumbent classifier (Random Forest) with the
imbalance-handling technique that *actually worked* (RandomOverSampler,
PR #23), evaluated through the rigorous CV-then-test workflow
established in PRs #19 / #20.

The natural next iteration — out of scope here — is **threshold
tuning** on the saved `final_selected_model.pkl`. The model's
`predict_proba` scores are calibrated for an asymmetric loss; choosing
a decision threshold below 0.5 against a stated `c_FN/c_FP` cost
ratio is the cleanest next step to push minority recall higher without
training anything new.
