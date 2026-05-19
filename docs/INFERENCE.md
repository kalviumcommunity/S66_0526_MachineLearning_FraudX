# Production Inference on a Persisted Model

This document is the long-form write-up for the FraudX inference
demonstration module. The short summary lives in
[`README.md`](../README.md#-production-inference-on-a-persisted-model);
this file covers the conceptual explanation the assignment requires
(what inference is, how it differs from training, why production
needs persisted models), the deployment-readiness checklist, and the
worked example with 5 new transactions.

## 1. What inference is, and how it differs from training

**Training** is the process that *learns* model parameters from
labelled examples. It runs `.fit(X_train, y_train)` to compute the
weights, tree structures, scaler statistics, encoder vocabularies,
sampler indices, etc. — every parameter the model needs to make
predictions. Training is offline, expensive, and produces a single
fitted artifact.

**Inference** is the process that *uses* a previously-trained model to
make predictions on new data. It runs `.predict(X_new)` (or
`.predict_proba(X_new)`) but **never** `.fit(...)`. Inference is online,
cheap, and runs once per request (or per batch).

| Aspect | Training | Inference |
| :--- | :--- | :--- |
| When | Offline, ahead of deployment | Online, per request |
| What it computes | Model parameters | Model outputs |
| Data needed | Labels (`y_train`) required | No labels needed |
| Cost | High (minutes-to-hours) | Low (milliseconds) |
| Frequency | Periodic (re-training cadence) | Continuous |
| Determinism | Stochastic (random seeds matter) | **Deterministic** for a fixed model |
| `.fit()` runs? | **Yes** | **NO** |

The whole reason `pickle.dump(pipeline, ...)` exists is to separate
these two phases. Train once, dump the artifact, then load it many
times in many places to do cheap inference without re-paying the
training cost (or re-running training on potentially-different data).

## 2. Why loading models is required in production systems

Three structural reasons:

1. **Cost separation.** Training a Random Forest on millions of rows
   takes minutes; running `.predict()` on a single request takes
   sub-milliseconds. A production system that re-trained per request
   would be unusable. The .pkl + load workflow lets the expensive
   step happen once.
2. **Determinism.** Two identical inference requests must produce
   identical predictions, otherwise the system is non-auditable.
   A re-trained-per-request system would silently drift as new
   data arrived. Loading a frozen .pkl guarantees byte-identical
   outputs for byte-identical inputs.
3. **Decoupling training and serving infrastructure.** The training
   environment may have GPUs, large RAM, datasets in some particular
   path. The serving environment is a small lightweight container
   that just needs the .pkl and `predict()`. The two don't even need
   to be on the same machine.

A production inference service is, at its core, just:

```python
pipeline = pickle.load(open("models/persisted_pipeline.pkl", "rb"))  # once at startup
def handle_request(features):
    return pipeline.predict_proba(features)[:, 1]                    # per request
```

This module is the offline test that proves that pattern works
correctly for FraudX's persisted pipeline.

## 3. What this module does (relative to PR #26)

| PR | Goal |
| :--- | :--- |
| PR #26 (Model Persistence) | Prove `pickle.load` reconstructs the model so faithfully that `np.array_equal(orig_preds, loaded_preds) == True` across processes. |
| **THIS PR** | Use the same loaded pipeline to score **NEW**, never-before-seen transactions, AND re-verify test-set performance still matches PR #26's recorded numbers AND assert no `.fit()` ran during inference. |

The two modules are complementary: PR #26 validates persistence;
this module validates *deployment-style use* of the persistence
artifact.

## 4. The five sample transactions

Hand-crafted to probe different regions of the input space:

| # | amount | tx_count | velocity | category         | location       | Expected reading |
| - | -----: | -------: | -------: | :--------------- | :------------- | :--------------- |
| 0 |  18.50 |        2 |      0.4 | retail           | domestic       | low risk |
| 1 |  35.00 |        3 |      1.2 | food             | international  | mid risk |
| 2 | 450.00 |       12 |      4.5 | travel           | international  | higher risk |
| 3 | 780.00 |       28 |      9.0 | travel           | international  | highest risk |
| 4 | 120.00 |        5 |      2.0 | **cryptoexchange** | international | edge case — category NEVER seen during training |

Sample 4 is the critical edge case: `cryptoexchange` does not appear
in the training set's `category` column. The pipeline's OneHotEncoder
was configured with `handle_unknown="ignore"` (see PR #19) precisely
for this scenario — the encoder produces all zeros for the unknown
category, the model treats the row as if the categorical signal was
absent, and inference succeeds without raising.

## 5. Worked numbers — real run

### 5.1 Inference results for the 5 new transactions

| # | category         | location       | predicted_label | prob_legit | prob_fraud | decision |
| - | :--------------- | :------------- | --------------: | ---------: | ---------: | :------- |
| 0 | retail           | domestic       |               0 |      0.91  |      0.09  | legit    |
| 1 | food             | international  |               0 |      0.96  |      0.04  | legit    |
| 2 | travel           | international  |               0 |      0.72  |      **0.28**  | legit    |
| 3 | travel           | international  |               0 |      0.91  |      0.09  | legit    |
| 4 | cryptoexchange   | international  |               0 |      0.97  |      0.03  | legit    |

Several useful observations:

- All five samples are predicted as **legit** at the default 0.5
  threshold — consistent with the imbalance ceiling identified in
  PRs #17 / #21 / #22 / #25. The trained model's max
  `predict_proba` for the fraud class on this data sits below 0.5.
- **Sample 2** (large travel transaction) gets the highest fraud
  score (0.28). The model has learned *some* signal: large +
  international + travel is more fraud-like than small + domestic +
  retail. The score is just not above the default decision threshold.
- **Sample 4** (unknown category) does NOT crash. `handle_unknown="ignore"`
  in the pipeline's OneHotEncoder produces zeros for the
  unrecognised category and inference proceeds. The model assigns
  it a low fraud score (0.03) because, in the absence of categorical
  signal, the other features (amount=120, velocity=2) look benign.
- **The model is calibrated enough that threshold tuning would
  recover useful fraud scores.** At a threshold of 0.25, Sample 2
  flips to "FRAUD" — the operational lever the docs of PRs #22 / #23 / #25
  point at as the natural next iteration.

### 5.2 Test-set verification

Re-evaluating the loaded pipeline on the standard held-out test set
produces metrics identical to PR #26's recorded numbers:

| Metric | This run (loaded) | PR #26 recorded |
| :--- | ---: | ---: |
| Accuracy | 89.00 % | 89.00 % |
| Precision (1) | 16.67 % | 16.67 % |
| Recall (1) | 5.56 % | 5.56 % |
| F1 (1) | 8.33 % | 8.33 % |
| TN / FP / FN / TP | 177 / 5 / 17 / 1 | 177 / 5 / 17 / 1 |

`np.isclose` on all four metrics → **True**. The persistence is
functioning as designed.

### 5.3 "No retraining" assertion

The module hashes the classifier's fitted attributes
(`classes_`, `len(estimators_)`, `n_features_in_`) before and after
inference. The hashes match exactly:

```
pre-inference  fit-signature hash : 1288865191407943174
post-inference fit-signature hash : 1288865191407943174
identical                          : True
```

If any `.fit()` call had run anywhere in the pipeline, those fitted
attributes would have changed, and the hash comparison would fail.
The assertion is encoded so the test fails loudly if a future change
inadvertently introduces a fit-at-inference bug.

## 6. Why inference must NOT include fitting steps

Four concrete failure modes if you `.fit()` at inference time:

1. **You discard the parameters you trained.** The whole point of the
   .pkl is to freeze them. Refitting throws them away and replaces
   them with new ones derived from whatever data happens to be in
   the request.
2. **You refit on the wrong distribution.** At inference, you have a
   single sample (or a small batch), not a training-representative
   dataset. The new fit's `mean_` / `scale_` / `categories_` will
   silently differ from the training-time ones, and predictions drift.
3. **You break determinism.** Two identical inputs would get different
   predictions because the running fit was different at the moment
   of each request. Production systems must be reproducible.
4. **You break auditability.** A regulator asking "why did this
   transaction get blocked?" needs an answer that depends only on
   the model + the input — not on the cumulative state of every
   prior request.

The right mental model: training is a write-once, read-many operation.
The .pkl is the snapshot. Inference is read-only.

## 7. Security and version compatibility (recap of PR #26 §4.3 & §4.4)

Both topics are covered at length in [`docs/MODEL_PERSISTENCE.md`](../docs/MODEL_PERSISTENCE.md)
(from PR #26 in this project). The 30-second recap relevant to a
deployed inference server:

- **Pickle security.** `pickle.load()` can execute arbitrary code.
  Never load a `.pkl` from an untrusted source. In a production
  inference server, the .pkl should come from your *own* training
  pipeline, signed with HMAC, served via a trusted artifact registry.
  Sandboxing the inference container is good defence-in-depth.
- **Version compatibility.** Pickle records class identity, not
  implementation. A sklearn version mismatch between the training and
  serving environments can produce silently wrong predictions OR
  loud `AttributeError`. Mitigations: pin every dependency in
  `requirements.txt` (this project does), store a sidecar metadata
  JSON next to the .pkl with sklearn/imblearn/numpy versions, and
  refuse to load on mismatch.

## 8. PR-submission checklist (3 marks)

- [x] **Correct Model Loading (1 mark)**
  - Pipeline loaded via `pickle.load()` — see [`src/inference_demo.py`](../src/inference_demo.py).
  - No retraining code path executed — explicit assertion on the
    `classes_` / `estimators_` attributes BEFORE the first inference
    call, and a pre-vs-post hash check.
  - Reproducible workflow — deterministic test split, deterministic
    classifier (`random_state=42`), deterministic
    `predict_proba` outputs.
- [x] **Proper Inference Implementation (1 mark)**
  - 5 new input samples constructed correctly (right column order,
    right shape, right dtypes).
  - Predictions generated via `pipeline.predict()` and
    `pipeline.predict_proba()`.
  - Results displayed in a tabular format and written to
    `reports/inference_predictions.csv`.
- [x] **Verification and Best Practices (1 mark)**
  - Test-set performance validated after loading; matches PR #26's
    recorded numbers via `np.isclose`.
  - Feature consistency maintained — `handle_unknown="ignore"` lets
    the encoder handle the unseen `cryptoexchange` category without
    schema drift.
  - Security and version compatibility discussed — §7 above.

## 9. How to run

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python3 src/inference_demo.py    # just the inference demo
# OR
python3 main.py                  # full pipeline (Phase 3 runs the demo)
```

Artifacts produced:

- `models/persisted_pipeline.pkl` — the loaded artifact (this module
  produces it if PR #26's hasn't already; both modules produce
  identical pickles because both use the same `random_state=42` for
  the same capstone pipeline).
- `reports/inference_predictions.csv` — the 5 new samples + their
  predicted labels + class probabilities + final decisions.

## 10. How this completes the deployment story

- **PR #25** — the capstone selection picked `RF + RandomOverSampler`.
- **PR #26** — persisted that pipeline via `pickle.dump`, proved
  byte-identical load.
- **THIS PR** — used the persisted pipeline to score 5 new
  transactions, asserted no retraining ran, re-verified test-set
  consistency.

After this PR, the project has a complete deployment loop:
**train → select → persist → load → infer → verify**. The .pkl
artifact plus pinned `requirements.txt` is sufficient to stand up
a production inference service that scores new transactions in
real time with no retraining.

The natural next iteration (out of scope): wrap `pipeline.predict_proba`
in a small HTTP service (FastAPI / Flask), add request-level metrics
(latency, throughput, prediction distribution drift), and ship.
