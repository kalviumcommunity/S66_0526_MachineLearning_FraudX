# Model Persistence with Pickle

This document is the long-form write-up for the FraudX model
persistence module. The short summary lives in
[`README.md`](../README.md#-model-persistence-with-pickle); this file
covers the serialization mechanics, the fresh-environment verification
approach, the four required reflection answers, security and
versioning considerations, and how the persisted artifact fits into
the rest of the project.

## 1. What this module does

Takes the capstone pipeline selected in PR #25 (`RF + RandomOverSampler`)
and exercises the full save → fresh-process load → byte-identical
verification cycle the assignment requires:

| Step | Where | What runs |
| :--- | :--- | :--- |
| Train | `model_persistence.py` | Build + fit `imblearn.Pipeline(preprocessor + RandomOverSampler + RandomForestClassifier)` on `X_train`, predict on `X_test`, record metrics + predictions hash. |
| Save | `model_persistence.py` | `pickle.dump(pipeline, open("models/persisted_pipeline.pkl", "wb"), protocol=HIGHEST_PROTOCOL)` |
| Load (fresh env) | `load_and_verify.py` (subprocess) | `pickle.load(...)`, rebuild the same test split, call `pipeline.predict(X_test)`, write predictions + metrics to `reports/load_and_verify.json` |
| Verify | `model_persistence.py` | Read the JSON. Assert `np.array_equal(original_predictions, loaded_predictions)`. Assert metrics match. Fail loudly if either differs. |

The "fresh environment" is a separate `python3` subprocess invoked via
`subprocess.run`. It imports nothing from `model_persistence.py`'s
in-memory state — only the project's data-loading utilities and
`pickle.load`. That's the closest analog to the assignment's
"restart the kernel" / "fresh environment" instruction inside a
`python3 src/...` workflow.

## 2. Why a subprocess, not just a separate function

A same-process load (e.g., `pipeline_new = pickle.load(open(p, "rb"))`
inside the same script that pickled it) is a weaker test. The current
process already has all the relevant classes parsed, modules loaded,
imports resolved. `pickle.load` can succeed even when a genuinely fresh
process would fail (missing dependency, missing import path, class
identity mismatch from a version bump). A subprocess gives:

- a clean module-import cache (every import re-runs),
- no in-memory references to the original `pipeline` object,
- a fresh interpreter state that mirrors what a deployed inference
  server would see,
- a separate stdout we can inspect explicitly,
- a clean exit code we can check (non-zero = subprocess failed).

The orchestrator runs the subprocess, parses its JSON output, and
performs the byte-equality check on the predictions array. If the
subprocess fails for any reason, the orchestrator raises.

## 3. Worked numbers on FraudX

Test set: 200 samples (182 class 0 / 18 class 1).

Both runs (original in-memory + loaded subprocess) produced:

| Metric | Value |
| :--- | ---: |
| Accuracy | 89.00 % |
| Precision (class 1) | 16.67 % |
| Recall (class 1) | 5.56 % |
| F1 (class 1) | 8.33 % |
| TN / FP / FN / TP | 177 / 5 / 17 / 1 |

Verification asserts:
- `np.array_equal(original_predictions, loaded_predictions)` → **True**
- accuracy / precision / recall / F1 match (`np.isclose`) → **True**

The `.pkl` file is ~2 MB (the RandomOverSampler-trained RF has more
data in its internal trees than a vanilla RF).

## 4. Part 4 — Reflection (required)

### 4.1 What is serialization?

Serialization is the process of converting a live in-memory Python
object — with all its references, attributes, learned parameters, and
nested object graph — into a byte stream that can be written to disk
or sent over a network. The reverse process (deserialisation,
`pickle.load`) reconstructs an equivalent object in a different
Python process by reading the byte stream and recreating the class
instances.

`pickle` does this for arbitrary Python objects via a recursive
protocol that records, for each object, its class identity (module
path + class name) and the state needed to rebuild it (typically the
`__dict__`, plus any `__reduce__`-customised state). For an sklearn
Pipeline, "state" includes every fitted attribute on every step —
the imputer's `statistics_`, the scaler's `mean_` and `scale_`, the
encoder's `categories_`, every tree node in the RandomForest's `n_estimators` trees, the random seed used by the RandomOverSampler.

The result is a single `.pkl` file that captures the *exact* model
configuration produced by training, ready to be loaded in a different
process or on a different machine — provided the receiving environment
has compatible library versions installed.

### 4.2 Why is saving the entire pipeline better than saving only the model?

Because the trained classifier expects features that were produced by
the SAME preprocessor that was fitted during training:

- The imputer's median was computed from the training data.
- The scaler's mean / variance were computed from the training data.
- The encoder's category vocabulary was learned from the training data.

If you save only the classifier and re-fit the preprocessor at load
time on whatever data happens to be available (the test set, a
production batch, new data, a different fold of the same data), the
preprocessor's parameters will silently differ from the training-time
ones, and inference predictions will drift away from what the model
was trained to handle. The model expects, say, "amount scaled with
the training mean μ = 97.25 and variance σ² = 12345"; if you re-fit
the scaler on a different sample and get μ = 102.0, every prediction
is computed against a different feature distribution than the model
was trained on.

The Pipeline construct freezes all of these parameters together in a
single fitted object. `pickle.dump` writes them all. `pickle.load`
restores them all. A loaded pipeline produces byte-identical
predictions because every transformer in the chain is initialised to
the exact same fitted state as the original.

The runtime verification in this module makes this concrete:
`np.array_equal(original_predictions, loaded_predictions) → True`.
That equality would NOT hold if we had saved only the classifier and
re-fit the preprocessor.

### 4.3 What security risk exists when loading Pickle files?

`pickle.load()` can execute **arbitrary code** during deserialisation.
The pickle protocol allows any object to define a `__reduce__` method
that returns a callable + arguments; when the pickle is loaded,
`pickle.load` invokes that callable to reconstruct the object. A
malicious pickle can therefore include a `__reduce__` that returns
`(os.system, ("rm -rf /",))` or `(subprocess.run, (["curl", "...", "|", "sh"]))`
— and `pickle.load` will run it, no questions asked.

This is not a hypothetical: it's a well-documented and exploited
attack surface. The official Python documentation explicitly warns:
*"Never unpickle data that could have come from an untrusted source."*

Mitigations:

1. **Provenance.** Only load `.pkl` files produced by your own
   training pipeline, in your own infrastructure. Treat external
   `.pkl` files the same way you'd treat an arbitrary executable
   from the internet — i.e., don't run it.
2. **Signed artifacts.** Sign your `.pkl` files with HMAC or
   asymmetric crypto when produced; verify the signature before
   loading. Keep the signing keys out of band.
3. **Sandboxing.** Load `.pkl` files in restricted execution
   environments (containers with no network egress, no filesystem
   write access, dropped capabilities).
4. **Safer formats for cross-trust loading.** ONNX, JSON-described
   models (for simple linear models), or framework-native formats
   like XGBoost's `.ubj` are all safer because they describe data,
   not callable graphs.
5. **`joblib` is not a security solution.** `joblib.load` uses
   pickle under the hood and has the same vulnerability. It's just
   faster for large numpy arrays.

In short: pickle is for *trusted* artifacts inside *your own*
infrastructure. The moment a `.pkl` crosses a trust boundary, the
security model breaks.

### 4.4 What could go wrong if library versions differ?

Pickle records the class **identity** (module path + class name) but
not the implementation. When `pickle.load` reconstructs an object, it
looks up the class in the *current* Python environment by that
identity and instantiates it with the recorded state.

If the receiving environment's version of that class has a different
internal layout, several things can go wrong:

1. **Silent wrong predictions.** sklearn might rename an attribute
   between versions (e.g., `n_outputs_` → something else). The
   unpickled object will have the old attribute populated but the
   new code paths will read the new one, defaulting to `0` or `None`
   and producing wrong predictions WITHOUT raising any error.
2. **AttributeError at load.** A required attribute the new class
   expects but the old pickle didn't write triggers an
   `AttributeError` at first access. Easier to catch — at least it
   fails loud.
3. **Unpickle errors.** If a class was removed or its module path
   changed, `pickle.load` raises `AttributeError: Can't get attribute
   'XYZ' on <module ...>`. Also loud.
4. **Compatibility shims that work today but break tomorrow.**
   sklearn sometimes adds backward-compatibility shims for one major
   version and removes them in the next. A pickle that loads on 1.5
   may silently break on 1.6.

Mitigations:

1. **Pin every version.** Lock `requirements.txt` to specific patch
   versions (`scikit-learn==1.8.0`, not `>=1.8`).
2. **Capture metadata.** Store a sidecar JSON next to the `.pkl`
   containing `{"sklearn_version": "1.8.0", "imblearn_version":
   "0.12.4", "numpy_version": "2.4.4", "python_version": "3.13"}`.
3. **Refuse to load when versions don't match.** A few lines of
   defensive code in the loader: read the sidecar, compare to
   `sklearn.__version__`, raise if they differ.
4. **Re-train on version upgrades.** The cleanest fix is to re-train
   the model whenever you bump library versions, rather than trying
   to load an old pickle into a new environment.

The FraudX project's `requirements.txt` pins every dependency to a
specific patch version precisely for this reason.

## 5. PR submission checklist (3 marks)

- [x] **Correct Serialization (1 mark)**
  - Pipeline saved via `pickle.dump(pipeline, fp, protocol=pickle.HIGHEST_PROTOCOL)` — see [`src/model_persistence.py`](../src/model_persistence.py).
  - File extension `.pkl` — `models/persisted_pipeline.pkl`.
- [x] **Proper Deserialization & Verification (1 mark)**
  - Loaded via `pickle.load(fp)` in [`src/load_and_verify.py`](../src/load_and_verify.py).
  - Predictions generated without retraining — the subprocess never
    calls `.fit(...)` on the loaded pipeline.
  - Performance verified after loading — runtime asserts
    `np.array_equal(original_predictions, loaded_predictions)` and
    metric equality.
- [x] **Best Practices & Explanation (1 mark)**
  - Entire pipeline saved (preprocessor + sampler + classifier) — not
    just the model.
  - Clear explanation of serialization concept — §4.1.
  - Security and versioning considerations mentioned — §4.3 and §4.4.

## 6. How to run

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python3 src/model_persistence.py    # train + save + verify
# OR
python3 main.py                      # full pipeline (Phase 3 runs the persistence module)
```

Artifacts produced:

- `models/persisted_pipeline.pkl` — the fitted
  `imblearn.Pipeline(preprocessor + RandomOverSampler + RandomForestClassifier)`.
  ~2 MB.
- `reports/load_and_verify.json` — the subprocess's
  predictions + metrics for the orchestrator to diff against.

## 7. How this completes the project arc

- [PR #15](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/15)
  through PR #24 — built the model + evaluation infrastructure.
- [PR #25](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/25)
  — capstone selection picked `RF + RandomOverSampler` as the final
  model.
- **THIS PR** — persists that final model and proves it survives a
  fresh-process load. The model is now deployment-ready: the `.pkl`
  artifact + `requirements.txt` are sufficient to serve predictions
  from a different machine without retraining.

The natural next iteration, out of scope here: pair the persisted
pipeline with threshold tuning (using `predict_proba` rather than
`predict`) against a stated `c_FN/c_FP` cost ratio. The saved
pipeline supports that workflow directly — `predict_proba` is
available on the loaded object.
