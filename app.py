"""
app.py

FraudX Streamlit interface — final-system deliverable.

Launch with:
    streamlit run app.py

This app loads the deployment pipeline (models/pipeline.joblib + the
sidecar metadata in models/pipeline_metadata.json), exposes input
widgets for every feature with realistic min/max constraints, and
displays the prediction plus the fraud probability plus a
plain-language verdict for the user.

Design choices:
- `@st.cache_resource` loads the pipeline ONCE per server session
  (it's expensive to deserialise; cheap to call .predict() afterwards).
- Input widgets have realistic min/max derived from training data
  (see src/config.FEATURE_VALUE_RANGES).
- The categorical selectors are populated from CATEGORY_OPTIONS /
  LOCATION_OPTIONS so the user can't enter free-text that the encoder
  hasn't seen — though if they did, `handle_unknown="ignore"` would
  handle it gracefully.
- If `models/pipeline.joblib` does not exist, the app shows a clear
  setup message rather than crashing. Run
  `python3 src/deployment.py` (or `python3 main.py`) to build it.
- The metadata JSON is displayed in an expander so the user can
  audit which sklearn / imblearn version produced the artifact.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    CATEGORY_OPTIONS,
    DEPLOYMENT_METADATA_PATH,
    DEPLOYMENT_PIPELINE_PATH,
    FEATURE_VALUE_RANGES,
    LOCATION_OPTIONS,
    NUMERICAL_FEATURES,
)


# ----------------------------------------------------------------------
# Cached loaders (one read per server session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained pipeline ...")
def _load_pipeline_and_metadata() -> Optional[Dict[str, Any]]:
    """
    Read models/pipeline.joblib + models/pipeline_metadata.json once.

    Returns None when the artifact doesn't exist so the caller can show
    a graceful setup message instead of crashing.

    `@st.cache_resource` ensures the loaded pipeline is reused across
    user interactions — same object in memory, same behaviour for
    every request.
    """
    if not os.path.exists(DEPLOYMENT_PIPELINE_PATH):
        return None
    pipeline = joblib.load(DEPLOYMENT_PIPELINE_PATH)

    metadata = {}
    if os.path.exists(DEPLOYMENT_METADATA_PATH):
        with open(DEPLOYMENT_METADATA_PATH) as fp:
            metadata = json.load(fp)

    return {"pipeline": pipeline, "metadata": metadata}


# ----------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------
def _build_input_frame(values: Dict[str, Any]) -> pd.DataFrame:
    """Wrap a dict of widget values in a single-row DataFrame whose
    column order matches the training feature schema."""
    return pd.DataFrame([{col: values[col] for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES}])


def _plain_language_verdict(label: int, prob_fraud: float) -> str:
    if label == 1:
        return (
            f"🚨 **Predicted FRAUD** with confidence {prob_fraud:.1%}. "
            "Review this transaction manually before approval."
        )
    # Label 0: the trained pipeline is leaning legit. Refine the wording
    # by where the fraud probability sits.
    if prob_fraud < 0.10:
        return (
            f"✅ **Predicted LEGITIMATE.** Fraud probability is low ({prob_fraud:.1%}). "
            "No additional review needed under default risk policy."
        )
    if prob_fraud < 0.30:
        return (
            f"✅ **Predicted LEGITIMATE.** Fraud probability is modest ({prob_fraud:.1%}). "
            "Below the default decision threshold but worth tracking if multiple "
            "similar transactions arrive in quick succession."
        )
    return (
        f"⚠️ **Predicted LEGITIMATE, but borderline.** Fraud probability is "
        f"{prob_fraud:.1%} — close to the default 0.5 threshold. "
        "Consider manual review or threshold tuning for this category."
    )


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="FraudX — Transaction Fraud Detection",
    page_icon="🔒",
    layout="centered",
)

st.title("🔒 FraudX — Transaction Fraud Detection")
st.markdown(
    "Score a single transaction with the trained FraudX pipeline "
    "(`RandomForestClassifier` + `RandomOverSampler`, selected in "
    "[PR #25](https://github.com/kalviumcommunity/S66_0526_MachineLearning_FraudX/pull/25))."
)
st.divider()


# ----------------------------------------------------------------------
# Load the model
# ----------------------------------------------------------------------
loaded = _load_pipeline_and_metadata()
if loaded is None:
    st.error(
        "Deployment pipeline not found at `models/pipeline.joblib`. "
        "Build it first by running:\n\n"
        "```bash\nexport PYTHONPATH=.\npython3 src/deployment.py\n```\n\n"
        "(or run the full pipeline via `python3 main.py`.)"
    )
    st.stop()

pipeline = loaded["pipeline"]
metadata = loaded["metadata"]


# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Transaction details")

with st.form("transaction_input_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        amount = st.number_input(
            "Amount (currency units)",
            min_value=float(FEATURE_VALUE_RANGES["amount"]["min"]),
            max_value=float(FEATURE_VALUE_RANGES["amount"]["max"]),
            value=float(FEATURE_VALUE_RANGES["amount"]["default"]),
            step=float(FEATURE_VALUE_RANGES["amount"]["step"]),
            help="Transaction value. Realistic range: 0 – 1000.",
        )
        transaction_count = st.number_input(
            "Recent transaction count",
            min_value=int(FEATURE_VALUE_RANGES["transaction_count"]["min"]),
            max_value=int(FEATURE_VALUE_RANGES["transaction_count"]["max"]),
            value=int(FEATURE_VALUE_RANGES["transaction_count"]["default"]),
            step=int(FEATURE_VALUE_RANGES["transaction_count"]["step"]),
            help="How many transactions this account has made recently. "
                 "Higher = more activity.",
        )
        velocity = st.number_input(
            "Velocity (transactions per minute equivalent)",
            min_value=float(FEATURE_VALUE_RANGES["velocity"]["min"]),
            max_value=float(FEATURE_VALUE_RANGES["velocity"]["max"]),
            value=float(FEATURE_VALUE_RANGES["velocity"]["default"]),
            step=float(FEATURE_VALUE_RANGES["velocity"]["step"]),
            help="Rate of transactions. High velocity is a known fraud signal.",
        )

    with c2:
        category = st.selectbox(
            "Merchant category",
            options=CATEGORY_OPTIONS,
            index=2,  # default: 'retail'
            help="Spending category. `online` and `travel` carry higher fraud base rates.",
        )
        location = st.selectbox(
            "Transaction location",
            options=LOCATION_OPTIONS,
            index=0,  # default: 'domestic'
            help="`international` transactions are higher-risk than `domestic`.",
        )

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
if submitted:
    sample = _build_input_frame({
        "amount":            amount,
        "transaction_count": transaction_count,
        "velocity":          velocity,
        "category":          category,
        "location":          location,
    })

    label = int(pipeline.predict(sample)[0])
    proba = pipeline.predict_proba(sample)[0]
    prob_legit, prob_fraud = float(proba[0]), float(proba[1])

    st.subheader("Prediction")

    # Headline
    if label == 1:
        st.error(_plain_language_verdict(label, prob_fraud), icon="🚨")
    elif prob_fraud >= 0.30:
        st.warning(_plain_language_verdict(label, prob_fraud), icon="⚠️")
    else:
        st.success(_plain_language_verdict(label, prob_fraud), icon="✅")

    # Probability metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted label", "Fraud" if label == 1 else "Legitimate")
    m2.metric("P(fraud)", f"{prob_fraud:.2%}")
    m3.metric("P(legit)", f"{prob_legit:.2%}")

    # Raw input echo for the demo
    with st.expander("See the exact input row scored"):
        st.dataframe(sample, use_container_width=True)
        st.markdown(
            "All preprocessing (impute + scale + encode) happens **inside** the "
            "loaded pipeline. No `.fit()` call runs at inference time — the "
            "parameters are the ones the pipeline learned at training time."
        )


# ----------------------------------------------------------------------
# Model card / metadata expander
# ----------------------------------------------------------------------
st.divider()
with st.expander("📋 Model card (training metadata)"):
    if metadata:
        st.markdown(
            f"**Pipeline:** {metadata.get('model_description', 'N/A')}\n\n"
            f"**Trained at:** `{metadata.get('training', {}).get('trained_at', 'unknown')}`\n\n"
            f"**Random state:** `{metadata.get('training', {}).get('random_state', 'unknown')}`\n\n"
            f"**Training samples:** {metadata.get('training', {}).get('train_size', '?')}\n\n"
            f"**Test samples:**     {metadata.get('training', {}).get('test_size', '?')}\n\n"
            f"**Fraud share (train):** {metadata.get('training', {}).get('train_fraud_share', 0):.2%}\n\n"
            f"**Fraud share (test):**  {metadata.get('training', {}).get('test_fraud_share', 0):.2%}"
        )
        st.markdown("**Test-set metrics (when this pipeline was saved):**")
        tm = metadata.get("test_metrics", {})
        st.table(pd.DataFrame([{
            "Accuracy": f"{tm.get('accuracy', 0):.2%}",
            "Precision (fraud)": f"{tm.get('precision_1', 0):.2%}",
            "Recall (fraud)":    f"{tm.get('recall_1', 0):.2%}",
            "F1 (fraud)":        f"{tm.get('f1_1', 0):.2%}",
        }]))
        st.markdown("**Library versions at training time:**")
        st.json(metadata.get("library_versions", {}))
    else:
        st.info("`pipeline_metadata.json` not present alongside the pipeline.")

st.caption(
    "FraudX is a Kalvium x LPU course project. Built by [@jet6ki](https://github.com/jet6ki) "
    "with disciplined ML engineering across 12 prior PRs (preprocessing → leakage audit → "
    "imbalance handling → tuning → comparison → serialisation → inference)."
)
