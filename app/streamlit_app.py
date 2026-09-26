from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dedup_split
from src.engagement import engagement_score
from src.features import apply_scaler

ARTIFACTS_DIR = ROOT / "artifacts"
COLUMNS_PATH = ARTIFACTS_DIR / "columns.json"
SAMPLE_INPUT_PATH = ARTIFACTS_DIR / "sample_input.csv"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
LOGREG_PATH = ARTIFACTS_DIR / "logreg.pkl"
XGB_PATH = ARTIFACTS_DIR / "xgb.pkl"

CLASS_NAMES = {0.0: "relaxed", 1.0: "neutral", 2.0: "concentrating"}


def load_expected_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        cols = json.load(f)
    if not isinstance(cols, list):
        raise ValueError("artifacts/columns.json must be a list of column names.")
    return cols


def generate_sample_input(path: Path) -> pd.DataFrame:
    _, X_test, _, _, _ = load_dedup_split()
    sample_df = X_test.head(5).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(path, index=False)
    return sample_df


@st.cache_resource
def load_artifacts() -> dict:
    """Load ALL training artifacts once and keep them in memory.

    Using @st.cache_resource so repeated uploads don't reload from disk.
    """
    return {
        "scaler": joblib.load(SCALER_PATH),
        "logreg": joblib.load(LOGREG_PATH),
        "xgb": joblib.load(XGB_PATH),
        "expected_columns": load_expected_columns(COLUMNS_PATH),
    }


def run_inference(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Apply the full training pipeline: reorder columns → scale+clip → predict → score.

    Returns a DataFrame with one row per input row containing: predicted class
    (LogReg and XGB), engagement score, and both models' probability vectors.
    """
    X = raw_df[artifacts["expected_columns"]]  # enforce exact train-time column order
    X_scaled = apply_scaler(artifacts["scaler"], X)

    lr = artifacts["logreg"]
    xgb = artifacts["xgb"]

    lr_pred = lr.predict(X_scaled)
    xgb_pred = xgb.predict(X_scaled)
    lr_probs = lr.predict_proba(X_scaled)
    score = engagement_score(lr, X_scaled)

    out = pd.DataFrame({
        "row": np.arange(len(X)) + 1,
        "state_logreg": [CLASS_NAMES[v] for v in lr_pred],
        "state_xgb": [CLASS_NAMES[v] for v in xgb_pred],
        "engagement_score": score.round(2),
    })
    classes = list(lr.classes_)
    for lbl, name in CLASS_NAMES.items():
        out[f"P_{name}"] = lr_probs[:, classes.index(lbl)].round(4)
    return out


st.set_page_config(page_title="EEG Mental State Classifier", layout="wide")
st.title("EEG Mental State Classifier")
st.markdown(
    """
This app expects a CSV file with **exactly the 988 feature columns**
defined in `artifacts/columns.json` (the `Label` column is **optional**).

Each CSV row is treated as **one independent time window**.

For each row, the app outputs:
- the predicted mental state
- the engagement score
"""
)

with st.sidebar:
    st.subheader("Utilities")
    if st.button("Generate sample CSV"):
        try:
            sample_df = generate_sample_input(SAMPLE_INPUT_PATH)
            st.success(f"Sample generated: {SAMPLE_INPUT_PATH.as_posix()}")
            st.caption(f"{len(sample_df)} rows saved.")
        except Exception as exc:
            st.error(f"Failed to generate sample CSV: {exc}")

    if SAMPLE_INPUT_PATH.exists():
        st.download_button(
            label="Download sample CSV",
            data=SAMPLE_INPUT_PATH.read_bytes(),
            file_name=SAMPLE_INPUT_PATH.name,
            mime="text/csv",
        )

uploaded_file = st.file_uploader("Upload a CSV for validation", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        if "Label" in uploaded_df.columns:
            uploaded_df = uploaded_df.drop(columns=["Label"])

        expected_columns = load_expected_columns(COLUMNS_PATH)
    except Exception as exc:
        st.error(f"Error reading the file or the expected columns: {exc}")
        st.stop()

    expected_set = set(expected_columns)
    uploaded_set = set(uploaded_df.columns)
    missing_columns = sorted(expected_set - uploaded_set)

    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        suffix = " ..." if len(missing_columns) > 10 else ""
        st.error(
            f"Invalid CSV: {len(missing_columns)} expected columns are missing. "
            f"First missing: {preview}{suffix}"
        )
        st.stop()

    n_rows = int(np.int64(len(uploaded_df)))
    st.success(f"CSV validated: {n_rows} windows accepted")
    st.info(
        f"{len(uploaded_df.columns)} columns received, {len(expected_columns)} expected"
    )

    try:
        artifacts = load_artifacts()
        results = run_inference(uploaded_df, artifacts)
    except Exception as exc:
        st.error(f"Inference error: {exc}")
        st.stop()

    st.subheader("Results per window")
    st.dataframe(results, use_container_width=True)

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Mean engagement score",
            f"{results['engagement_score'].mean():.1f}",
            help="Mean score (0–100) across all windows in the CSV.",
        )
    with c2:
        st.metric(
            "Predominant state (LogReg)",
            results["state_logreg"].mode().iloc[0],
        )
    with c3:
        st.metric(
            "LogReg–XGB agreement",
            f"{(results['state_logreg'] == results['state_xgb']).mean() * 100:.0f}%",
            help="Fraction of windows where both models agree on the predicted state.",
        )

    st.download_button(
        label="Download results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="eeg_predictions.csv",
        mime="text/csv",
    )
