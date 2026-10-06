"""The committed artifacts load under the pinned environment and serve predictions."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.engagement import engagement_score
from src.features import CLIP_BOUND, apply_scaler, load_scaler
from tests.synthetic import ROOT, make_synthetic_dataset

ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="module")
def artifacts():
    return {
        "scaler": load_scaler(ARTIFACTS / "scaler.pkl"),
        "logreg": joblib.load(ARTIFACTS / "logreg.pkl"),
        "xgb": joblib.load(ARTIFACTS / "xgb.pkl"),
        "columns": json.loads((ARTIFACTS / "columns.json").read_text()),
    }


def test_artifacts_agree_on_the_feature_schema(artifacts):
    cols = artifacts["columns"]
    assert len(cols) == len(set(cols)) == 988
    assert artifacts["scaler"].n_features_in_ == 988
    assert artifacts["logreg"].n_features_in_ == 988
    assert list(artifacts["scaler"].feature_names_in_) == cols
    assert list(artifacts["logreg"].classes_) == [0.0, 1.0, 2.0]


def test_example_rows_are_classified_as_their_label(artifacts):
    df = pd.read_csv(ROOT / "data" / "example-mental-state.csv")
    X = apply_scaler(artifacts["scaler"], df[artifacts["columns"]])
    for name in ("logreg", "xgb"):
        pred = artifacts[name].predict(X)
        np.testing.assert_array_equal(pred.astype(float), df["Label"].to_numpy())


def test_inference_path_on_synthetic_rows(artifacts):
    df = make_synthetic_dataset(n_rows=30, n_duplicates=0)
    X = apply_scaler(artifacts["scaler"], df[artifacts["columns"]])
    assert X.abs().to_numpy().max() <= CLIP_BOUND

    proba = artifacts["logreg"].predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    score = engagement_score(artifacts["logreg"], X)
    assert score.shape == (30,)
    assert np.all((score >= 0) & (score <= 100))
    assert set(artifacts["xgb"].predict(X)) <= {0, 1, 2}


def test_recorded_test_metrics_match_confusion_matrices():
    evaluation = json.loads((ARTIFACTS / "evaluation.json").read_text())
    for name, expected in (("logreg", 0.9528), ("xgb", 0.9704)):
        cm = np.array(evaluation[name]["confusion_matrix"])
        assert cm.sum() == 473
        tp = np.diag(cm)
        precision = tp / cm.sum(axis=0)
        recall = tp / cm.sum(axis=1)
        macro_f1 = np.mean(2 * precision * recall / (precision + recall))
        assert round(macro_f1, 4) == evaluation[name]["test_macro_f1"] == expected


def test_streamlit_app_starts():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "EEG Mental State Classifier"
