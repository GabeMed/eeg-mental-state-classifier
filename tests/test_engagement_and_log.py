from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from src import report_log
from src.engagement import engagement_score
from tests.synthetic import COLUMNS, ROOT


class _FixedProba:
    def __init__(self, probs, classes):
        self._probs = np.asarray(probs, dtype=float)
        self.classes_ = np.asarray(classes, dtype=float)

    def predict_proba(self, X):
        return self._probs


def test_engagement_score_endpoints():
    model = _FixedProba([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0, 0.5]], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(engagement_score(model, None), [0.0, 50.0, 100.0, 50.0])


def test_engagement_score_uses_class_labels_not_column_order():
    # Columns ordered (concentrating, relaxed, neutral).
    model = _FixedProba([[0.9, 0.1, 0.0]], [2.0, 0.0, 1.0])
    np.testing.assert_allclose(engagement_score(model, None), [90.0])


def test_engagement_score_is_monotonic_in_p_concentrating():
    p = np.linspace(0, 1, 11)
    probs = np.column_stack([1 - p, np.zeros_like(p), p])
    scores = engagement_score(_FixedProba(probs, [0.0, 1.0, 2.0]), None)
    assert np.all(np.diff(scores) > 0)
    assert scores.min() >= 0 and scores.max() <= 100


def test_run_log_is_append_only(tmp_path, monkeypatch):
    log = tmp_path / "run_log.jsonl"
    monkeypatch.setattr(report_log, "LOG_PATH", log)

    report_log.log_metric("cv", 0.95, note="synthetic")
    report_log.log_doubt("open question")
    report_log.log_doubt("closed question", resolution="answered")
    report_log.log_finding("t", "d")
    report_log.log_decision("t2", "d2")

    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["type"] for e in entries] == ["metric", "doubt", "doubt", "finding", "decision"]
    assert entries[1]["status"] == "open"
    assert entries[2]["status"] == "resolved"
    assert all("ts" in e for e in entries)

    report_log.log_metric("cv", 0.96)
    assert len(log.read_text().splitlines()) == len(entries) + 1


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_log_groups_entries():
    render_log = _load_script("render_log")
    md = render_log.render(
        [
            {"type": "finding", "title": "F", "detail": "fd"},
            {"type": "decision", "title": "D", "detail": "dd"},
            {"type": "doubt", "question": "Q", "resolution": "R", "status": "resolved"},
            {"type": "metric", "name": "m", "value": 1, "note": None},
        ]
    )
    for heading in ("Empirical findings", "Methodological decisions", "Doubts", "Key metrics"):
        assert heading in md
    assert "| `m` | 1 |  |" in md


def test_committed_run_log_is_valid_jsonl():
    lines = (ROOT / "artifacts" / "run_log.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    assert entries
    assert {e["type"] for e in entries} <= {"finding", "decision", "doubt", "metric"}


@pytest.fixture(scope="module")
def evaluate_module():
    return _load_script("evaluate")


def test_every_feature_maps_to_a_known_family(evaluate_module):
    families = {evaluate_module._family_of(c) for c in COLUMNS}
    assert "other" not in families
    assert len(families) == 13


@pytest.mark.parametrize(
    "col, family, channel",
    [
        ("freq_101_2", "freq", "AF8"),
        ("lag1_logcovM_2_2", "logcovM", "AF8"),
        ("covM_0_3", "covM", "TP10"),
        ("mean_d_h2h1_1", "mean_d", "AF7"),
        ("mean_q1_0", "mean_q", "TP9"),
        ("mean_3", "mean", "TP10"),
        ("topFreq_1_0", "topFreq", "TP9"),
    ],
)
def test_family_and_channel_parsing(evaluate_module, col, family, channel):
    assert evaluate_module._family_of(col) == family
    assert evaluate_module._channel_of(col) == channel
