"""Run train.py → evaluate.py → plot_engagement.py on synthetic data in a temp dir.

Paths are redirected with EEG_DATA_PATH / EEG_ARTIFACTS_DIR / EEG_FIGURES_DIR,
so the committed artifacts and figures are never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from tests.synthetic import ROOT

pytestmark = pytest.mark.slow


def _digest(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_full_pipeline_on_synthetic_data(tmp_path, synthetic_csv):
    committed = sorted((ROOT / "artifacts").glob("*")) + sorted((ROOT / "notebooks" / "figures").glob("*"))
    before = _digest(committed)

    artifacts = tmp_path / "artifacts"
    figures = tmp_path / "figures"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "EEG_DATA_PATH": str(synthetic_csv),
        "EEG_ARTIFACTS_DIR": str(artifacts),
        "EEG_FIGURES_DIR": str(figures),
        "MPLBACKEND": "Agg",
    }
    for script in ("train.py", "evaluate.py", "plot_engagement.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            env=env,
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    for name in ("scaler.pkl", "logreg.pkl", "xgb.pkl", "columns.json", "evaluation.json", "run_log.jsonl"):
        assert (artifacts / name).exists(), name
    for name in (
        "confusion_logreg.png",
        "confusion_xgb.png",
        "family_importance.png",
        "engagement_boxplot.png",
        "engagement_scatter.png",
        "engagement_histogram.png",
    ):
        assert (figures / name).exists(), name

    evaluation = json.loads((artifacts / "evaluation.json").read_text())
    for model in ("logreg", "xgb"):
        cm = evaluation[model]["confusion_matrix"]
        assert sum(map(sum, cm)) == 60  # 300 unique rows, 20% held out
        assert evaluation[model]["test_macro_f1"] > 0.9  # synthetic classes are well separated

    log = [json.loads(line) for line in (artifacts / "run_log.jsonl").read_text().splitlines()]
    metric_names = {e["name"] for e in log if e["type"] == "metric"}
    assert {"logreg_cv_pipeline_mean", "xgb_cv_pipeline_mean", "logreg_test_macro_f1"} <= metric_names

    assert _digest(committed) == before
