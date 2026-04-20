"""End-to-end training entry point.

Reproducibility contract:
    PYTHONPATH=. python scripts/train.py

Produces, atomically:
    artifacts/scaler.pkl     — RobustScaler fit on TRAIN only
    artifacts/logreg.pkl     — final LogReg on scaled TRAIN
    artifacts/xgb.pkl        — final XGBoost on scaled TRAIN
    artifacts/columns.json   — exact train-time column order (consumed by the app)

Also logs honest CV macro-F1 (scaler refit per fold) for both models to
artifacts/run_log.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_dedup_split
from src.features import apply_scaler, fit_scaler, save_scaler
from src.models import (
    honest_cv,
    logreg_estimator,
    train_logreg,
    train_xgboost,
    xgb_estimator,
)
from src.report_log import log_finding, log_metric


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    warnings.filterwarnings("ignore")

    X_train, X_test, y_train, _, info = load_dedup_split()
    print(f"loaded: n_train={info['train_size']}  n_test={info['test_size']}  n_features={info['n_features']}")

    # Honest pipeline CV — scaler refit per fold, clip applied per fold.
    print("running honest CV (scaler inside the pipeline)...")
    lr_cv = honest_cv(logreg_estimator(), X_train, y_train)
    xgb_cv = honest_cv(xgb_estimator(), X_train, y_train)

    # Final fit on full train with externalized scaler (persisted for inference).
    scaler = fit_scaler(X_train)
    save_scaler(scaler)
    X_train_scaled = apply_scaler(scaler, X_train)
    train_logreg(X_train_scaled, y_train)
    train_xgboost(X_train_scaled, y_train)

    # Column manifest for the Streamlit app's schema check.
    (ARTIFACTS / "columns.json").write_text(json.dumps(X_train.columns.tolist(), indent=2))

    # Log honest numbers.
    log_metric("logreg_cv_pipeline_mean", round(float(lr_cv.mean()), 4), note="honest CV, scaler refit per fold")
    log_metric("logreg_cv_pipeline_std", round(float(lr_cv.std()), 4))
    log_metric("xgb_cv_pipeline_mean", round(float(xgb_cv.mean()), 4), note="honest CV, scaler refit per fold")
    log_metric("xgb_cv_pipeline_std", round(float(xgb_cv.std()), 4))
    log_finding(
        "Honest pipeline-CV numbers",
        f"LogReg macro-F1 = {lr_cv.mean():.4f} ± {lr_cv.std():.4f}. "
        f"XGBoost macro-F1 = {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}. "
        f"Scaler refit per fold inside sklearn Pipeline. "
        f"Test-set numbers (scripts/evaluate.py) are unaffected — they come from a locked holdout.",
    )

    # Summary.
    print(f"\n=== training complete ===")
    print(f"LogReg  honest CV macro-F1 : {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
    print(f"XGB     honest CV macro-F1 : {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")
    print(f"\nartifacts/ updated: scaler.pkl, logreg.pkl, xgb.pkl, columns.json")
    print(f"next: PYTHONPATH=. python scripts/evaluate.py")


if __name__ == "__main__":
    main()
