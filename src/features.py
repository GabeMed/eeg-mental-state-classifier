from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SCALER_PATH = ARTIFACTS / "scaler.pkl"
CLIP_BOUND = 10.0


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    """Fit RobustScaler on TRAIN ONLY. Never on the concatenation of train+test."""
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: RobustScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Transform with the fitted scaler and clip to ±CLIP_BOUND (bounds residual outliers)."""
    arr = scaler.transform(X)
    arr = np.clip(arr, -CLIP_BOUND, CLIP_BOUND)
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def save_scaler(scaler: RobustScaler, path: Path = SCALER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path = SCALER_PATH) -> RobustScaler:
    return joblib.load(path)
