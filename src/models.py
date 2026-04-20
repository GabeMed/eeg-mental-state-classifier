"""Model training with honest pipeline-based cross-validation.

Two models by design:
- LogReg L2 — linear baseline. Anchors expectations for how much signal is
  capturable by a linear decision boundary.
- XGBoost    — non-linear tree ensemble. Gap between the two measures how much
  of the problem is genuinely non-linear.

CV protocol:
- Scaler lives INSIDE the CV pipeline. Each fold fits a fresh RobustScaler on
  its own train subset, then applies transform+clip to the val subset. This
  avoids the subtle leak of fitting the scaler on the full train before CV.
- Final models are fit on the full training set using the external scaler
  (`src/features.py`) so that `artifacts/scaler.pkl` is saved and can be
  reused at inference time in the Streamlit app.

Same StratifiedKFold(5) and macro-F1 for both models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from xgboost import XGBClassifier

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RANDOM_STATE = 42
CV_FOLDS = 5
CLIP_BOUND = 10.0


def _clip_to_bound(X):
    return np.clip(X, -CLIP_BOUND, CLIP_BOUND)


def logreg_estimator() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def xgb_estimator() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_pipeline(estimator) -> Pipeline:
    """Mirror the production preprocessing (RobustScaler + clip ±10) inside a CV pipeline."""
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clip", FunctionTransformer(_clip_to_bound, validate=False)),
            ("clf", estimator),
        ]
    )


def honest_cv(
    estimator, X_train_raw: pd.DataFrame, y_train: pd.Series, folds: int = CV_FOLDS
) -> np.ndarray:
    """Stratified K-fold CV with scaler INSIDE the pipeline. Returns macro-F1 per fold."""
    pipe = build_pipeline(estimator)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipe, X_train_raw, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)


def train_logreg(
    X_train_scaled: pd.DataFrame, y_train: pd.Series
) -> Tuple[LogisticRegression, dict]:
    """Fit the final LogReg on already-scaled X_train and persist it."""
    model = logreg_estimator()
    model.fit(X_train_scaled, y_train)
    info = {"n_features": X_train_scaled.shape[1]}
    joblib.dump(model, ARTIFACTS / "logreg.pkl")
    return model, info


def train_xgboost(
    X_train_scaled: pd.DataFrame, y_train: pd.Series
) -> Tuple[XGBClassifier, dict]:
    """Fit the final XGBoost on already-scaled X_train and persist it."""
    model = xgb_estimator()
    model.fit(X_train_scaled, y_train)
    info = {"n_features": X_train_scaled.shape[1]}
    joblib.dump(model, ARTIFACTS / "xgb.pkl")
    return model, info
