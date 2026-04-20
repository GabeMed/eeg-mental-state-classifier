"""Model training with stratified CV.

Two models by design:
- LogReg L2 — linear baseline. Anchors expectations for how much signal is
  capturable by a linear decision boundary.
- XGBoost    — non-linear tree ensemble. Gap between the two measures how much
  of the problem is genuinely non-linear.

Same CV protocol for both: StratifiedKFold(5), macro-F1 as primary metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RANDOM_STATE = 42
CV_FOLDS = 5


def cv_scores(
    model, X: pd.DataFrame, y: pd.Series, folds: int = CV_FOLDS
) -> np.ndarray:
    """Stratified K-fold CV. Returns macro-F1 per fold."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)


def train_logreg(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[LogisticRegression, dict]:
    """L2-regularized multinomial logistic regression baseline."""
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    fold_scores = cv_scores(model, X_train, y_train)
    model.fit(X_train, y_train)
    info = {
        "cv_mean": float(fold_scores.mean()),
        "cv_std": float(fold_scores.std()),
        "cv_folds": fold_scores.tolist(),
        "n_features": X_train.shape[1],
    }
    joblib.dump(model, ARTIFACTS / "logreg.pkl")
    return model, info


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[XGBClassifier, dict]:
    """Gradient-boosted trees. Sensible defaults, no hyperparameter tuning in project scope."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    fold_scores = cv_scores(model, X_train, y_train)
    model.fit(X_train, y_train)
    info = {
        "cv_mean": float(fold_scores.mean()),
        "cv_std": float(fold_scores.std()),
        "cv_folds": fold_scores.tolist(),
        "n_features": X_train.shape[1],
    }
    joblib.dump(model, ARTIFACTS / "xgb.pkl")
    return model, info
