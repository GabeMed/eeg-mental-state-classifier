"""Engagement score — continuous 0–100 signal of attentional state.

Design choice (logged in DESIGN.md §D2): model-based proxy.
  score = 50 * (P(concentrating) - P(relaxed) + 1)

Rationale:
  - P(concentrating) - P(relaxed) captures the attentional polarity.
    (+1 when fully concentrated, -1 when fully relaxed, 0 when neutral/balanced).
  - Shifted to [0, 100] for interpretability.
  - Uses LogReg (calibrated probabilities), not XGBoost (probabilities are less
    calibrated unless wrapped in CalibratedClassifierCV, which we skipped for scope).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


RELAXED_LABEL = 0.0
CONCENTRATING_LABEL = 2.0


def engagement_score(model: LogisticRegression, X_scaled: pd.DataFrame) -> np.ndarray:
    """Return engagement score per row, 0–100.

    Requires a classifier with `predict_proba` and class labels matching
    {0.0: relaxed, 1.0: neutral, 2.0: concentrating}.
    """
    probs = model.predict_proba(X_scaled)
    classes = list(model.classes_)
    idx_relaxed = classes.index(RELAXED_LABEL)
    idx_concentrating = classes.index(CONCENTRATING_LABEL)

    polarity = probs[:, idx_concentrating] - probs[:, idx_relaxed]  # [-1, +1]
    score = 50.0 * (polarity + 1.0)  # [0, 100]
    return np.clip(score, 0.0, 100.0)
