"""EDA helpers for the EEG mental-state project.

Pure functions — notebook imports these and renders outputs.
Keeping logic here means the notebook stays readable and the code is testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

DATA_PATH = "data/mental-state.csv"
LABEL_COL = "Label"
CLASS_NAMES = {0.0: "relaxed", 1.0: "neutral", 2.0: "concentrating"}


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def describe_dataset(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=[LABEL_COL], errors="ignore"
    )
    return {
        "shape": df.shape,
        "n_features": numeric.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": float(df.duplicated().mean() * 100),
        "class_counts": df[LABEL_COL].value_counts().to_dict(),
        "class_proportions": df[LABEL_COL]
        .value_counts(normalize=True)
        .round(3)
        .to_dict(),
        "nulls_total": int(df.isna().sum().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }


def outlier_profile(df: pd.DataFrame) -> dict:
    """Quantifies outlier severity to motivate RobustScaler over StandardScaler."""
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=[LABEL_COL], errors="ignore"
    )
    flat = numeric.values.flatten()
    return {
        "max_abs": float(np.abs(flat).max()),
        "p99_abs": float(np.percentile(np.abs(flat), 99)),
        "p999_abs": float(np.percentile(np.abs(flat), 99.9)),
        "median": float(np.median(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "ratio_max_to_p99": float(
            np.abs(flat).max() / max(np.percentile(np.abs(flat), 99), 1e-9)
        ),
    }


def anova_feature_ranking(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Rank features by one-way ANOVA F-statistic between the 3 mental states.

    High F = group means differ relative to within-group variance → discriminative feature.
    This is a fast, univariate proxy for "which features carry class signal" — NOT a
    substitute for multivariate feature importance from the model.
    """
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    groups = [
        df.loc[df[LABEL_COL] == lbl, feature_cols]
        for lbl in sorted(df[LABEL_COL].unique())
    ]
    f_stats = []
    for col in feature_cols:
        samples = [g[col].values for g in groups]
        f, p = f_oneway(*samples)
        f_stats.append((col, f, p))
    out = pd.DataFrame(f_stats, columns=["feature", "f_stat", "p_value"])
    out = out.sort_values("f_stat", ascending=False).reset_index(drop=True)
    return out.head(top_n) if top_n else out


def top_variance_features(df: pd.DataFrame, top_n: int = 50) -> list[str]:
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    variances = df[feature_cols].var().sort_values(ascending=False)
    return variances.head(top_n).index.tolist()


def class_name(label: float) -> str:
    return CLASS_NAMES.get(label, str(label))
