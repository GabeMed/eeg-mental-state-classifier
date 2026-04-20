"""Data loading, de-duplication, and stratified splitting.

Invariant: de-dup happens BEFORE the split. Otherwise identical windows could
land in train and test, inflating metrics via memorization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mental-state.csv"
LABEL_COL = "Label"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    """Drop exact-duplicate rows. Runs BEFORE any split."""
    before = len(df)
    out = df.drop_duplicates(ignore_index=True)
    dropped = before - len(out)
    pct = 100.0 * dropped / before if before else 0.0
    return out, dropped, pct


def stratified_split(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split on the Label column."""
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def load_dedup_split(
    path: Path | str = DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """End-to-end: load → dedup → stratified split. Returns (Xtr, Xte, ytr, yte, info)."""
    df = load_raw(path)
    df_clean, dropped, pct = deduplicate(df)
    X_train, X_test, y_train, y_test = stratified_split(df_clean)
    info = {
        "rows_raw": len(df),
        "rows_after_dedup": len(df_clean),
        "duplicates_dropped": dropped,
        "duplicate_pct": round(pct, 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_class_props": y_train.value_counts(normalize=True).round(3).to_dict(),
        "test_class_props": y_test.value_counts(normalize=True).round(3).to_dict(),
        "n_features": X_train.shape[1],
    }
    return X_train, X_test, y_train, y_test, info
