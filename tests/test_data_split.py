"""Leakage-aware split: de-dup happens before the split, and the split is stratified."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import LABEL_COL, deduplicate, load_dedup_split, stratified_split


def _row_set(X: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, X.to_numpy()))


def test_deduplicate_drops_exact_duplicates_only(synthetic_df):
    n_unique = len(synthetic_df.drop_duplicates())
    out, dropped, pct = deduplicate(synthetic_df)

    assert len(out) == n_unique
    assert dropped == len(synthetic_df) - n_unique == 15
    assert pct == 100.0 * dropped / len(synthetic_df)
    assert not out.duplicated().any()
    assert list(out.index) == list(range(len(out)))


def test_deduplicate_keeps_near_duplicates(synthetic_df):
    near = synthetic_df.iloc[[0]].copy()
    near.iloc[0, 0] += 1e-9
    df = pd.concat([synthetic_df.drop_duplicates(), near], ignore_index=True)
    out, dropped, _ = deduplicate(df)
    assert dropped == 0
    assert len(out) == len(df)


def test_no_window_lands_in_both_train_and_test(synthetic_csv):
    X_train, X_test, _, _, _ = load_dedup_split(synthetic_csv)
    assert _row_set(X_train).isdisjoint(_row_set(X_test))


def test_naive_split_without_dedup_would_leak(synthetic_df):
    """Guards the test above: on this data, skipping de-dup does put copies on both sides."""
    X = synthetic_df.drop(columns=[LABEL_COL])
    y = synthetic_df[LABEL_COL]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    assert not _row_set(X_train).isdisjoint(_row_set(X_test))


def test_split_sizes_and_class_balance(synthetic_csv):
    X_train, X_test, y_train, y_test, info = load_dedup_split(synthetic_csv)

    assert len(X_train) + len(X_test) == info["rows_after_dedup"]
    assert info["rows_raw"] - info["duplicates_dropped"] == info["rows_after_dedup"]
    assert abs(len(X_test) / info["rows_after_dedup"] - 0.20) < 0.01

    train_props = y_train.value_counts(normalize=True).sort_index()
    test_props = y_test.value_counts(normalize=True).sort_index()
    assert (train_props - test_props).abs().max() < 0.02

    assert LABEL_COL not in X_train.columns
    assert info["n_features"] == X_train.shape[1] == 988


def test_split_is_deterministic(synthetic_df):
    clean, _, _ = deduplicate(synthetic_df)
    a = stratified_split(clean)
    b = stratified_split(clean)
    assert a[0].index.equals(b[0].index)
    assert a[1].index.equals(b[1].index)

    c = stratified_split(clean, random_state=7)
    assert not a[1].index.equals(c[1].index)
