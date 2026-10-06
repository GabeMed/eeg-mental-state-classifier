"""Shared fixtures. See tests/synthetic.py for how the synthetic data is built."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.synthetic import make_synthetic_dataset


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_dataset()


@pytest.fixture
def small_synthetic_df() -> pd.DataFrame:
    return make_synthetic_dataset(n_rows=240, n_duplicates=20, columns=[f"f{i}" for i in range(40)])


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    path = tmp_path / "mental-state.csv"
    synthetic_df.to_csv(path, index=False)
    return path
