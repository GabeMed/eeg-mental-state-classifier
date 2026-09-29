"""Synthetic stand-in for the Kaggle CSV.

The real dataset is not redistributed with the repo, so every test runs on
synthetic data that has the same schema (the 988 columns in
artifacts/columns.json + a float `Label` in {0.0, 1.0, 2.0}), a few exact
duplicate rows, and heavy-tailed outliers, which is what the leakage and
scaling logic has to handle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLUMNS = json.loads((ROOT / "artifacts" / "columns.json").read_text())


def make_synthetic_dataset(
    n_rows: int = 300,
    n_duplicates: int = 15,
    columns: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Balanced 3-class data with class-dependent means, outliers and exact duplicates."""
    rng = np.random.default_rng(seed)
    columns = COLUMNS if columns is None else columns
    n_features = len(columns)

    y = np.tile([0.0, 1.0, 2.0], n_rows // 3 + 1)[:n_rows]
    X = rng.normal(size=(n_rows, n_features))
    informative = rng.choice(n_features, size=max(1, n_features // 10), replace=False)
    X[:, informative] += y[:, None] * 1.5

    # A few features on a much larger scale with sparse spikes, like covM_* in the real data.
    wide = rng.choice(n_features, size=max(1, n_features // 20), replace=False)
    X[:, wide] *= 1_000.0
    spikes = rng.random(X[:, wide].shape) < 0.01
    X[:, wide] += spikes * 500_000.0

    df = pd.DataFrame(X, columns=columns)
    df["Label"] = y

    dup_idx = rng.choice(n_rows, size=n_duplicates, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write a synthetic mental-state CSV.")
    parser.add_argument("out", type=Path)
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_dataset(n_rows=args.rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
