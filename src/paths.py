"""Filesystem locations shared by src/, scripts/ and the app.

Each location can be overridden with an environment variable. The tests use
this to run the full train → evaluate pipeline on a synthetic dataset in a
temporary directory, so the committed artifacts are never overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(os.environ.get("EEG_DATA_PATH", ROOT / "data" / "mental-state.csv"))
ARTIFACTS = Path(os.environ.get("EEG_ARTIFACTS_DIR", ROOT / "artifacts"))
FIGURES = Path(os.environ.get("EEG_FIGURES_DIR", ROOT / "notebooks" / "figures"))
