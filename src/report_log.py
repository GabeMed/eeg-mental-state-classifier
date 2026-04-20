"""Append-only run log for this project.

Usage:
    from src.report_log import log_finding, log_decision, log_doubt, log_metric
    log_finding("Outlier ratio", "max/p99 = 1704x — confirms RobustScaler need")
    log_doubt("Is RobustScaler really different?", resolution="For ~10% of features yes; switched to RobustScaler + clip after measuring")

Log is written to artifacts/run_log.jsonl (one JSON object per line, append-only).
Use render_log.py to produce the REPORT.md storyline section.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "run_log.jsonl"


def _append(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_finding(title: str, detail: str) -> None:
    """Empirical discovery about the data or pipeline."""
    _append({"type": "finding", "title": title, "detail": detail})


def log_decision(title: str, detail: str) -> None:
    """Deliberate methodological choice, with reasoning."""
    _append({"type": "decision", "title": title, "detail": detail})


def log_doubt(question: str, resolution: Optional[str] = None) -> None:
    """Question or concern that came up during the project. Pass resolution when resolved."""
    _append(
        {
            "type": "doubt",
            "question": question,
            "resolution": resolution,
            "status": "resolved" if resolution else "open",
        }
    )


def log_metric(name: str, value, note: Optional[str] = None) -> None:
    """Numeric result worth citing in the report (CV score, outlier ratio, etc.)."""
    _append({"type": "metric", "name": name, "value": value, "note": note})
