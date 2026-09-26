"""Render artifacts/run_log.jsonl as a markdown storyline for REPORT.md.

Usage:
    python scripts/render_log.py > /tmp/storyline.md

Sections produced:
  - Findings (chronological)
  - Decisions (chronological)
  - Doubts & Resolutions (the thinking-out-loud record)
  - Key Metrics (flat table)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


LOG = Path(__file__).resolve().parents[1] / "artifacts" / "run_log.jsonl"


def load_entries() -> list[dict]:
    if not LOG.exists():
        return []
    with LOG.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def render(entries: list[dict]) -> str:
    findings = [e for e in entries if e["type"] == "finding"]
    decisions = [e for e in entries if e["type"] == "decision"]
    doubts = [e for e in entries if e["type"] == "doubt"]
    metrics = [e for e in entries if e["type"] == "metric"]

    out = ["# Storyline — reasoning during the project\n"]
    out.append("Auto-generated from `artifacts/run_log.jsonl`. Do not edit by hand.\n")

    if findings:
        lines = []
        for f in findings:
            lines.append(f"- **{f['title']}**  \n  {f['detail']}")
        out.append(section("Empirical findings", "\n".join(lines)))

    if decisions:
        lines = []
        for d in decisions:
            lines.append(f"- **{d['title']}**  \n  {d['detail']}")
        out.append(section("Methodological decisions", "\n".join(lines)))

    if doubts:
        lines = []
        for d in doubts:
            status = "✅" if d.get("status") == "resolved" else "🟡 open"
            lines.append(f"- {status} **{d['question']}**")
            if d.get("resolution"):
                lines.append(f"  Resolution: {d['resolution']}")
        out.append(section("Doubts raised during the project", "\n".join(lines)))

    if metrics:
        lines = ["| metric | value | note |", "|---|---|---|"]
        for m in metrics:
            note = m.get("note") or ""
            lines.append(f"| `{m['name']}` | {m['value']} | {note} |")
        out.append(section("Key metrics", "\n".join(lines)))

    return "\n".join(out)


def main():
    entries = load_entries()
    sys.stdout.write(render(entries))


if __name__ == "__main__":
    main()
