# CLAUDE.md — Agent Orchestration Guidelines

This file tells coding agents how to collaborate on this repo. It is durable
guidance, not task-level instructions — tasks live in `.claude/TASKS.md`.

## 1. Model-tier delegation

Two agents operate on this repo:

- **Claude (Opus 4.7)** — reasoning, judgment calls, DS decisions, code review,
  narrative writing (REPORT.md, log entries), tricky debugging.
- **Cursor (cheaper tier)** — mechanical translation of spec → code: plotting,
  boilerplate, CSV writers, Streamlit layout, README scaffolding, tests.

**Rule:** before doing anything, classify the task. If the spec is unambiguous
and the judgment-per-line-of-code is low, write a Cursor prompt and hand it
off. Act as senior reviewer, not typist.

Tasks in `.claude/TASKS.md` are explicitly tagged `[CLAUDE]` or `[CURSOR]`.
Do not absorb `[CURSOR]` work into `[CLAUDE]` sessions to save round-trips —
that defeats the cost model and produces lower-quality code (Cursor is better
at boilerplate than Claude is at *writing* boilerplate patiently).

## 2. Append-only decision log

`artifacts/run_log.jsonl` is the project's memory. Every non-trivial finding,
decision, doubt, and metric is logged via `src/report_log.py`:

- `log_finding(title, detail)` — empirical discovery
- `log_decision(title, detail)` — methodological choice + why
- `log_doubt(question, resolution=...)` — open question or resolved concern
- `log_metric(name, value, note=...)` — numeric result worth citing

**Rule:** if a future reviewer would ask "why did you do X?", log it. The final
REPORT.md is a *consequence* of the log, not a separate exercise. `scripts/render_log.py`
turns JSONL into a markdown storyline.

Before adding a new entry, check for outdated ones. If a decision was
superseded (e.g., RobustScaler re-justified after empirical check), log the
update — do not silently overwrite history.

## 3. Knowledge verification chain

When researching, designing, or making a technical claim, follow this chain in
order. Never skip steps.

1. **This codebase** — read the relevant `src/*.py`, notebooks, artifacts.
2. **Project docs** — `.claude/PLANO.md`, `.claude/DESIGN.md`, `REPORT.md`, inline docstrings.
3. **Authoritative external sources** — official library docs, original papers
   (Pope 1995 for BATR, Posner & Petersen 1990 for attention hemispheric bias),
   dataset provenance repos (`jordan-bird/eeg-feature-generation`).
4. **Web search** — only after 1–3 were tried.
5. **Flag as uncertain** — "I'm not sure about X, here is my reasoning, verify before acting."

**Never fabricate.** Do not invent column semantics, API signatures, or
parameter meanings. Uncertainty is always preferable to a confident-sounding
guess. This was violated once in this project (claim of "2 subjects" was
unverified and wrong — real answer: 4 subjects). That entry is logged as a
correction to serve as a reminder.

## 4. Measure, don't assert

When a design choice is debatable (which scaler, which metric, which model),
run the comparison before committing to a direction. Half a paragraph of
justification in a doc is worth less than three numbers in a notebook cell.

Example from this repo: the RobustScaler vs StandardScaler decision was
re-opened after a skeptical question. The empirical comparison (std/IQR per
feature) changed the framing of the choice and strengthened the final
decision. This pattern is the default.

## 5. Gates and teaching moments

Some steps in `.claude/TASKS.md` are marked as **teaching moments** — pause
before executing, surface the reasoning, and only proceed after it is stated.
These are the steps where the risk is not bugs but *unlearned lessons*.
Current list lives in `.claude/TASKS.md` §Gates.

## 6. Repo conventions

- Python 3.12 (sklearn 1.4.2 incompatible with 3.14).
- `uv` from Astral for env management — not `venv`, not `pip` directly.
- `PYTHONPATH=.` when running scripts that import from `src/`.
- `artifacts/` is committed (so the app runs out-of-the-box).
- `data/mental-state.csv` is gitignored (Kaggle distribution terms + size).
- `.claude/CHALLENGE.md` is gitignored (external brief, not for this repo).
- Artifacts and log files are authoritative for the historical record. Do not
  regenerate them cosmetically — regenerate only when the underlying code or
  data changed.

## 7. Scope discipline

Do not:

- Add features, refactors, or abstractions beyond the task.
- Write backwards-compatibility shims for a codebase that has no users yet.
- Add comments explaining *what* code does (the code does that). Only add
  comments explaining *why* a non-obvious choice was made.
- Mark a task complete if any gate check (test, CV score, manual verification)
  has not passed.
