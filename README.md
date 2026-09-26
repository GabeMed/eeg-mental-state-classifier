# EEG Mental State Classifier

A personal project exploring mental-state classification from EEG on the Kaggle dataset `birdy654/eeg-brainwave-dataset-mental-state`. The problem has 3 classes (`relaxed`, `neutral`, `concentrating`) and 988 features already extracted from the signal, which was recorded with a 4-electrode Muse headband (TP9, AF7, AF8, TP10). This repo contains the end-to-end pipeline: EDA notebook generation, training, evaluation artifacts, engagement plots, and a Streamlit app for inference.

## Headline Results

| Model | Test macro-F1 |
|---|---:|
| Logistic Regression (L2) | **0.9528** |
| XGBoost | **0.9704** |

## Repo Layout

```text
eeg-mental-state-classifier/
├── .claude/
│   ├── DESIGN.md
│   ├── PLAN.md
│   └── TASKS.md
├── app/
│   └── streamlit_app.py
├── artifacts/
│   ├── columns.json
│   ├── evaluation.json
│   ├── logreg.pkl
│   ├── run_log.jsonl
│   ├── scaler.pkl
│   └── xgb.pkl
├── data/
│   ├── example-mental-state.csv
│   └── mental-state.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── figures/
│       ├── confusion_logreg.png
│       ├── confusion_xgb.png
│       ├── engagement_boxplot.png
│       ├── engagement_histogram.png
│       ├── engagement_scatter.png
│       └── family_importance.png
├── scripts/
│   ├── build_eda_notebook.py
│   ├── evaluate.py
│   ├── plot_engagement.py
│   ├── render_log.py
│   ├── save_columns.py
│   ├── seed_log.py
│   └── train.py
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── engagement.py
│   ├── features.py
│   ├── models.py
│   └── report_log.py
├── .gitignore
├── CLAUDE.md
├── README.md
├── REPORT.md
└── requirements.txt
```

## Setup (`uv`, Python 3.12)

`scikit-learn==1.4.2` does not run on Python 3.14 in this project. Use Python 3.12.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

1. Download `mental-state.csv` from Kaggle (`birdy654/eeg-brainwave-dataset-mental-state`).
2. Place the file at `data/mental-state.csv` (with a hyphen).
3. The file is ignored by git (`.gitignore`).

## How To Run (recommended order)

### a) Generate the EDA notebook

```bash
PYTHONPATH=. python scripts/build_eda_notebook.py
```

Creates or updates `notebooks/01_eda.ipynb`.

### b) Train the scaler and both models

```bash
PYTHONPATH=. python scripts/train.py
```

One command produces `artifacts/{scaler.pkl, logreg.pkl, xgb.pkl, columns.json}` and logs the honest CV (scaler inside the `sklearn.Pipeline`, refit per fold) to `artifacts/run_log.jsonl`.

### c) Evaluate on the holdout + confusion matrices

```bash
PYTHONPATH=. python scripts/evaluate.py
```

Produces `artifacts/evaluation.json` and figures in `notebooks/figures/`.

### d) Generate the engagement figures

```bash
PYTHONPATH=. python scripts/plot_engagement.py
```

### e) Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`, click **Generate sample CSV** in the sidebar, and upload the generated CSV through the file uploader.

## Artifacts

- `artifacts/logreg.pkl`: Logistic Regression model trained on the scaled training set.
- `artifacts/xgb.pkl`: XGBoost model trained on the same training set.
- `artifacts/scaler.pkl`: `RobustScaler` fit on the training set (clipping is applied at use time).
- `artifacts/columns.json`: ordered catalog of the 988 columns expected by the pipeline and the app.
- `artifacts/evaluation.json`: final test metrics and serialized confusion matrices.
- `artifacts/run_log.jsonl`: append-only log of the project's metrics, decisions, and findings.

## Context

For the technical narrative and project decisions, read `REPORT.md` and `.claude/PLAN.md`.
