# EEG Mental State Classifier

Projeto pessoal explorando classificação de estados mentais a partir de EEG no dataset Kaggle `birdy654/eeg-brainwave-dataset-mental-state`. O problema tem 3 classes (`relaxed`, `neutral`, `concentrating`) e 988 features já extraídas do sinal, coletado com headband Muse de 4 eletrodos (TP9, AF7, AF8, TP10). Este repo contém o pipeline end-to-end: geração de notebook de EDA, treino, artefatos de avaliação, plots de engajamento, e um app Streamlit para inferência.

## Headline Results

| Modelo | Test macro-F1 |
|---|---:|
| Logistic Regression (L2) | **0.9528** |
| XGBoost | **0.9704** |

## Repo Layout

```text
synapse/
├── .claude/
│   ├── DESIGN.md
│   ├── PLANO.md
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
│   └── seed_log.py
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── engagement.py
│   ├── features.py
│   ├── models.py
│   └── report_log.py
├── .gitignore
├── CLAUDE.md
├── REPORT.md
└── requirements.txt
```

## Setup (`uv`, Python 3.12)

`scikit-learn==1.4.2` não roda em Python 3.14 neste projeto. Use Python 3.12.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

1. Baixe `mental-state.csv` no Kaggle (`birdy654/eeg-brainwave-dataset-mental-state`).
2. Coloque o arquivo em `data/mental-state.csv` (com hífen).
3. O arquivo é ignorado no git (`.gitignore`).

## How To Run (ordem recomendada)

### a) Gerar notebook de EDA

```bash
PYTHONPATH=. python scripts/build_eda_notebook.py
```

Gera/atualiza `notebooks/01_eda.ipynb`.

### b) Treinar scaler + ambos os modelos

```bash
PYTHONPATH=. python scripts/train.py
```

Um comando gera `artifacts/{scaler.pkl, logreg.pkl, xgb.pkl, columns.json}` e loga a CV honesta (scaler dentro da `sklearn.Pipeline`, re-ajustado por fold) em `artifacts/run_log.jsonl`.

### c) Avaliar no holdout + matrizes de confusão

```bash
PYTHONPATH=. python scripts/evaluate.py
```

Produz `artifacts/evaluation.json` e figuras em `notebooks/figures/`.

### d) Gerar figuras de engagement

```bash
PYTHONPATH=. python scripts/plot_engagement.py
```

### e) Subir app Streamlit

```bash
streamlit run app/streamlit_app.py
```

Abra `http://localhost:8501`, clique em **Gerar CSV de exemplo** na sidebar e faça upload do CSV gerado no `file_uploader`.

## Artifacts

- `artifacts/logreg.pkl`: modelo Logistic Regression treinado no treino escalado.
- `artifacts/xgb.pkl`: modelo XGBoost treinado no mesmo conjunto de treino.
- `artifacts/scaler.pkl`: `RobustScaler` ajustado no treino (com clip aplicado no uso).
- `artifacts/columns.json`: ordem/catálogo das 988 colunas esperadas no pipeline e no app.
- `artifacts/evaluation.json`: métricas finais de teste e matrizes de confusão serializadas.
- `artifacts/run_log.jsonl`: log append-only de métricas, decisões e achados da projeto.

## Contexto

Para narrativa técnica e decisões da projeto, leia `REPORT.md` e `.claude/PLANO.md`.
