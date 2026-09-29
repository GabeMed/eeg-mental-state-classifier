# Entry points for the EEG mental-state pipeline. Run `make help` for the list.

PY ?= .venv/bin/python
RUN = PYTHONPATH=. $(PY)
DATA = data/mental-state.csv
KAGGLE_DATASET = birdy654/eeg-brainwave-dataset-mental-state

.PHONY: help setup data check-data eda train evaluate figures all app test test-fast lint

help:
	@echo "make setup      create .venv (Python 3.12) and install the pinned environment"
	@echo "make data       download $(DATA) with the Kaggle CLI (needs your own Kaggle API token)"
	@echo "make all        train -> evaluate -> engagement figures (needs $(DATA))"
	@echo "make eda        rebuild and execute notebooks/01_eda.ipynb"
	@echo "make app        launch the Streamlit app on the committed artifacts"
	@echo "make test       lint + full test suite on synthetic data (no dataset needed)"
	@echo "make test-fast  unit tests only, skips the end-to-end pipeline run"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.lock

data:
	@command -v kaggle >/dev/null 2>&1 || { \
		echo "Kaggle CLI not found. Either 'uv pip install kaggle' and set up ~/.kaggle/kaggle.json,"; \
		echo "or download the CSV from https://www.kaggle.com/datasets/$(KAGGLE_DATASET)"; \
		echo "and save it as $(DATA)."; exit 1; }
	kaggle datasets download -d $(KAGGLE_DATASET) -p data --unzip

check-data:
	@test -f $(DATA) || { echo "$(DATA) not found. Run 'make data' or see README > Data."; exit 1; }

eda: check-data
	$(RUN) scripts/build_eda_notebook.py

train: check-data
	$(RUN) scripts/train.py

evaluate: check-data
	$(RUN) scripts/evaluate.py

figures: check-data
	$(RUN) scripts/plot_engagement.py

all: train evaluate figures

app:
	$(PY) -m streamlit run app/streamlit_app.py

lint:
	$(PY) -m ruff check .

test: lint
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"
