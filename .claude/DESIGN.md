# DESIGN.md — Technical decisions

A short document to settle the 4 gray areas before coding.

---

## D1. Streamlit schema

**Decision:** the app accepts a CSV in the same format as the training dataset (988 feature columns + an optional `Label` column).

**Reason:** within the time-box it is not feasible to package raw-signal → feature extraction. Document it clearly in the UI and the README: "this is an inference platform over pre-extracted features; extraction from the raw signal is future work."

**Risk mitigation:** include a `sample_input.csv` in the repo (a few rows from the test set) so anyone can try the app in 10 seconds.

---

## D2. Engagement score — RESOLVED: **Option B (model-based proxy)**

**T1 inspection result:** no column contains `alpha`/`beta`/`theta`/`gamma`/`delta`. The frequency features are `freq_XXX_N`, where X ranges from 010 to 750 and N is the electrode index (0–3).

**Frequency axiom (discovered after T1, via external research in the `jordan-bird/eeg-feature-generation` repo):**
- Data resampled to 150 Hz, FFT window of 148 samples → each bin `freq_XXX` ≈ X/100 Hz.
- Gap at `freq_486`→`freq_517`: 50 Hz notch filter (UK mains power).
- So the classic bands map to:
  - theta (4–8 Hz): `freq_040`–`freq_080`
  - alpha (8–13 Hz): `freq_080`–`freq_130`
  - beta (13–30 Hz): `freq_130`–`freq_300`

**Decision adopted:** **score = P(concentrating) from the logistic regression, normalized to 0–100.**

- **Why B and not BATR now:** the time-box cannot accommodate validating the band mappings above while still guaranteeing an interpretable BATR. The model-based proxy is coherent by construction with the dataset's own ground truth.
- **What becomes future work (documented in the REPORT):** implement classic BATR using the band mapping derived from the frequency axiom above. Add it as a panel in the app.
- **Proxy validation:** correlation between the score and the `concentrating` class on the test set, boxplot by class.

---

## D3. Validation protocol

**Decision:**
1. Stratified 80/20 split (train/test) with `random_state=42`. The test set stays locked until the end.
2. On the training set: `StratifiedKFold(n_splits=5)` for model selection/comparison.
3. Final reported number: test-set metrics (1 number per model). CV is mentioned as internal stability validation.

**Reason:** 988 features × ~2364 rows → CV alone is optimistic if we only look at the mean. A fixed holdout gives a number anyone can reproduce exactly and that is not used for any decision.

---

## D4. Subject leakage — RESOLVED: document as a limitation

**T1 inspection result:** the aggregated Kaggle CSV contains no subject/participant/session IDs. LOSO is not feasible without downloading Bird's original repository (jordan-bird/eeg-feature-generation) and rebuilding the pipeline from the raw files, which is out of scope.

**Decision adopted:** window-level stratified CV, with a stratified 80/20 holdout. The limitation is recorded honestly in the REPORT as "real deployment on a new user would require a LOSO protocol; not tested on this dataset."

---

## D5. Repository structure

```
eeg-mental-state-classifier/
├── data/                       # Kaggle CSV (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data.py                 # load + dedup + split
│   ├── features.py             # RobustScaler + derivations
│   ├── models.py               # LR + XGB training
│   └── engagement.py           # BATR or fallback
├── app/
│   └── streamlit_app.py
├── artifacts/                  # model.pkl + scaler.pkl + columns.json
├── sample_input.csv
├── PLAN.md
├── REPORT.md
└── README.md
```

**Why separate `src/` from `notebooks/`:** notebooks are for exploring and communicating; `.py` modules are the source of truth the app imports. Never rewrite logic in the app that already exists in `src/`.
