# TASKS.md — Atomic execution

Legend:
- **[CLAUDE]** — do with me. Involves a DS decision, not boilerplate. A moment to understand.
- **[CURSOR]** — delegate to Cursor/a cheaper LLM. It is translating instructions → code.
- **[T]** — mandatory check before marking as done.

Ordered by phase. Each phase has a time ceiling (PLAN §3).

---

## Phase 0 — Setup (20 min)

- [ ] **T1** [CLAUDE, 10min] Download the Kaggle dataset and inspect: shape, the names of the 988 columns, whether subject IDs exist, presence/absence of band names in the columns.
  - **[T]** `df.shape == (2479, 989)` and we print the first 20 column names.
  - **Decision this task resolves:** D2 (BATR vs fallback), D4 (LOSO or not).
- [ ] **T2** [CURSOR, 5min] `git init`, create the folder structure (§D5), `.gitignore` for `data/` and `artifacts/`.
- [ ] **T3** [CURSOR, 5min] `requirements.txt` with: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit, jupyter.

---

## Phase 1 — EDA (45 min)

- [ ] **T4** [CLAUDE, 15min] Notebook `01_eda.ipynb`: global statistics, confirmation of the 4.6% duplicates, class distribution, outlier detection (the 530656 from the PLAN).
  - **[T]** final cell prints "OK: dataset has 2479 rows, 3 balanced classes, N duplicates detected, M outliers >|1e4|."
  - **Teaching moment:** why look at outliers BEFORE choosing a scaler.
- [ ] **T5** [CURSOR, 15min] Plots: class barplot, correlation heatmap (top-50 features by variance), boxplots of 6 features by class.
  - **[T]** 5 figures generated, all with a title and legend.
- [ ] **T6** [CLAUDE, 15min] 1-way ANOVA per feature to rank "discriminative features" between states. Print the top-20.
  - **[T]** DataFrame sorted by F-stat, descending.
  - **Teaching moment:** F-stat as a quick proxy for "this feature carries class signal".

---

## Phase 2 — Feature pipeline (30 min)

- [ ] **T7** [CLAUDE, 15min] `src/data.py`: `load_raw()`, `deduplicate()`, `stratified_split()` with `random_state=42`. **De-dup BEFORE the split.**
  - **[T]** `len(train) + len(test) == len(df_deduped)`, `train.Label.value_counts(normalize=True)` ≈ stratified.
  - **Teaching moment:** why de-dup before the split avoids subtle leakage.
- [ ] **T8** [CLAUDE, 10min] `src/features.py`: `RobustScaler` fit on train, transform on test. Save to `artifacts/scaler.pkl`.
  - **[T]** `train_scaled.median(axis=0)` ≈ 0, `test_scaled` uses training statistics.
  - **Teaching moment:** why `fit` only on train (data leakage 101).
- [ ] **T9** [CURSOR, 5min] Save `artifacts/columns.json` with the ordered list of the 988 columns (the app needs it to validate uploads).

---

## Phase 3 — Models (45 min)

- [ ] **T10** [CLAUDE, 15min] `src/models.py::train_logreg(X_train, y_train)`: LogReg L2, `max_iter=2000`, stratified 5-fold CV, report macro-F1 mean±std.
  - **[T]** CV score > 0.33 (chance). Model saved to `artifacts/logreg.pkl`.
  - **Teaching moment:** why LogReg + L2 is the honest baseline.
- [ ] **T11** [CLAUDE, 15min] `src/models.py::train_xgb(X_train, y_train)`: default XGBoost + `eval_metric='mlogloss'`, same CV, same reporting.
  - **[T]** CV score > LogReg, saved to `artifacts/xgb.pkl`.
  - **Teaching moment:** the LogReg→XGB gap measures how much of the problem is non-linear.
- [ ] **T12** [CLAUDE, 15min] Final evaluation on the locked test set. Confusion matrices + classification_report for both.
  - **[T]** 2 confusion matrices rendered + 2 reports printed.
  - **Teaching moment:** why look at the confusion matrix, not just accuracy.

---

## Phase 4 — Engagement score (30 min)

- [ ] **T13** [CLAUDE, 20min] `src/engagement.py`: implement BATR if D2 allows it, otherwise the fallback. Normalize to 0–100.
  - **[T]** score.min() ≥ 0, score.max() ≤ 100, no NaN.
  - **Teaching moment:** what "coherent" means for a score with no direct ground truth.
- [ ] **T14** [CURSOR, 10min] Validation: boxplot of the score by class + scatter `score vs P(concentrating)` + Pearson correlation.
  - **[T]** 2 figures + 1 correlation number printed.

---

## Phase 5 — Streamlit (45 min)

- [ ] **T15** [CURSOR, 15min] `app/streamlit_app.py`: layout with `st.file_uploader`, CSV reading, schema validation against `artifacts/columns.json`.
  - **[T]** app runs locally and rejects a CSV with wrong columns with a clear message.
- [ ] **T16** [CLAUDE, 20min] Integration: load scaler + model, apply them, compute the engagement score, render the result.
  - **[T]** uploading `sample_input.csv` shows the predicted class + score for each row.
  - **Teaching moment:** why the app must load the SAME scaler used in training.
- [ ] **T17** [CURSOR, 10min] Polish: title, description explaining the expected schema, button to download `sample_input.csv`.

---

## Phase 6 — Report + README (25 min)

- [ ] **T18** [CURSOR, 10min] `README.md`: how to install, how to run the EDA, how to run the app, how to run the tests.
  - **[T]** following the README from scratch in a clean terminal works.
- [ ] **T19** [CLAUDE, 15min] `REPORT.md` with sections:
  1. What worked
  2. What did not work
  3. What I would do differently
  4. **Plan vs. Outcome** (2-column table)
  - **[T]** each section has at least 3 specific bullets, none generic.
  - **Teaching moment:** the difference between an honest report and a defensive one.

---

## Phase 7 — Buffer (20 min)

- [ ] **T20** [CLAUDE, 20min] Git: one commit per phase (retroactive is fine), push, final app test in a clean session.

---

## Teaching gates (where to pause and discuss)

Moments where I will EXPLAIN before doing — do not skip them:

1. **T1 output** — what the column names tell us about choices D2 and D4.
2. **T7** — why de-dup before the split (subtle leakage even without labels).
3. **T8** — fit only on train (the "hello world" of data leakage).
4. **T11** — interpreting the LogReg vs XGB gap.
5. **T13** — validity of a score with no direct ground truth.
6. **T19** — honest framing of the report.

Everything else is execution. I delegate to Cursor whatever can be expressed as "write a function that does X, Y, Z".
