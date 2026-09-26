# REPORT — EEG Mental State Classification

A time-boxed personal project on `birdy654/eeg-brainwave-dataset-mental-state`. Three classes (`relaxed`, `neutral`, `concentrating`), 988 features pre-extracted by Bird, 4 subjects (2 male, 2 female), Muse headband (TP9, AF7, AF8, TP10) resampled to 150 Hz.

## Headline numbers

| Model | CV macro-F1 (5-fold, train) | Test macro-F1 (473 locked rows) |
|---|---|---|
| LogReg L2 multinomial | 0.9534 ± 0.0068 | **0.9528** |
| XGBoost (n=300, depth=6) | 0.9690 ± 0.0070 | **0.9704** |

The CV is honest: the scaler lives **inside** the cross-validation `sklearn.Pipeline` — `RobustScaler` + ±10 clipping are refit in every fold, so the validation fold never touches statistics fit on itself. CV→test gap: -0.0006 (LogReg), +0.0014 (XGB) — within noise, **no suspicious discrepancy** between CV and holdout. The test set was never seen by the scaler or the models until `scripts/evaluate.py`.

Mean engagement score (0–100) per true class on the test set: relaxed=3.0, neutral=46.5, concentrating=99.5. Spearman `r(score, label_ordinal) = 0.9269`.

---

## 1. What worked

- **De-dup before the split.** 4.64% of the rows (115) were exact duplicates; removing them before the split eliminated the subtle risk of the same window landing in both train and test with inflated scores. 2479 → 2364 → 1891/473.
- **RobustScaler + clipping at ±10, justified empirically.** The initial decision was "use RobustScaler because there are outliers". After a skeptical question ("median ≈ 0, small IQR — does it even make a difference?"), we measured feature by feature: for ~50% of the features the two scalers are nearly identical, but for ~10% (covariance-matrix entries) the `std/IQR` ratio is 30–248×, making `StandardScaler` produce values around 25 where `RobustScaler` produces 6290 on the raw scale. The subsequent clip bounds the residual. Clipped fraction: 0.94% train, 0.99% test — bounded without cutting real mass.
- **Two models with distinct roles.** LogReg for the engagement score (calibrated, interpretable probabilities) and XGBoost for the headline number + per-family importance. It is not "which one wins", it is "which one is for what".
- **Small LogReg→XGB gap (+1.56 pt in CV, +1.76 pt on test), read correctly.** A sign that most of the problem is linearly capturable — Bird's pre-extracted features (FFT, covariance, kurtosis) already do the heavy non-linear lifting, and XGBoost only adds marginal interactions.
- **Importance by family, not by feature.** Aggregating XGBoost's 988 gains into 13 families, `freq` dominates with 46% of the cumulative importance. It matches the univariate ANOVA ranking — two independent angles agreeing is a strong signal.
- **Right hemisphere dominates the ANOVA top-20.** AF8=7, TP10=7, TP9=6, AF7=0 features. Consistent with Posner & Petersen (1990): sustained attention has a right-hemisphere bias. We did not go looking for this finding; it fell into our lap when sorting F-stats.
- **Coherent engagement without direct ground truth.** Monotonicity holds across classes (3.0 < 46.5 < 99.5), within-class variance is non-zero (usable as a continuous signal, not just a disguised 3-level one), and Spearman is 0.93 against the ordinal order.
- **Append-only log as the narrative backbone.** 58 entries in `artifacts/run_log.jsonl` — findings, decisions, doubts with resolutions, metrics. This REPORT is a consequence of the log, not the other way around.

## 2. What did not work (or: what we honestly did not do)

- **Classic BATR (Pope 1995) was not implemented.** The original plan (`.claude/PLAN.md`) called for `beta/(alpha+theta)`, but the Kaggle columns carry no band names (`alpha_*`, `beta_*`), only FFT bins (`freq_010_0` … `freq_750_3`). We decoded the axis post hoc (freq_XXX/100 ≈ Hz, via the `jordan-bird/eeg-feature-generation` repository) — theta = `freq_040–080`, alpha = `freq_080–130`, beta = `freq_130–300` — but validating that mapping, implementing BATR, and calibrating its scale within the time-box was optimistic. It remains documented future work.
- **LOSO (leave-one-subject-out) was not run.** The aggregated Kaggle CSV **has no subject IDs** — it would require downloading Bird's original repo and rebuilding the pipeline from the raw files. That was out of scope. Direct consequence: the 0.97 figure is honest for "new windows from the same 4 subjects", **not** for "a new user never seen before".
- **Raw-signal → feature extraction is not in the app.** The Streamlit app accepts CSVs already in the 988-feature format. Anyone wanting to test with raw EEG would first need to run Bird's pipeline. Documented as the platform's #1 limitation.
- **Sample too small for generalization claims.** 4 subjects is too few for any "works on new people" claim. Gender balance helps but does not replace a larger N. Our score is *real* in what it measures, but what it measures is narrower than the phrase "concentration detector" suggests.

## 3. What I would do differently

- **Start by decoding the frequency axis, not with traditional EDA.** Knowing that `freq_XXX ≈ Hz/100` from hour 1 unlocks BATR and unlocks interpreting the top features as "right posterior alpha at 10.1 Hz" instead of "freq_101_2". It would have been the first 20-minute investment to make a difference.
- **Validate scaler saturation on 3 specific features, not a global histogram.** The question "is RobustScaler different?" has a *per-feature* answer. Global stats mislead — raw `covM_1_1` has a range of 530k while 95% of features have a range <100. Aggregation hid the heterogeneity. It would have saved an iteration.
- **Run LOSO even with "only 4 folds"** by downloading the per-subject CSVs from Bird's repo before starting. Four LOSO folds would be noisy but would show whether the drop is 2 points or 20 — a huge difference for the framing of this REPORT.
- **Decouple `engagement_score` from LogReg from the start instead of bolting it on later.** The score ended up depending on LogReg's calibration; had I exposed it as an interface from the start (`score(probs) -> float`), the probability source could be swapped without refactoring.
- **Family-weighted feature importance instead of per-feature SHAP.** With 988 features, individual top-20 lists are opaque. Aggregating by family (`freq`, `covM`, `eigenval`, etc.) turned 988 numbers into 13 — enough for a conversation with a non-ML audience.

## 4. Plan vs. Outcome

| PLAN item | Done? | Notes |
|---|---|---|
| EDA: shape, duplicates, outliers, classes | Yes | `notebooks/01_eda.ipynb` with an empirical scaler comparison |
| ANOVA ranking | Yes | top-20 by F-stat; lag1_logcovM_2_2 at the top with F=1104 |
| De-dup before the split | Yes | 115 duplicates removed, stratified 80/20 split |
| Scaler (RobustScaler) | Yes | + clipping at ±10 after empirical verification |
| LogReg L2 multinomial, 5-fold CV | Yes | 0.9534 ± 0.0068 (scaler inside the Pipeline) |
| XGBoost default + same CV | Yes | 0.9690 ± 0.0070 (scaler inside the Pipeline) |
| Confusion matrices + classification_report | Yes | `notebooks/figures/confusion_{logreg,xgb}.png` |
| BATR (beta/(alpha+theta)) | **No** | CSV has no band names; axis decoded but BATR left as future work |
| Engagement score 0–100 | Yes | Option B: `50*(P(concentrating) - P(relaxed) + 1)` via LogReg |
| Score validation (scatter, boxplot, correlation) | Yes | Spearman 0.9269, 3 figures in `notebooks/figures/` |
| Streamlit accepting CSV | Yes | validates against `artifacts/columns.json`, button to generate/download a sample |
| Streamlit showing class + score per row | Yes | + LogReg–XGB agreement, mean score, predominant class |
| LOSO | **No** | Kaggle CSV has no subject IDs; documented as a limitation |
| REPORT.md | Yes | this file |
| README.md | Yes | see `README.md` |

## 5. Limitations — what was NOT proven

1. **Cross-subject generalization.** 0.97 holds for "a new window from the same 4 subjects". Adjacent windows from the same session are near-identical; a random split lets the model memorize session signatures.
2. **Cross-session generalization.** Not even a within-subject, cross-session protocol was tested.
3. **Generalization to raw EEG.** The app takes features; Bird's "raw signal → 988 features" bridge was not repackaged.
4. **Robustness of the engagement score.** Validated by monotonicity and ordinal correlation; not validated against an external psychometric measure (NASA-TLX, time-on-task, etc.).
5. **Calibration of XGBoost probabilities.** We did not apply `CalibratedClassifierCV`; that is why the score uses LogReg, whose probabilities are naturally well calibrated.

## 6. Artifacts produced

- `artifacts/logreg.pkl`, `artifacts/xgb.pkl`, `artifacts/scaler.pkl`
- `artifacts/columns.json` (988 columns in training order — the app validates against it)
- `artifacts/sample_input.csv` (5 test rows, for a quick app test)
- `artifacts/evaluation.json` (confusion matrices + per-family importance)
- `artifacts/run_log.jsonl` (58 entries: findings, decisions, doubts, metrics)
- `notebooks/01_eda.ipynb`
- `notebooks/figures/` — confusion matrices, family importance, engagement boxplot/scatter/histogram

## 7. How to reproduce

See `README.md`. The short path:

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
PYTHONPATH=. python scripts/build_eda_notebook.py    # EDA
PYTHONPATH=. python scripts/train.py                 # scaler + LogReg + XGB + columns.json
PYTHONPATH=. python scripts/evaluate.py              # test-set metrics + family importance
streamlit run app/streamlit_app.py
```
