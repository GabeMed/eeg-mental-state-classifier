# PLAN.md — EEG Mental State Classifier

**Author:** Gabriel Medeiros
**Date:** 2026-04-19
**Self-imposed time-box:** ~4 effective hours

---

## 1. Reading the problem

The goal is to build a system that classifies mental states (relaxed, neutral, concentrating) from EEG signals, and to derive a continuous engagement score. The `birdy654/eeg-brainwave-dataset-mental-state` dataset contains 2479 time windows already converted into 988 statistical features (time domain, spectral, inter-channel covariance), recorded with a Muse headband (4 electrodes: TP9, AF7, AF8, TP10) at 200 Hz.

The classes are balanced (33% each), so the problem is not imbalance — it is **high-dimensional generalization with few samples** (≈2.5 rows per feature) and **the noise characteristic of EEG**. The real work lies in (1) validating honestly, (2) justifying feature and model choices, and (3) delivering an engagement score consistent with the neuroscience of the problem.

---

## 2. Scope decisions

- **Dataset:** the preprocessed Kaggle CSV (988 features + Label), as provided.
- **Cleaning:** remove the 115 duplicate rows (4.6%) before any split, so metrics are not inflated.
- **Scaling:** `RobustScaler` (median + IQR) instead of `StandardScaler`. The data has severe outliers — a maximum absolute value of 530656 vs. a p99 of 311, a four-orders-of-magnitude gap that would break mean/std normalization.
- **Validation:** `StratifiedKFold` with 5 folds. Primary metric: macro-F1.
- **Models:** two, chosen to be **different in kind**, not variations of the same thing:
  - **Logistic Regression** (L2) — a fast, interpretable linear baseline. It anchors expectations about how much signal is linearly capturable.
  - **XGBoost** — state of the art for tabular data in this regime (see Grinsztajn et al., 2022, "Why do tree-based models still outperform deep learning on tabular data").

  The comparison tells us how much of the problem's structure is linear vs. non-linear. Two models from the same family (e.g., RF + XGB) would give a less informative comparison.
- **Engagement score:** the classic BATR formula (β / (α + θ)) from Pope, Bogart & Bartolome (1995), computed over the dataset's frequency features, normalized to a 0–100 scale, and validated via correlation with the `concentrating` class.

---

## 3. Steps and time budget

Execution compressed into ~4 effective hours (self-imposed time-box). Each step has a ceiling; if it overruns, I cut scope and move on.

| Order | Step | Time | Output |
|-------|------|------|--------|
| 1 | PLAN.md + repo setup | 20 min | This document + git structure |
| 2 | EDA (notebook) | 45 min | 5–7 targeted plots: class distribution, feature correlations, per-state statistics |
| 3 | Feature pipeline | 30 min | De-dup → `RobustScaler` → derived features (band ratios) |
| 4 | Models + comparison | 45 min | LogReg and XGBoost with stratified CV, confusion matrices, overfitting discussion |
| 5 | Engagement score | 30 min | BATR + validation against ground truth + illustrative curve |
| 6 | Streamlit app | 45 min | CSV upload → state prediction + engagement score |
| 7 | Final report + README | 25 min | Honest markdown on what worked, what failed, and what I would do differently |
| — | Buffer / polish / git | 20 min | Final commit, push, sanity check |

**Total planned:** ~4 effective hours.

---

## 4. Success criteria

Below is what I consider a defensible result — not the ideal, but the minimum that justifies the time invested.

- **Modeling:** both models beat chance (0.33 macro-F1) by a clear margin under stratified cross-validation. The gap between LogReg and XGBoost is reported honestly, with an interpretation of what it reveals about the problem.
- **Feature engineering:** every feature family used or derived is justified in one sentence — not "I added X just because".
- **Engagement score:** correlates positively with the `concentrating` class on the validation set. I do not expect r > 0.8; I expect a statistically non-trivial correlation and an interpretable curve.
- **App:** runs end-to-end on a test CSV. It does not need to be pretty; it needs to work.
- **Methodological honesty:** known limitations are documented in the final report. Optimistic numbers are flagged as such.

---

## 5. Known limitations

I record here what I know will limit the results, so it does not look like I discovered them afterwards:

- **Pre-extracted features, no raw signal.** I do not have access to the original time series, so I cannot apply EEG-specific architectures (EEGNet, 1D CNNs). Documented as future work.
- **The CSV has no subject IDs.** Cross-validation is stratified over windows, not subjects. That is enough for this project's scope, but a real deployment on a new user would require a subject-independent protocol (LOSO) — I mention this in the final report as a next step.
- **Consumer-grade hardware (Muse, dry electrodes).** Lower signal-to-noise ratio than clinical EEG. The accuracy ceiling is intrinsically limited.
- **~4-hour time-box.** There will be no exhaustive hyperparameter tuning or extensive ablation. Sensible defaults + one round of regularization.

---

## 6. Contingency plan

If something stalls and I fall behind budget:

- **Cut first:** Streamlit polish (keep it functional, not pretty).
- **Cut second:** extra derived features (keep only de-dup + RobustScaler).
- **Never cut:** the two models, cross-validation, the honest final report, and the README that lets anyone run the project.

If something goes wrong with XGBoost, fall back to Random Forest as a substitute — same family, similar behavior, fewer hyperparameters.

---

**End of plan. Starting execution now.**
