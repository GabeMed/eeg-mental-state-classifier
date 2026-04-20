"""Seed the run log with everything we discovered/decided BEFORE the log existed.

Run once, then delete or keep for reproducibility.
"""

from src.report_log import log_finding, log_decision, log_doubt, log_metric


# ---- FINDINGS from T1 (dataset inspection) ----
log_finding(
    "Dataset shape confirms PLANO assumptions",
    "2479 rows × 989 cols (988 features + Label). Classes balanced at 33%/33.5%/33.5%. No nulls. All float64.",
)
log_finding(
    "Exact duplicate rate: 4.64% (115 rows)",
    "Matches PLANO's expectation. De-dup before split is mandatory — else same window appears in train and test.",
)
log_finding(
    "Outlier severity confirms RobustScaler need",
    "Global max |value| = 530,657 vs p99 = 311. Ratio 1704x. StandardScaler std would be blown up by outliers.",
)
log_finding(
    "No subject IDs in aggregated CSV",
    "No columns matching subject/participant/id/user/session. LOSO validation not feasible from this CSV alone. Logged as known limitation.",
)
log_finding(
    "Frequency axis decoded (external research)",
    "freq_XXX / 100 ≈ Hz, because Bird resampled Muse data to 150Hz with 148-sample FFT window. 50Hz notch-filter gap (UK AC line noise). Band mapping: theta=freq_040-080, alpha=freq_080-130, beta=freq_130-300.",
)

# ---- FINDINGS from T4-T6 (EDA) ----
log_finding(
    "Right hemisphere dominates top-20 ANOVA features",
    "AF8 (right frontal): 7 features. TP10 (right temporal): 7. TP9 (left temporal): 6. AF7 (left frontal): ZERO. Consistent with Posner & Petersen (1990) right-hemisphere dominance for sustained attention.",
)
log_finding(
    "Discriminative frequencies cluster at 10-13 Hz (alpha boundary)",
    "Top frequency features: freq_101, freq_111, freq_122, freq_132 (all channels). This is the alpha-band boundary — classical Berger (1929) alpha-blocking effect.",
)
log_finding(
    "ANOVA top-10 is really 4 distinct signals, doubled",
    "Each raw_* feature has a near-identical lag1_* counterpart. Effective informative set is smaller than naive count suggests. Matters for interpretation, not for modeling (L2 handles it).",
)
log_finding(
    "Feature families are lenses, not metadata",
    "988 features decompose into families: freq_ (576 total, spectral lens), statistics (std/min/max/skew/kurt — distribution-shape lens), covM/eigenval (spatial-coordination lens), lag1_ (temporal-change lens). All are predictors.",
)

# ---- DECISIONS ----
log_decision(
    "Engagement score: Option B (model-based proxy)",
    "score = 100 * P(concentrating) from LogReg. Reason: frequency-axis units were not documented in the CSV; BATR would require assumptions we couldn't verify inside the time-box. Documented in DESIGN.md §D2.",
)
log_decision(
    "Scaler: RobustScaler + clip to ±10",
    "After user-prompted empirical check: for ~10% of features (covariance matrix entries, std/IQR ratios 30-248x) the scalers differ meaningfully. RobustScaler preserves typical-value separation; clip bounds the residual outliers.",
)
log_decision(
    "Feature families treated as equal in model input, stratified in analysis",
    "No family-aware regularization (Group Lasso) or hand-weighting. With 2479 rows and 988 features, injecting family priors risks bias without generalization gain. Instead: report XGBoost feature importance aggregated by family.",
)
log_decision(
    "Validation protocol: stratified 80/20 holdout + 5-fold CV on train",
    "CV on train for model selection/stability. Holdout test is locked until final eval — single unbiased number per model in the report.",
)

# ---- DOUBTS (with resolutions) ----
log_doubt(
    "Is RobustScaler really different from StandardScaler when median≈0 and IQR is small?",
    resolution=(
        "Measured empirically. Half of features: std/IQR ≈ 1 (scalers ~identical). For 10% of features "
        "(covariance matrix entries): std/IQR is 30-248x — scalers diverge massively. Example: covM_1_1 "
        "raw=369k → SS=25.3, RS=6290. Switched to RobustScaler + post-clip to ±10 for bounded inputs."
    ),
)
log_doubt(
    "Should we use BATR (Pope 1995) or a model-based engagement score?",
    resolution=(
        "BATR requires knowing which FFT columns map to alpha/beta/theta. The CSV's column names (freq_XXX) "
        "didn't label units. Used Option B (model-based proxy). Decoded the axis mapping post-hoc for future work."
    ),
)
log_doubt(
    "Are we treating all 988 features as equal? Is that correct given their semantic families?",
    resolution=(
        "In input: yes (uniform L2, no family priors). Correct because (a) no justified family-level prior, "
        "(b) L2 + XGBoost handle intra-family redundancy gracefully, (c) hand-crafted priors risk bias with "
        "small sample. In analysis: no — will report importance by family in T12."
    ),
)
log_doubt(
    "Are lag1_* and non-frequency features just metadata, not real predictors?",
    resolution=(
        "No — every column is a predictor. lag1_X = same statistic on the previous window. std/min/max/covM "
        "are legitimate statistical lenses on the signal. ANOVA happens to favor freq_* because they carry "
        "cleaner univariate linear separation, but multivariate models use the rest too."
    ),
)

# ---- METRICS so far ----
log_metric("rows", 2479)
log_metric("features", 988)
log_metric("duplicate_rate_pct", 4.64)
log_metric("outlier_ratio_max_to_p99", 1704.09)
log_metric("top_anova_f_stat", 1104.66, note="lag1_logcovM_2_2 — channel AF8 covariance")

print("log seeded.")
