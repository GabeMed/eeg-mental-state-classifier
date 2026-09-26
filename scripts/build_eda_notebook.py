"""Build and execute notebooks/01_eda.ipynb programmatically.

Why: keeping the notebook content in a Python script (vs. hand-editing .ipynb JSON)
makes diffs readable and reproducible. Run `python scripts/build_eda_notebook.py`
to regenerate the notebook with fresh outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_eda.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
# EEG Mental State — EDA (01)

**Question we want to answer before modeling:**
Does the dataset have enough structure to separate `relaxed`, `neutral`, and `concentrating`? Which features carry the most signal? How should we scale the data?

**Outputs of this notebook are used to justify:**
- The scaler choice (RobustScaler vs StandardScaler)
- De-duplication before the split
- A realistic performance expectation for the baseline model
"""
        ),
        code(
            """
import sys, os
# Allows importing from src/ when the notebook runs from the repo root
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.eda import (
    load_raw, describe_dataset, outlier_profile,
    anova_feature_ranking, top_variance_features, class_name, LABEL_COL,
)

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_style('whitegrid')

df = load_raw(os.path.join(ROOT, 'data', 'mental-state.csv'))
print(f"loaded: {df.shape}")
"""
        ),
        md(
            """
## 1. Sanity check — what we have in hand

Before any plot, check the basics: shape, classes, nulls, duplicates. If anything here is wrong, everything else falls apart.
"""
        ),
        code(
            """
info = describe_dataset(df)
for k, v in info.items():
    print(f"{k}: {v}")
"""
        ),
        md(
            """
**Reading:**
- 2479 rows × 989 columns → **~2.5 rows per feature**. High-dimensional, small-sample regime. This defines all the overfitting risk ahead.
- Balanced classes (33% each) → no need for class weighting or resampling.
- **115 duplicate rows (4.6%)** must be removed before the split. If we keep them, the same window may appear in both train and test, inflating the metric.
"""
        ),
        md(
            """
## 2. Class distribution
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(7, 4))
counts = df[LABEL_COL].value_counts().sort_index()
ax.bar([class_name(c) for c in counts.index], counts.values, color=['#4C72B0', '#55A868', '#C44E52'])
ax.set_title('Window distribution by mental state')
ax.set_ylabel('Number of windows')
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 3. Scaler choice — what the numbers actually say

The initial intuition: extreme outliers → StandardScaler distorts the scale → use RobustScaler. But that justification is lazy unless it is measured. Let's run the empirical comparison.
"""
        ),
        code(
            """
outliers = outlier_profile(df)
for k, v in outliers.items():
    if isinstance(v, float):
        print(f"{k}: {v:,.2f}")
    else:
        print(f"{k}: {v}")
"""
        ),
        md(
            """
**Aggregate view:** the absolute maximum is 1704× the p99. The metric screams "StandardScaler fails". But this collapses all 988 features into a single number. Per feature, the story is more subtle.
"""
        ),
        code(
            """
from sklearn.preprocessing import StandardScaler, RobustScaler

feats = df.drop(columns=['Label'])

# std/IQR ratio per feature. Ratio ~1 = equivalent scalers; high ratio = outliers dominate std.
ratios = (feats.std() / (feats.quantile(0.75) - feats.quantile(0.25)).replace(0, np.nan)).dropna()
print('Distribution of the std/IQR ratio across the 988 features:')
print(f'  median  : {ratios.median():.2f}  ← half of the features: equivalent scalers')
print(f'  p90     : {ratios.quantile(0.90):.2f}')
print(f'  p99     : {ratios.quantile(0.99):.2f}')
print(f'  max     : {ratios.max():.2f}  ← channel covariance-matrix entries')

# Top-5 features where the scalers diverge most
print('\\nTop-5 features with the highest std/IQR (where the choice matters):')
for feat in ratios.sort_values(ascending=False).head(5).index:
    print(f'  {feat:25s}  std/IQR = {ratios[feat]:>7.1f}')
"""
        ),
        code(
            """
# Direct comparison: what each scaler produces for covM_1_1 (worst case)
worst = ratios.idxmax()
raw = feats[worst]
ss = (raw - raw.mean()) / raw.std()
rs = (raw - raw.median()) / (raw.quantile(0.75) - raw.quantile(0.25))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(raw, bins=60, color='#666'); axes[0].set_title(f'raw — {worst}\\nmax={raw.max():,.0f}')
axes[1].hist(ss, bins=60, color='#4C72B0'); axes[1].set_title(f'StandardScaler\\nmax={ss.abs().max():.1f}, p99={np.percentile(ss.abs(),99):.2f}')
axes[2].hist(rs, bins=60, color='#C44E52'); axes[2].set_title(f'RobustScaler\\nmax={rs.abs().max():,.0f}, p99={np.percentile(rs.abs(),99):.1f}')
for ax in axes: ax.set_yscale('log')
plt.tight_layout(); plt.show()
"""
        ),
        md(
            """
**Final reading — decision:**

For **half of the features** (std/IQR ≈ 0.92) both scalers produce nearly identical values. The debate only matters for ~10% of the features — covariance-matrix entries (`covM_*`) with ratios of 30–248×.

Within that minority group:
- **StandardScaler** produces bounded output (max ~25 for `covM_1_1`), but compresses normal variation because the std is pulled up by the outliers.
- **RobustScaler** preserves the separation between typical values, but leaves outliers at extreme magnitude (max >6000 for `covM_1_1`).

**Decision adopted:** `RobustScaler` followed by clipping to ±10. It combines the best of both — median/IQR ignores outliers when computing the scale, and the post-transform clip keeps residual extreme values from blowing up logistic regression. For XGBoost it is irrelevant (the model is scale-invariant), but we keep the same pipeline for consistency.
"""
        ),
        md(
            """
## 4. ANOVA — which features discriminate between states?

High F-stat ≡ group means differ relative to within-group variance ≡ the feature "sees" the class. It is a fast univariate ranking, useful for diagnosis — it does **not** replace the multivariate importance the model will compute.
"""
        ),
        code(
            """
anova_top = anova_feature_ranking(df, top_n=20)
anova_top
"""
        ),
        code(
            """
# Boxplot of the 6 most discriminative features, by class
top6 = anova_top.head(6)['feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.ravel(), top6):
    data_by_class = [df.loc[df[LABEL_COL] == c, feat].values for c in sorted(df[LABEL_COL].unique())]
    ax.boxplot(data_by_class, labels=[class_name(c) for c in sorted(df[LABEL_COL].unique())], showfliers=False)
    ax.set_title(feat, fontsize=10)
plt.suptitle('Top-6 features by F-stat — distribution by class (outliers hidden)', y=1.02)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
**Reading:** visually separable medians → there is signal a linear model can capture. With no visible separation, this would be a red flag.
"""
        ),
        md(
            """
## 5. Feature correlation — redundancy risk

988 features is a lot. Many will be redundant. L2 logistic regression handles multicollinearity fine, but XGBoost benefits more from independent features. Visualize the pattern before modeling.
"""
        ),
        code(
            """
top50 = top_variance_features(df, top_n=50)
corr = df[top50].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Pearson r'}, ax=ax)
ax.set_title('Correlation among the top-50 features by variance')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
**Reading:** visible blocks along the diagonal indicate groups of correlated features — typical when features are lags/statistics computed over the same channel. Note: we will not remove correlated features manually — we let L2 and XGBoost handle it. Aggressive feature selection with so few samples adds overfitting risk to the selection process itself.
"""
        ),
        md(
            """
## 6. Conclusions that feed the modeling

| Finding | Consequence |
|--------|--------------|
| Outliers 4+ orders of magnitude beyond the p99 | RobustScaler instead of StandardScaler |
| 115 duplicates (4.6%) | De-dup before the stratified split |
| Balanced classes (33/33/33) | No class weighting; macro-F1 as the primary metric |
| Top-ANOVA features show visible separation | Capturable linear signal exists — LogReg has a real chance |
| Clear correlation among the top-50 | Let L2 regularization and XGBoost handle it; no manual selection |
| 988 features × 2479 rows | High overfitting risk; stratified CV is mandatory |
"""
        ),
    ]
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    return nb


def main():
    nb = build_notebook()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()
    nbf.write(nb, OUT)
    print(f"wrote and executed: {OUT}")


if __name__ == "__main__":
    main()
