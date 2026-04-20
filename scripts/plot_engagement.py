import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings("ignore")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from src.data import load_dedup_split
from src.features import apply_scaler, load_scaler
from src.engagement import engagement_score

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "notebooks" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0.0: "relaxed", 1.0: "neutral", 2.0: "concentrating"}

Xtr, Xte, ytr, yte, _ = load_dedup_split()
scaler = load_scaler()
Xte_s = apply_scaler(scaler, Xte)
lr = joblib.load(ROOT / "artifacts" / "logreg.pkl")

scores = engagement_score(lr, Xte_s)
probs = lr.predict_proba(Xte_s)
classes = list(lr.classes_)
p_concentrating = probs[:, classes.index(2.0)]
true_names = [CLASS_NAMES[v] for v in yte.values]

plot_df = pd.DataFrame(
    {
        "engagement_score": scores,
        "p_concentrating": p_concentrating,
        "true_class": true_names,
    }
)

class_order = ["relaxed", "neutral", "concentrating"]
palette = {
    "relaxed": "#4C78A8",
    "neutral": "#F58518",
    "concentrating": "#54A24B",
}

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=plot_df,
    x="true_class",
    y="engagement_score",
    order=class_order,
    palette=palette,
    showmeans=True,
    meanprops={
        "marker": "^",
        "markerfacecolor": "green",
        "markeredgecolor": "green",
        "markersize": 8,
    },
)
plt.xlabel("Classe verdadeira")
plt.ylabel("Score de engajamento")
plt.title("Score de engajamento por estado (conjunto de teste)")
plt.tight_layout()
plt.savefig(FIGURES / "engagement_boxplot.png", dpi=130)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=plot_df,
    x="engagement_score",
    y="p_concentrating",
    hue="true_class",
    hue_order=class_order,
    palette=palette,
    alpha=0.75,
    s=45,
)
plt.xlabel("Score de engajamento")
plt.ylabel("P(concentrating)")
plt.title("Score vs P(concentrating) — colorido por classe verdadeira")
plt.tight_layout()
plt.savefig(FIGURES / "engagement_scatter.png", dpi=130)
plt.close()

plt.figure(figsize=(8, 5))
for name in class_order:
    subset = plot_df.loc[plot_df["true_class"] == name, "engagement_score"]
    plt.hist(subset, bins=20, alpha=0.45, label=name, color=palette[name])
plt.xlabel("Score de engajamento")
plt.ylabel("Frequência")
plt.title("Distribuição do score por classe")
plt.legend(title="Classe")
plt.tight_layout()
plt.savefig(FIGURES / "engagement_histogram.png", dpi=130)
plt.close()
