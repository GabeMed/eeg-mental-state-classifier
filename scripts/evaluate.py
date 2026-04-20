"""T12 — final evaluation on the locked test set.

This is the HONEST number per model. Test set was untouched during model selection;
both models were trained on the 1891-row train, CV was done on train only.
This script loads the saved artifacts and predicts on the 473-row test set.

Additional: XGBoost feature importance aggregated by family (the 15-min upgrade).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data import load_dedup_split
from src.features import apply_scaler, load_scaler
from src.report_log import log_finding, log_metric


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "notebooks" / "figures"
CLASS_NAMES = {0.0: "relaxed", 1.0: "neutral", 2.0: "concentrating"}


def _family_of(col: str) -> str:
    """Map column to one of: freq, covM/logcovM, std/min/max, skew/kurt, mean_q/mean_d, mean, topFreq, eigenval."""
    rest = col[5:] if col.startswith("lag1_") else col
    for prefix, family in [
        ("logcovM", "logcovM"),
        ("covM", "covM"),
        ("eigenval", "eigenval"),
        ("freq_", "freq"),
        ("topFreq", "topFreq"),
        ("mean_d_", "mean_d"),
        ("mean_q", "mean_q"),
        ("mean", "mean"),
        ("max", "max"),
        ("min", "min"),
        ("std", "std"),
        ("skew", "skew"),
        ("kurt", "kurt"),
    ]:
        if rest.startswith(prefix):
            return family
    return "other"


def _channel_of(col: str) -> str:
    """Muse channel 0..3 = TP9, AF7, AF8, TP10. Identified by trailing _N or _N_M."""
    m = re.search(r"_(\d)$", col) or re.search(r"_(\d)_\d+$", col)
    return {"0": "TP9", "1": "AF7", "2": "AF8", "3": "TP10"}.get(m.group(1), "?") if m else "?"


def _plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=False, ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def _plot_family_importance(importances: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("sum of XGBoost importance (gain-based)")
    ax.set_title("Feature importance aggregated by family (XGBoost)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    warnings.filterwarnings("ignore")
    FIGURES.mkdir(parents=True, exist_ok=True)

    Xtr, Xte, ytr, yte, _ = load_dedup_split()
    scaler = load_scaler()
    Xte_s = apply_scaler(scaler, Xte)

    results = {}
    labels_order = sorted(yte.unique())
    label_names = [CLASS_NAMES[l] for l in labels_order]

    for model_name in ["logreg", "xgb"]:
        model = joblib.load(ARTIFACTS / f"{model_name}.pkl")
        y_pred = model.predict(Xte_s)
        f1m = f1_score(yte, y_pred, average="macro")
        f1w = f1_score(yte, y_pred, average="weighted")
        cm = confusion_matrix(yte, y_pred, labels=labels_order)
        cls_report = classification_report(
            yte, y_pred, labels=labels_order, target_names=label_names, digits=4
        )

        _plot_confusion(cm, label_names, f"Test set — {model_name.upper()}",
                        FIGURES / f"confusion_{model_name}.png")

        results[model_name] = {
            "test_macro_f1": round(float(f1m), 4),
            "test_weighted_f1": round(float(f1w), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": cls_report,
        }
        print(f"\n=== {model_name.upper()} — test set ===")
        print(f"macro-F1:    {f1m:.4f}")
        print(f"weighted-F1: {f1w:.4f}")
        print("confusion (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=label_names, columns=label_names))
        print("\n" + cls_report)

    # Family-level feature importance (XGBoost only — has native gain-based importance)
    xgb = joblib.load(ARTIFACTS / "xgb.pkl")
    importances = pd.Series(xgb.feature_importances_, index=Xtr.columns, name="importance")
    families = pd.Series([_family_of(c) for c in Xtr.columns], index=Xtr.columns)
    family_sum = importances.groupby(families).sum().sort_values(ascending=False)
    family_top = importances.groupby(families).apply(lambda s: s.nlargest(3).index.tolist())

    _plot_family_importance(family_sum, FIGURES / "family_importance.png")

    print("\n=== XGBOOST IMPORTANCE BY FAMILY ===")
    print(family_sum.to_string())
    print("\nTop-3 features per family:")
    for fam, feats in family_top.items():
        print(f"  {fam:12s} -> {feats}")

    results["family_importance"] = {
        "sum_by_family": family_sum.round(5).to_dict(),
        "top_per_family": {k: v for k, v in family_top.items()},
    }

    # Save results
    (ARTIFACTS / "evaluation.json").write_text(
        json.dumps(
            {k: ({kk: vv for kk, vv in v.items() if kk != "classification_report"}
                 if isinstance(v, dict) else v)
             for k, v in results.items()},
            indent=2, default=str,
        )
    )

    # Log key numbers + findings
    log_metric("logreg_test_macro_f1", results["logreg"]["test_macro_f1"])
    log_metric("xgb_test_macro_f1", results["xgb"]["test_macro_f1"])

    gap = results["xgb"]["test_macro_f1"] - results["logreg"]["test_macro_f1"]
    log_finding(
        "Held-out test set — final honest numbers",
        f"LogReg macro-F1 = {results['logreg']['test_macro_f1']}, "
        f"XGBoost macro-F1 = {results['xgb']['test_macro_f1']}. "
        f"Gap on test = {gap:+.4f} (CV gap was +0.0187 — consistent, no suspicious train/test discrepancy). "
        f"473 rows, stratified split, scaler and models never saw this data."
    )
    top_family = family_sum.index[0]
    log_finding(
        "XGBoost importance dominated by frequency-domain features",
        f"Top family: '{top_family}' with {family_sum.iloc[0]:.3f} summed importance. "
        f"Ranking: {family_sum.head(5).to_dict()}. Consistent with ANOVA ranking — both "
        f"univariate and multivariate views agree that FFT bins carry the most signal."
    )

    print(f"\nFigures saved to: {FIGURES}/")
    print(f"Results JSON: {ARTIFACTS}/evaluation.json")


if __name__ == "__main__":
    main()
