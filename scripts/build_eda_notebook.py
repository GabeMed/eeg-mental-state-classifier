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

**Pergunta que queremos responder antes de modelar:**
O dataset tem estrutura suficiente para separar `relaxed`, `neutral` e `concentrating`? Quais features carregam mais sinal? Como devemos escalar os dados?

**Saídas deste notebook são usadas para justificar:**
- Escolha do scaler (RobustScaler vs StandardScaler)
- De-duplicação antes do split
- Expectativa realista de desempenho do modelo baseline
"""
        ),
        code(
            """
import sys, os
# Permite importar de src/ quando o notebook é executado na raiz do repo
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
## 1. Sanity check — o que temos nas mãos

Antes de qualquer plot, conferir o básico: forma, classes, nulos, duplicatas. Se algo aqui estiver errado, todo o resto desmorona.
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
**Leitura:**
- 2479 linhas × 989 colunas → **~2.5 linhas por feature**. Regime de alta dimensão + amostras pequenas. Isto define todo o risco de overfitting adiante.
- Classes balanceadas (33% cada) → não vamos precisar de class weighting ou resampling.
- **115 linhas duplicadas (4.6%)** precisam ser removidas antes do split. Se deixamos, corremos o risco de a mesma janela aparecer em treino e teste, inflando a métrica.
"""
        ),
        md(
            """
## 2. Distribuição de classes
"""
        ),
        code(
            """
fig, ax = plt.subplots(figsize=(7, 4))
counts = df[LABEL_COL].value_counts().sort_index()
ax.bar([class_name(c) for c in counts.index], counts.values, color=['#4C72B0', '#55A868', '#C44E52'])
ax.set_title('Distribuição de janelas por estado mental')
ax.set_ylabel('Número de janelas')
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
## 3. Escolha de scaler — o que os números realmente dizem

A intuição inicial: outliers extremos → StandardScaler compromete a escala → usar RobustScaler. Mas essa justificativa é preguiçosa se não for medida. Vamos fazer a comparação empírica.
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
**Visão agregada:** máximo absoluto é 1704× o p99. A métrica grita "StandardScaler falha". Mas isso mistura as 988 features num único número. Per-feature, a história é mais sutil.
"""
        ),
        code(
            """
from sklearn.preprocessing import StandardScaler, RobustScaler

feats = df.drop(columns=['Label'])

# Razão std/IQR por feature. Ratio ~1 = scalers equivalentes; ratio alto = outliers dominam std.
ratios = (feats.std() / (feats.quantile(0.75) - feats.quantile(0.25)).replace(0, np.nan)).dropna()
print('Distribuição do ratio std/IQR entre as 988 features:')
print(f'  mediana : {ratios.median():.2f}  ← metade das features: scalers equivalentes')
print(f'  p90     : {ratios.quantile(0.90):.2f}')
print(f'  p99     : {ratios.quantile(0.99):.2f}')
print(f'  máximo  : {ratios.max():.2f}  ← covariâncias da matriz de canais')

# Top-5 features onde os scalers mais divergem
print('\\nTop-5 features com maior std/IQR (onde a escolha importa):')
for feat in ratios.sort_values(ascending=False).head(5).index:
    print(f'  {feat:25s}  std/IQR = {ratios[feat]:>7.1f}')
"""
        ),
        code(
            """
# Comparação direta: o que cada scaler produz para covM_1_1 (pior caso)
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
**Leitura final — decisão:**

Para **metade das features** (std/IQR ≈ 0.92) os dois scalers produzem valores quase idênticos. A discussão só importa para ~10% das features — entradas da matriz de covariância (`covM_*`) com ratios 30–248×.

Nesse grupo minoritário:
- **StandardScaler** produz saída bounded (máx ~25 para `covM_1_1`), mas comprime a variação normal porque o std é puxado pelos outliers.
- **RobustScaler** preserva a separação entre valores típicos, mas deixa outliers com magnitude extrema (máx >6000 para `covM_1_1`).

**Decisão adotada:** `RobustScaler` seguido de clip para ±10. Combina o melhor dos dois — mediana/IQR ignora outliers no cálculo da escala, e o clip pós-transformação evita que valores residuais estourem em regressão logística. Para XGBoost é irrelevante (modelo é invariante a escala), mas mantemos o mesmo pipeline por consistência.
"""
        ),
        md(
            """
## 4. ANOVA — quais features discriminam entre estados?

F-stat alto ≡ média dos grupos difere relativo à variância interna ≡ a feature "vê" a classe. É um ranking univariado, rápido, útil para diagnóstico — **não** substitui a importância multivariada que o modelo vai calcular.
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
# Boxplot das 6 features mais discriminativas, por classe
top6 = anova_top.head(6)['feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.ravel(), top6):
    data_by_class = [df.loc[df[LABEL_COL] == c, feat].values for c in sorted(df[LABEL_COL].unique())]
    ax.boxplot(data_by_class, labels=[class_name(c) for c in sorted(df[LABEL_COL].unique())], showfliers=False)
    ax.set_title(feat, fontsize=10)
plt.suptitle('Top-6 features por F-stat — distribuição por classe (outliers ocultos)', y=1.02)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
**Leitura:** medianas visualmente separáveis → existe sinal capturável por um modelo linear. Sem separação visível → teríamos um alerta vermelho aqui.
"""
        ),
        md(
            """
## 5. Correlação entre features — risco de redundância

988 features é muito. Boa parte será redundante. A regressão logística L2 lida com multicolinearidade ok, mas XGBoost se beneficia mais de features independentes. Visualizar o padrão antes de modelar.
"""
        ),
        code(
            """
top50 = top_variance_features(df, top_n=50)
corr = df[top50].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Pearson r'}, ax=ax)
ax.set_title('Correlação entre top-50 features por variância')
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
**Leitura:** blocos visíveis na diagonal indicam grupos de features correlacionadas — típico quando features são lags/estatísticas computadas sobre o mesmo canal. Nota: não removeremos features correlacionadas manualmente — deixamos L2 e XGBoost lidarem com isso. Fazer feature selection agressiva com tão poucas amostras adiciona risco de overfitting no processo de seleção.
"""
        ),
        md(
            """
## 6. Conclusões que alimentam a modelagem

| Achado | Consequência |
|--------|--------------|
| Outliers 4+ ordens de magnitude além do p99 | RobustScaler em vez de StandardScaler |
| 115 duplicatas (4.6%) | De-dup antes do split estratificado |
| Classes balanceadas (33/33/33) | Sem class weighting; macro-F1 como métrica principal |
| Features top-ANOVA mostram separação visível | Existe sinal linear capturável — LogReg tem chance real |
| Correlação evidente entre top-50 | Deixar regularização L2 e XGBoost lidarem; não fazer seleção manual |
| 988 features × 2479 linhas | Alto risco de overfitting; CV estratificada é obrigatória |
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
