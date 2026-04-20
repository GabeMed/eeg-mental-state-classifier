# DESIGN.md — Decisões técnicas

Documento curto para resolver as 4 zonas cinzentas antes de codar.

---

## D1. Schema do Streamlit (resolve o #1 da crítica)

**Decisão:** app aceita CSV no mesmo formato do dataset de treino (988 colunas de features + coluna `Label` opcional).

**Razão:** dentro do time-box não dá para empacotar extração de sinal bruto → features. Documentar claramente na UI e no README: "esta é uma plataforma de inferência sobre features já extraídas; extração a partir de sinal bruto é trabalho futuro."

**Mitigação do risco:** incluir no repo um `sample_input.csv` (algumas linhas do dataset de teste) para o avaliador conseguir testar em 10s.

---

## D2. Score de engajamento — RESOLVIDO: **Opção B (model-based proxy)**

**Resultado da inspeção T1:** nenhuma coluna contém `alpha`/`beta`/`theta`/`gamma`/`delta`. As features de frequência são `freq_XXX_N` onde X vai de 010 a 750 e N é o índice do eletrodo (0–3).

**Axioma de frequência (descoberta pós-T1, via pesquisa externa no repo `jordan-bird/eeg-feature-generation`):**
- Dados reamostrados para 150 Hz, janela FFT de 148 amostras → cada bin `freq_XXX` ≈ X/100 Hz.
- Gap em `freq_486`→`freq_517`: notch filter de 50 Hz (AC no Reino Unido).
- Logo bandas clássicas mapeiam para:
  - theta (4–8 Hz): `freq_040`–`freq_080`
  - alpha (8–13 Hz): `freq_080`–`freq_130`
  - beta (13–30 Hz): `freq_130`–`freq_300`

**Decisão adotada:** **score = P(concentrating) da regressão logística, normalizado para 0–100.**

- **Por quê B e não BATR agora:** o time-box não comporta validar os mapeamentos de banda acima e ainda garantir um BATR interpretável. O model-based proxy é coerente por construção com o ground-truth do próprio dataset.
- **O que vira trabalho futuro (documentado no REPORT):** implementar BATR clássico usando o mapeamento de bandas derivado do axioma de frequência acima. Adicionar como painel no app.
- **Validação do proxy:** correlação entre score e classe `concentrating` no conjunto de teste, boxplot por classe.

---

## D3. Protocolo de validação (resolve o #3)

**Decisão:**
1. Split estratificado 80/20 (treino/teste) com `random_state=42`. Teste fica trancado até o fim.
2. No treino: `StratifiedKFold(n_splits=5)` para seleção/comparação de modelos.
3. Número reportado final: métricas no conjunto de teste (1 número por modelo). CV é mencionada como validação interna de estabilidade.

**Razão:** 988 features × ~2364 linhas → CV sozinha é otimista se olharmos só a média. Um holdout fixo dá um número que o avaliador pode reproduzir exatamente e que não é usado para decisão nenhuma.

---

## D4. Vazamento de sujeito — RESOLVIDO: documentar como limitação

**Resultado da inspeção T1:** o CSV agregado do Kaggle não contém IDs de sujeito/participante/sessão. LOSO não é viável sem baixar o repositório original do Bird (jordan-bird/eeg-feature-generation) e reconstruir o pipeline por arquivo-bruto, o que está fora de escopo.

**Decisão adotada:** CV estratificada por janela, com holdout 80/20 estratificado. Limitação registrada honestamente no REPORT como "deployment real em novo usuário exigiria protocolo LOSO; não testado neste dataset."

---

## D5. Estrutura do repositório

```
synapse/
├── data/                       # CSV do Kaggle (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data.py                 # load + dedup + split
│   ├── features.py             # RobustScaler + derivações
│   ├── models.py               # treino LR + XGB
│   └── engagement.py           # BATR ou fallback
├── app/
│   └── streamlit_app.py
├── artifacts/                  # modelo.pkl + scaler.pkl + columns.json
├── sample_input.csv
├── PLANO.md
├── REPORT.md
└── README.md
```

**Por que separar `src/` de `notebooks/`:** notebooks são para explorar e comunicar, módulos `.py` são a fonte de verdade que o app importa. Nunca reescrever lógica no app que já existe em `src/`.
