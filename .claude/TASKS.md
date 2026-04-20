# TASKS.md — Execução atômica

Legenda:
- **[CLAUDE]** — fazer comigo. Tem decisão de DS, não é boilerplate. Momento de entender.
- **[CURSOR]** — delegar para Cursor/LLM mais barato. É tradução de instrução → código.
- **[T]** — verificação obrigatória antes de marcar como done.

Ordem por fase. Cada fase tem um teto de tempo (PLANO §3).

---

## Fase 0 — Setup (20 min)

- [ ] **T1** [CLAUDE, 10min] Baixar dataset do Kaggle, inspecionar: shape, nomes das 988 colunas, existência de IDs de sujeito, presença/ausência de nomes de banda nas colunas.
  - **[T]** `df.shape == (2479, 989)` e imprimimos 20 primeiros nomes de coluna.
  - **Decisão que esta task resolve:** D2 (BATR vs fallback), D4 (LOSO vs não).
- [ ] **T2** [CURSOR, 5min] `git init`, criar estrutura de pastas (§D5), `.gitignore` para `data/` e `artifacts/`.
- [ ] **T3** [CURSOR, 5min] `requirements.txt` com: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit, jupyter.

---

## Fase 1 — EDA (45 min)

- [ ] **T4** [CLAUDE, 15min] Notebook `01_eda.ipynb`: estatísticas globais, confirmação dos 4.6% de duplicatas, distribuição de classes, detecção de outliers (os 530656 do PLANO).
  - **[T]** cell final imprime "OK: dataset tem 2479 linhas, 3 classes balanceadas, N duplicatas detectadas, M outliers >|1e4|."
  - **Momento de ensino:** por que olhar outliers ANTES de escolher scaler.
- [ ] **T5** [CURSOR, 15min] Plots: barplot de classes, heatmap de correlação (top-50 features por variância), boxplots de 6 features por classe.
  - **[T]** 5 figuras geradas, todas com título e legenda.
- [ ] **T6** [CLAUDE, 15min] ANOVA 1-way por feature para ranquear "features discriminativas" entre estados. Imprimir top-20.
  - **[T]** DataFrame ordenado por F-stat decrescente.
  - **Momento de ensino:** F-stat como proxy rápido de "esta feature carrega sinal de classe".

---

## Fase 2 — Pipeline de features (30 min)

- [ ] **T7** [CLAUDE, 15min] `src/data.py`: `load_raw()`, `deduplicate()`, `stratified_split()` com `random_state=42`. **De-dup ANTES do split.**
  - **[T]** `len(train) + len(test) == len(df_deduped)`, `train.Label.value_counts(normalize=True)` ≈ stratified.
  - **Momento de ensino:** por que de-dup antes do split evita vazamento sutil.
- [ ] **T8** [CLAUDE, 10min] `src/features.py`: `RobustScaler` fit no treino, transform no teste. Salvar em `artifacts/scaler.pkl`.
  - **[T]** `train_scaled.median(axis=0)` ≈ 0, `test_scaled` usa estatísticas do treino.
  - **Momento de ensino:** por que `fit` só no treino (data leakage 101).
- [ ] **T9** [CURSOR, 5min] Salvar `artifacts/columns.json` com a lista ordenada das 988 colunas (o app precisa disso para validar upload).

---

## Fase 3 — Modelos (45 min)

- [ ] **T10** [CLAUDE, 15min] `src/models.py::train_logreg(X_train, y_train)`: LogReg L2, `max_iter=2000`, CV estratificada 5-fold, reportar macro-F1 mean±std.
  - **[T]** CV score > 0.33 (chance). Modelo salvo em `artifacts/logreg.pkl`.
  - **Momento de ensino:** por que LogReg + L2 é o baseline honesto.
- [ ] **T11** [CLAUDE, 15min] `src/models.py::train_xgb(X_train, y_train)`: XGBoost default + `eval_metric='mlogloss'`, mesma CV, mesmo reporting.
  - **[T]** CV score > LogReg, salvo em `artifacts/xgb.pkl`.
  - **Momento de ensino:** o gap LogReg→XGB mede o quanto do problema é não-linear.
- [ ] **T12** [CLAUDE, 15min] Avaliação final no conjunto de teste trancado. Matrizes de confusão + classification_report para ambos.
  - **[T]** 2 matrizes de confusão renderizadas + 2 relatórios impressos.
  - **Momento de ensino:** por que olhar a matriz de confusão, não só accuracy.

---

## Fase 4 — Score de engajamento (30 min)

- [ ] **T13** [CLAUDE, 20min] `src/engagement.py`: implementar BATR se D2 permitir, senão fallback. Normalizar para 0–100.
  - **[T]** score.min() ≥ 0, score.max() ≤ 100, sem NaN.
  - **Momento de ensino:** o que significa "coerente" em um score sem ground-truth direto.
- [ ] **T14** [CURSOR, 10min] Validação: boxplot do score por classe + scatter `score vs P(concentrating)` + correlação de Pearson.
  - **[T]** 2 figuras + 1 número de correlação impresso.

---

## Fase 5 — Streamlit (45 min)

- [ ] **T15** [CURSOR, 15min] `app/streamlit_app.py`: layout com `st.file_uploader`, leitura do CSV, validação de schema contra `artifacts/columns.json`.
  - **[T]** app roda local, rejeita CSV com colunas erradas com mensagem clara.
- [ ] **T16** [CLAUDE, 20min] Integração: carregar scaler + modelo, aplicar, computar score de engajamento, renderizar resultado.
  - **[T]** upload do `sample_input.csv` mostra classe predita + score para cada linha.
  - **Momento de ensino:** por que o app precisa carregar o MESMO scaler do treino.
- [ ] **T17** [CURSOR, 10min] Polimento: título, descrição explicando o schema esperado, botão para baixar `sample_input.csv`.

---

## Fase 6 — Relatório + README (25 min)

- [ ] **T18** [CURSOR, 10min] `README.md`: como instalar, como rodar EDA, como rodar app, como rodar testes.
  - **[T]** seguir o README do zero em terminal limpo funciona.
- [ ] **T19** [CLAUDE, 15min] `REPORT.md` com seções:
  1. O que funcionou
  2. O que não funcionou
  3. O que eu faria diferente
  4. **Plano vs. Entrega** (tabela de 2 colunas)
  - **[T]** cada seção tem ao menos 3 bullets específicos, nenhum genérico.
  - **Momento de ensino:** a diferença entre um relatório honesto e um relatório defensivo.

---

## Fase 7 — Buffer (20 min)

- [ ] **T20** [CLAUDE, 20min] Git: commit por fase (retroativo tudo bem), push, teste final do app em sessão limpa.

---

## Gates de ensino (onde pausar e conversar)

Momentos em que vou EXPLICAR antes de fazer — não pulá-los:

1. **T1 output** — o que os nomes de coluna nos dizem sobre as escolhas D2 e D4.
2. **T7** — por que de-dup antes do split (vazamento sutil mesmo sem labels).
3. **T8** — fit só no treino (o "hello world" do data leakage).
4. **T11** — interpretando o gap LogReg vs XGB.
5. **T13** — validade de score sem ground-truth direto.
6. **T19** — framing honesto do relatório.

Tudo fora disso é execução. Delego para Cursor o que puder ser expresso como "escreva uma função que faz X, Y, Z".
