# REPORT — EEG Mental State Classification

Projeto pessoal time-boxed sobre `birdy654/eeg-brainwave-dataset-mental-state`. Três classes (`relaxed`, `neutral`, `concentrating`), 988 features já extraídas por Bird, 4 sujeitos (2 homens, 2 mulheres), Muse headband (TP9, AF7, AF8, TP10) reamostrado a 150 Hz.

## Números de cabeçalho

| Modelo | CV macro-F1 (5-fold, train) | Test macro-F1 (473 linhas trancadas) |
|---|---|---|
| LogReg L2 multinomial | 0.9534 ± 0.0068 | **0.9528** |
| XGBoost (n=300, depth=6) | 0.9690 ± 0.0070 | **0.9704** |

CV é honesta: o scaler vive **dentro** da `sklearn.Pipeline` de validação cruzada — `RobustScaler` + clip ±10 são re-ajustados em cada fold, então o fold de validação nunca toca estatísticas treinadas sobre ele mesmo. Gap CV→test: -0.0006 (LogReg), +0.0014 (XGB) — dentro do ruído, **sem discrepância suspeita** entre CV e holdout. O conjunto de teste jamais foi visto pelo scaler nem pelos modelos até `scripts/evaluate.py`.

Score de engajamento (0–100) médio por classe verdadeira no teste: relaxed=3.0, neutral=46.5, concentrating=99.5. Spearman `r(score, label_ordinal) = 0.9269`.

---

## 1. O que funcionou

- **De-dup antes do split.** 4.64% das linhas (115) eram duplicatas exatas; remover antes do split eliminou o risco sutil de a mesma janela cair em treino e teste com scores inflados. 2479 → 2364 → 1891/473.
- **RobustScaler + clip a ±10, justificado empiricamente.** A decisão inicial era "usa RobustScaler porque há outliers". Depois de um questionamento do usuário ("mediana ≈ 0, IQR pequeno — ele faz diferença?"), medimos feature-por-feature: para ~50% das features os dois scalers são quase idênticos, mas para ~10% (entradas da matriz de covariância) o `std/IQR` é 30–248×, fazendo `StandardScaler` produzir valores na casa de 25 onde `RobustScaler` produz 6290 em escala bruta. O clip posterior trava o residual. Fração clipada: 0.94% treino, 0.99% teste — bounded sem cortar massa.
- **Dois modelos com papéis distintos.** LogReg para score de engajamento (probabilidades calibradas, interpretáveis) e XGBoost para número de cabeçalho + importância por família. Não é "qual ganha", é "qual serve pra quê".
- **Gap LogReg→XGB pequeno (+1.56 pt em CV, +1.76 pt em teste) lido corretamente.** Sinal de que a maior parte do problema é capturável linearmente — as features pré-extraídas do Bird (FFT, covariância, kurtosis) já fazem o trabalho não-linear pesado, XGBoost só adiciona interações marginais.
- **Importância por família, não por feature.** Agregando os 988 gains do XGBoost em 13 famílias, `freq` domina com 46% da importância acumulada. Bate com o ranking ANOVA univariado — dois ângulos independentes concordando é um sinal forte.
- **Hemisfério direito dominante no top-20 ANOVA.** AF8=7, TP10=7, TP9=6, AF7=0 features. Consistente com Posner & Petersen (1990): atenção sustentada tem viés direito. Não é um achado que a gente procurou, é um que caiu no colo ao ordenar F-stats.
- **Engajamento coerente sem ground-truth direto.** Monotonicidade respeitada entre as classes (3.0 < 46.5 < 99.5), variância intra-classe não nula (usável como sinal contínuo, não apenas 3-níveis disfarçado), Spearman 0.93 com a ordem ordinal.
- **Log append-only como backbone narrativo.** 48 entradas em `artifacts/run_log.jsonl` — findings, decisions, doubts com resolutions, metrics. Este REPORT é consequência do log, não o contrário.

## 2. O que não funcionou (ou: honestamente não fizemos)

- **BATR clássico (Pope 1995) não foi implementado.** O PLANO previa `beta/(alpha+theta)`, mas as colunas do Kaggle não trazem nomes de banda (`alpha_*`, `beta_*`), só bins de FFT (`freq_010_0` … `freq_750_3`). Decodificamos o eixo post-hoc (freq_XXX/100 ≈ Hz, via repositório `jordan-bird/eeg-feature-generation`) — teta = `freq_040–080`, alfa = `freq_080–130`, beta = `freq_130–300` — mas validar esse mapeamento + implementar BATR + calibrar escala dentro do time-box era otimista. Ficou como trabalho futuro documentado.
- **LOSO (leave-one-subject-out) não foi rodado.** O CSV agregado do Kaggle **não traz IDs de sujeito** — exigiria baixar o repo original do Bird e reconstruir o pipeline por arquivo bruto. Ficou fora de escopo. Consequência direta: o número 0.97 é honesto para "janelas novas dos mesmos 4 sujeitos", **não** para "usuário novo que nunca foi visto".
- **Extração de sinal bruto → features não está no app.** O Streamlit aceita CSV já no formato das 988 features. Quem quiser testar com EEG bruto precisaria antes passar pelo pipeline do Bird. Documentado como limitação #1 da plataforma.
- **Amostra pequena demais para reivindicações de generalização.** 4 sujeitos é pouco para qualquer claim "funciona em gente nova". Balanceado por gênero ajuda, mas não substitui N maior. Nosso score é *real* no que mede, mas o que ele mede é mais estreito do que a palavra "concentration detector" sugere.
- **Cursor foi subutilizado nos primeiros passos.** O PLANO previa delegar boilerplate pra LLM mais barato; nas fases 1–2 (EDA, log infra) eu acabei escrevendo código que era delegável. A partir de T14 (plots) corrigi o rumo — T14, T15 e T18 foram do Cursor.

## 3. O que eu faria diferente

- **Começar pela decodificação do eixo de frequência, não pelo EDA tradicional.** Saber que `freq_XXX ≈ Hz/100` desde a hora 1 desbloqueia BATR, desbloqueia interpretação dos top-features em termos de "alfa posterior direito em 10.1 Hz" em vez de "freq_101_2". Seria o primeiro investimento de 20min a fazer diferença.
- **Validar saturação do scaler com 3 features específicas, não com histograma global.** A pergunta "RobustScaler é diferente?" tem resposta *por feature*. Stats globais enganam — `covM_1_1` cru tem range 530k; 95% das features têm range <100. Agregação escondeu a heterogeneidade. Teria economizado uma iteração.
- **Rodar LOSO mesmo que fosse "só com 4 folds"** baixando os CSVs por-sujeito do repo do Bird antes de começar. Quatro folds de LOSO seriam ruidosos mas mostrariam se o drop é de 2 ou 20 pontos — diferença enorme para o framing do REPORT.
- **Separar `engagement_score` do LogReg desde o início, não encaixar depois.** O score acabou dependendo da calibração do LogReg; se eu tivesse começado expondo-o como uma interface (`score(probs) -> float`) daria pra trocar a fonte das probabilidades sem refatorar.
- **Criar `sample_input.csv` no primeiro cadastro do repo.** Fica muito mais fácil para o avaliador testar o app — *ops, esse sim foi feito*, cortesia do botão no sidebar gerado em T15.
- **Feature importance ponderada por família ao invés de SHAP-por-feature.** Com 988 features, top-20 individuais são opacos. Agregar por família (`freq`, `covM`, `eigenval`, etc.) transformou 988 números em 13 — suficiente pra uma conversa com stakeholder não-ML.

## 4. Plano vs. Entrega

| Item do PLANO | Entregue? | Observação |
|---|---|---|
| EDA: shape, duplicatas, outliers, classes | Sim | `notebooks/01_eda.ipynb` com comparação empírica de scalers |
| Ranking ANOVA | Sim | top-20 em F-stat; lag1_logcovM_2_2 no topo com F=1104 |
| De-dup antes do split | Sim | 115 duplicatas removidas, split 80/20 estratificado |
| Scaler (RobustScaler) | Sim | + clip a ±10 após verificação empírica |
| LogReg L2 multinomial, 5-fold CV | Sim | 0.9534 ± 0.0068 (scaler dentro da Pipeline) |
| XGBoost default + same CV | Sim | 0.9690 ± 0.0070 (scaler dentro da Pipeline) |
| Matrizes de confusão + classification_report | Sim | `notebooks/figures/confusion_{logreg,xgb}.png` |
| BATR (beta/(alpha+theta)) | **Não** | CSV sem nomes de banda; axis decodificado mas BATR ficou como futuro |
| Score de engajamento 0–100 | Sim | Opção B: `50*(P(concentrating) - P(relaxed) + 1)` via LogReg |
| Validação do score (scatter, boxplot, correlação) | Sim | Spearman 0.9269, 3 figuras em `notebooks/figures/` |
| Streamlit aceitando CSV | Sim | valida contra `artifacts/columns.json`, botão pra gerar/baixar sample |
| Streamlit mostrando classe + score por linha | Sim | + concordância LogReg–XGB, score médio, classe predominante |
| LOSO | **Não** | Kaggle CSV não tem IDs de sujeito; documentado como limitação |
| REPORT.md | Sim | este arquivo |
| README.md | Sim | ver `README.md` |
| Delegação para Cursor | Parcial | T5/T14/T15/T17/T18 sim; T4/log/T16 ficaram comigo |

## 5. Limitações — o que NÃO foi provado

1. **Generalização entre sujeitos.** 0.97 vale para "nova janela dos mesmos 4 sujeitos". Janelas adjacentes da mesma sessão são quase-idênticas; split aleatório permite que o modelo decore assinaturas de sessão.
2. **Generalização entre sessões.** Nem mesmo protocolo intra-sujeito-inter-sessão foi testado.
3. **Generalização para EEG bruto.** O app recebe features; a ponte "sinal cru → 988 features" do Bird não foi reembalada.
4. **Robustez do score de engajamento.** Validado por monotonicidade e correlação ordinal; não validado contra medida psicométrica externa (NASA-TLX, tempo-na-tarefa, etc.).
5. **Calibração das probabilidades do XGBoost.** Não aplicamos `CalibratedClassifierCV`; por isso o score usa LogReg, cujas probabilidades são naturalmente calibradas.

## 6. Artefatos produzidos

- `artifacts/logreg.pkl`, `artifacts/xgb.pkl`, `artifacts/scaler.pkl`
- `artifacts/columns.json` (988 colunas, ordem do treino — o app valida contra isso)
- `artifacts/sample_input.csv` (5 linhas do teste, para teste rápido do app)
- `artifacts/evaluation.json` (matrizes de confusão + importância por família)
- `artifacts/run_log.jsonl` (48 entradas: findings, decisions, doubts, metrics)
- `notebooks/01_eda.ipynb`
- `notebooks/figures/` — confusion matrices, family importance, engagement boxplot/scatter/histogram

## 7. Como reproduzir

Ver `README.md`. O caminho curto:

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
PYTHONPATH=. python scripts/build_eda_notebook.py    # EDA
PYTHONPATH=. python scripts/train.py                 # scaler + LogReg + XGB + columns.json
PYTHONPATH=. python scripts/evaluate.py              # test-set metrics + family importance
streamlit run app/streamlit_app.py
```
