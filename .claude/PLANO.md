# PLANO.md — Desafio Synapsee

**Autor:** Gabriel Medeiros
**Data:** 19/04/2026
**Prazo de entrega:** 20/04/2026, 11h (apresentação ao vivo)

---

## 1. Leitura do problema

O desafio é construir um sistema de classificação de estados mentais (relaxado, neutro, concentrado) a partir de sinais de EEG, e derivar um score contínuo de engajamento. O dataset `birdy654/eeg-brainwave-dataset-mental-state` contém 2479 janelas temporais já convertidas em 988 features estatísticas (domínios temporal, espectral, covariância entre canais), coletadas com headband Muse (4 eletrodos: TP9, AF7, AF8, TP10) a 200 Hz.

As classes estão balanceadas (33% cada), então o problema não é desbalanceamento — é **generalização em alta dimensão com poucas amostras** (≈2.5 linhas por feature) e **ruído característico de EEG**. O trabalho real está em (1) validar honestamente, (2) justificar escolhas de feature e modelo, e (3) entregar um score de engajamento coerente com a neurociência do problema.

---

## 2. Decisões de escopo

- **Dataset:** Kaggle CSV pré-processado (988 features + Label), como fornecido.
- **Limpeza:** remover as 115 linhas duplicadas (4.6%) antes de qualquer split, para não inflar métricas.
- **Escala:** `RobustScaler` (mediana + IQR) em vez de `StandardScaler`. Os dados têm outliers severos — valor absoluto máximo de 530656 vs. p99 de 311, uma diferença de quatro ordens de magnitude que quebraria normalização por média/desvio.
- **Validação:** `StratifiedKFold` com 5 folds. Métrica principal: macro-F1.
- **Modelos:** dois, escolhidos para serem **diferentes em espécie**, não variações do mesmo:
  - **Regressão Logística** (L2) — baseline linear, rápido, interpretável. Ancora expectativas sobre quanto sinal é capturável linearmente.
  - **XGBoost** — estado da arte para dados tabulares neste regime (ver Grinsztajn et al., 2022, "Why do tree-based models still outperform deep learning on tabular data").

  A comparação informa quanto da estrutura do problema é linear vs. não-linear. Dois modelos da mesma família (ex: RF + XGB) dariam uma comparação menos informativa.
- **Score de engajamento:** fórmula clássica BATR (β / (α + θ)) de Pope, Bogart & Bartolome (1995), computada sobre as features de frequência do dataset, normalizada para escala 0–100, validada via correlação com a classe `concentrating`.

---

## 3. Etapas e orçamento de tempo

Execução compactada em sprint noturno (~4h efetivas). Cada etapa tem um teto; se estourar, corto escopo e sigo.

| Ordem | Etapa | Tempo | Saída |
|-------|-------|-------|-------|
| 1 | PLANO.md + setup do repo | 20 min | Este documento + estrutura git |
| 2 | EDA (notebook) | 45 min | 5–7 plots direcionados: distribuição de classes, correlações entre features, estatísticas por estado mental |
| 3 | Pipeline de features | 30 min | De-dup → `RobustScaler` → features derivadas (razões de bandas) |
| 4 | Modelos + comparação | 45 min | LogReg e XGBoost com CV estratificado, matrizes de confusão, discussão de overfitting |
| 5 | Score de engajamento | 30 min | BATR + validação contra ground-truth + curva ilustrativa |
| 6 | App Streamlit | 45 min | Upload CSV → predição de estado + score de engajamento |
| 7 | Relatório final + README | 25 min | Markdown honesto do que funcionou, falhou, e faria diferente |
| — | Buffer / polimento / git | 20 min | Commit final, push, sanity check |

**Total planejado:** ~4h. Começo agora e paro para dormir às ~4h30.

---

## 4. Critérios de sucesso

Defino abaixo o que considero um resultado defensável — não o ideal, mas o mínimo que justifica o tempo investido.

- **Modelagem:** ambos os modelos superam chance (0.33 macro-F1) com margem clara sob validação cruzada estratificada. O gap entre LogReg e XGBoost é reportado honestamente, com interpretação sobre o que ele revela do problema.
- **Feature engineering:** cada família de feature usada ou derivada é justificada com uma frase — não "adicionei X porque sim".
- **Score de engajamento:** correlaciona positivamente com a classe `concentrating` no conjunto de validação. Não espero r > 0.8; espero uma correlação estatisticamente não-trivial e uma curva interpretável.
- **App:** roda end-to-end em um CSV de teste. Não precisa ser bonito; precisa funcionar.
- **Honestidade metodológica:** limitações conhecidas estão documentadas no relatório final. Números otimistas são sinalizados como tal.
- **Prazo:** entrega antes das 11h de segunda, com repositório limpo e README que roda.

---

## 5. Limitações conhecidas

Registro aqui o que sei que vai limitar os resultados, para não parecer que descobri depois:

- **Features pré-extraídas, sem sinal bruto.** Não tenho acesso às séries temporais originais, então não posso aplicar arquiteturas EEG-específicas (EEGNet, CNNs 1D). Documentado como trabalho futuro.
- **CSV não inclui IDs de sujeito.** Validação cruzada é estratificada sobre janelas, não sobre sujeitos. Isso é suficiente para o escopo do desafio, mas um deploy real em um novo usuário exigiria protocolo subject-independent (LOSO) — menciono isso no relatório final como próximo passo.
- **Hardware consumer-grade (Muse, eletrodos secos).** Razão sinal/ruído inferior à de EEG clínico. Teto de accuracy intrinsecamente limitado.
- **Sprint noturno de 4h.** Não haverá tuning exaustivo de hiperparâmetros nem ablação extensa. Defaults sensatos + uma rodada de regularização.

---

## 6. Plano de contingência

Se algo travar e eu estiver atrasado em relação ao orçamento:

- **Corto primeiro:** polimento do Streamlit (mantenho funcional, não bonito).
- **Corto segundo:** features derivadas extras (fico só com de-dup + RobustScaler).
- **Nunca corto:** os dois modelos, a validação cruzada, o relatório final honesto, e o README que permite rodar o projeto.

Se algo der errado no XGBoost, cai para Random Forest como substituto — mesma família, comportamento similar, menos hiperparâmetros.

---

**Fim do plano. Começando execução agora.**
