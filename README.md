# DCT1401 — Inteligência Artificial

**Aluno:** José Samuel Silva Lima  
**Matrícula:** 20250038765  
**Disciplina:** DCT1401 — Inteligência Artificial  
**Unidade:** UFRN — CERES — DCT, Campus Caicó  
**Período:** 2026.2  
**Repositório:** [github.com/Jose-Samuel-Lima/InteligenciaArtificial](https://github.com/Jose-Samuel-Lima/InteligenciaArtificial)

**Dataset:** [Mental Health and Burnout Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/mobeenfatimah/mental-health-and-burnout-prediction-dataset/data) — 50.000 registros e 40 variáveis.

---

## Sobre o repositório

Este repositório reúne as atividades da disciplina de Inteligência Artificial. A **Atividade 02** compara três famílias de modelos no mesmo dataset de saúde mental e burnout:

1. **Sistema especialista** baseado em regras SE–ENTÃO;
2. **Árvore de decisão**;
3. **Ensemble learning**: Random Forest (obrigatório) e Gradient Boosting (segundo método).

São avaliados dois problemas:

- **Regressão:** prever `Burnout_Score` (0 a 100);
- **Classificação:** prever `Burnout_Risk` em `Low`, `Moderate` e `High`.

---

## Estrutura

```text
.
├── data/
│   ├── raw/                  # dataset original
│   └── processed/            # datasets codificados usados nos notebooks 03–07
├── docs/
│   ├── relatorio.pdf         # relatório da Atividade 01
│   └── relatorio_final.pdf   # relatório da Atividade 02 (gerado por scripts/gerar_relatorio.py)
├── logs/
│   └── cronometro_regras.csv # esforço humano medido com cronômetro
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_expert_system.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_ensemble.ipynb
│   ├── 06_evaluation_metrics.ipynb
│   └── 07_generalization_efficiency.ipynb
├── reports/
│   ├── figures/              # figuras geradas pelos notebooks
│   ├── models/               # modelos/parâmetros salvos
│   ├── predictions/          # predições (npz)
│   └── tables/               # tabelas/CSVs usados no relatório
├── rules/                    # base de conhecimento do sistema especialista
├── scripts/
│   └── gerar_relatorio.py    # gera docs/relatorio_final.pdf
├── src/
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── expert_system.py      # motor de inferência e base de regras
│   ├── metrics.py
│   ├── profiling.py
│   └── visualization.py
└── requirements.txt
```

Os notebooks `01_eda.ipynb` e `02_preprocessing.ipynb` registram a **Atividade 01** (análise exploratória e preparação). Os datasets finais usados na Atividade 02 estão versionados em `data/processed/`.

---

## Atividade 01 — análise e preparação

- 50.000 registros, 40 variáveis, valores ausentes tratados por mediana (numéricas) e moda (categóricas).
- Variáveis nominais codificadas com One-Hot; ordinais preservam a ordem; binárias em 0/1.
- Distribuição de `Burnout_Risk`: **Low 25.369 (50,738%)**, **Moderate 14.780 (29,560%)**, **High 9.851 (19,702%)**.
- Correlações mais fortes com `Burnout_Score`:

| Variável | Correlação |
|---|---:|
| `Anxiety_Score` | +0,917 |
| `Emotional_Stability` | −0,910 |
| `Mood_Score` | −0,908 |
| `Depression_Score` | +0,904 |
| `Productivity_Score` | −0,885 |
| `Stress_Level` | +0,866 |
| `Chronic_Stress` | +0,627 |
| `Work_Hours_Per_Week` | +0,608 |

---

## Atividade 02 — avaliação comparativa de modelos

### Protocolo

- Divisão **80/20**: 40.000 treino e 10.000 teste; `random_state=42`.
- Classificação com divisão estratificada; regressão com divisão aleatória simples.
- Prevenção de vazamento: excluídas `Burnout_Score`, `Burnout_Risk`, `Mental_Health_Status`, `AI_Wellness_Recommendation` e `Person_ID`.
- Nenhum ajuste foi feito no teste. A calibração da regressão do especialista e os limiares das regras usam apenas o treino.
- Validação cruzada **K=5** no treino para árvore, Random Forest e Gradient Boosting (média ± desvio-padrão).
- Tempo e memória medidos separadamente para treino e inferência; treinamento padronizado em `n_jobs=1` para comparação justa.

### Sistema especialista

- **19 regras** de classificação e **16 regras** de regressão, com condições estruturadas (variável, operador, limiar), peso e justificativa.
- Encadeamento para frente vetorizado; votos ponderados na classificação e soma de deltas calibrada na regressão.
- Esforço humano medido com cronômetro: **3 sessões de classificação** (127 min, 42,33 ± 3,79 min, 6,68 min/regra) e **3 sessões de regressão** (115 min, 38,33 ± 2,08 min, 7,19 min/regra). Total: **242 min**.

### Resultados no teste — classificação

| Modelo | Acurácia | F1-macro | ROC-AUC | PR-AUC | Gap F1 |
|---|---:|---:|---:|---:|---:|
| Especialista | 0,7987 | 0,7907 | 0,8884 | 0,7796 | −0,0078 |
| Árvore de decisão | 0,8096 | 0,7899 | 0,9417 | 0,8498 | 0,0004 |
| Random Forest | 0,8179 | 0,8000 | 0,9463 | 0,8653 | 0,0511 |
| Gradient Boosting | **0,8187** | **0,8009** | **0,9469** | **0,8671** | **0,0005** |

Na classe **High**, o especialista tem o maior recall (**0,8127**), contra 0,7142 da árvore e cerca de 0,77 dos ensembles: ele deixa passar menos casos graves, ao custo de mais falsos positivos.

### Resultados no teste — regressão

| Modelo | MAE | MSE | RMSE | MAPE (%) | R² | MedAE | Gap RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Especialista | 13,1626 | 250,6768 | 15,8328 | 92,55 | 0,6367 | 11,9892 | 0,0144 |
| Árvore de decisão | 7,7679 | 83,9369 | 9,1617 | 39,05 | 0,8784 | 7,2214 | 0,1214 |
| Random Forest | **7,5587** | **78,1057** | **8,8377** | 37,88 | **0,8868** | **7,1666** | 0,8302 |
| Gradient Boosting | 7,6073 | 79,3224 | 8,9063 | 37,90 | 0,8850 | 7,2081 | **0,0625** |

O MAPE é sensível a escores reais próximos de zero (700 observações de teste têm escore 0 e 1.116 têm escore de até 5); por isso MAE, RMSE e MedAE são as métricas principais.

### Validação cruzada K=5

| Modelo | F1-macro (classificação) | RMSE (regressão) |
|---|---:|---:|
| Árvore de decisão | 0,7870 ± 0,0033 | 9,1756 ± 0,0770 |
| Random Forest | 0,7962 ± 0,0045 | 8,8519 ± 0,0628 |
| Gradient Boosting | 0,7966 ± 0,0038 | 8,9115 ± 0,0496 |

### Generalização e eficiência

- **Gradient Boosting** é o mais estável: gap de 0,0005 no F1 e 0,0625 no RMSE.
- **Random Forest** tem o melhor RMSE absoluto, mas o maior gap (0,8302 no RMSE e 0,0511 no F1): árvores profundas memorizam o treino.
- **Árvore ajustada**: gap de 0,0004 no F1 e 0,1214 no RMSE; é o modelo mais leve e rápido.
- **Tamanho em disco**: árvore 5–6 KB; Gradient Boosting 26–57 KB; especialista 6,5–8,5 KB; Random Forest 2,5–2,8 MB.
- **Treinamento single-thread**: árvore 0,67–0,86 s; Random Forest 37,9–79,6 s; Gradient Boosting 23,4–70,7 s.
- **Inferência**: 0,0018–0,0022 ms/amostra (árvore), 0,0025–0,0117 (especialista), 0,0061–0,0146 (Gradient Boosting) e 0,0270–0,0306 (Random Forest).
- **Energia/CO₂** (CPU de 45 W, PUE 1,0, 0,0817 kgCO₂/kWh): treinamento entre 8,3 e 994,7 mWh; inferência de 1.000 amostras abaixo de 0,4 mWh. São estimativas de ordem de grandeza, não medições na tomada.

### Como reproduzir

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Executar os notebooks **na ordem**, a partir da pasta `notebooks/`:

1. `03_expert_system.ipynb` — constrói a base de regras, mede o esforço humano e avalia o especialista.
2. `04_decision_tree.ipynb` — compara critérios, ajusta a árvore com `GridSearchCV`, demonstra overfitting, gera a figura e as 3 regras.
3. `05_ensemble.ipynb` — ajusta Random Forest e Gradient Boosting, gera as curvas de erro e salva os modelos.
4. `06_evaluation_metrics.ipynb` — calcula todas as métricas, gaps exatos, matrizes de confusão, ROC/PR e CV K=5.
5. `07_generalization_efficiency.ipynb` — mede hardware, tempo/memória, tamanho, energia/CO₂ e gera as tabelas comparativas finais.

Os arquivos `reports/models/*.joblib` e `reports/predictions/*.npz` são artefatos binários reproduzíveis e não estão versionados; os notebooks 04 e 05 os geram. Depois de executar 03–07, gerar o relatório com:

```bash
python scripts/gerar_relatorio.py
```

O PDF final é escrito em `docs/relatorio_final.pdf`.

---

## Principais correções desta versão

- Os notebooks 03–07, antes esqueletos sem execução, foram reescritos e executados de ponta a ponta.
- O motor do sistema especialista foi corrigido: a versão anterior não avaliava as condições e retornava sempre falso.
- A base foi ampliada para 19 regras de classificação e 16 de regressão, compatível com o cronômetro.
- As tabelas comparativas foram separadas em **Tabela 2a (classificação)** e **Tabela 2b (regressão)** na seção 6.2; a seção 6.3 trata apenas de generalização.
- Os gaps são calculados diretamente sobre os arrays de predição, sem usar valores arredondados.
- O relatório não contém placeholders: todas as métricas, tempos, memórias e tamanhos vêm dos CSVs gerados pelos notebooks.
