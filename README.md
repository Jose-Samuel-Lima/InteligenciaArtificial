# DCT1401 — Inteligência Artificial

Repositório destinado às atividades, trabalhos, exercícios e projetos desenvolvidos na disciplina **DCT1401 — Inteligência Artificial**.

---

## 👤 Identificação

**Aluno:** José Samuel Silva Lima
**Matrícula:** 20250038765

**Disciplina:** DCT1401 — Inteligência Artificial

---

## 📚 Sobre a disciplina

Este repositório reúne os materiais e atividades desenvolvidos durante a disciplina de **Inteligência Artificial**, incluindo códigos, notebooks, análises de dados, experimentos e documentos relacionados aos trabalhos realizados.

Os projetos são organizados por atividade para facilitar a execução, acompanhamento e avaliação.

---

# 📂 Organização do repositório

```text
.
├── README.md
│
├── atividades/
│   ├── atividade-01/
│   ├── atividade-02/
│   └── ...
│
├── trabalhos/
│   ├── trabalho-01/
│   ├── trabalho-02/
│   └── ...
│
├── datasets/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   └── ...
│
├── reports/
│   └── ...
│
└── docs/
    └── ...
```

---

# 🧠 Atividades e projetos

## 📊 Análise Exploratória e Preparação de Dados para Machine Learning

Uma das atividades da disciplina consiste na análise de um conjunto de dados relacionado à **saúde mental e burnout**, com o objetivo de preparar os dados para futuros problemas de Machine Learning.

### Dataset

**Mental Health Burnout Prediction Dataset**

O conjunto utilizado possui:

* **50.000 registros**
* **40 variáveis**
* variáveis numéricas e categóricas;
* informações profissionais;
* informações comportamentais;
* informações relacionadas à saúde mental;
* indicadores de burnout.

Arquivo utilizado:

```text
mental_health_burnout_prediction_dataset.csv
```

**Link do dataset:**
(https://www.kaggle.com/datasets/mobeenfatimah/mental-health-and-burnout-prediction-dataset/data)

---

## 🎯 Objetivos da atividade

A atividade tem como objetivos:

* realizar análise exploratória dos dados;
* identificar valores ausentes;
* analisar estatísticas descritivas;
* identificar possíveis outliers;
* analisar correlações;
* verificar possíveis problemas de multicolinearidade;
* realizar limpeza e preparação dos dados;
* analisar um problema de regressão;
* analisar um problema de classificação;
* identificar características relevantes para os problemas;
* recomendar algoritmos para uma futura etapa de Machine Learning.

---

## 📈 Problema de regressão

Para o problema de regressão foi escolhido o atributo:

```text
Burnout_Score
```

O `Burnout_Score` é uma variável numérica utilizada para representar a pontuação de burnout.

Entre as principais relações encontradas na análise estão:

| Variável            | Correlação com Burnout_Score |
| ------------------- | ---------------------------: |
| Anxiety_Score       |                       +0,924 |
| Depression_Score    |                       +0,911 |
| Mood_Score          |                       −0,917 |
| Emotional_Stability |                       −0,910 |
| Productivity_Score  |                       −0,894 |
| Work_Hours_Per_Week |                       +0,608 |

Essas relações indicam que variáveis relacionadas à ansiedade, depressão, humor, estabilidade emocional e produtividade possuem forte associação com a pontuação de burnout.

### Algoritmos inicialmente considerados

* Regressão Linear;
* Ridge;
* Lasso;
* Random Forest Regressor;
* Gradient Boosting Regressor.

---

## 🏷️ Problema de classificação

Para o problema de classificação foi escolhido:

```text
Burnout_Risk
```

O atributo possui três categorias:

```text
Low
Moderate
High
```

### Distribuição das classes

| Categoria | Quantidade | Percentual |
| --------- | ---------: | ---------: |
| Low       |     25.369 |    50,738% |
| Moderate  |     14.780 |    29,560% |
| High      |      9.851 |    19,702% |

As três categorias possuem quantidade significativa de registros, embora exista diferença na distribuição entre elas.

### Principais características analisadas

* `Anxiety_Score`
* `Depression_Score`
* `Mood_Score`
* `Emotional_Stability`
* `Productivity_Score`
* `Stress_Level`
* `Chronic_Stress`
* `Sleep_Quality`
* `Life_Satisfaction`

### Algoritmos inicialmente considerados

* Regressão Logística;
* Random Forest Classifier;
* Gradient Boosting;
* KNN.

---

# 🧹 Análise e preparação dos dados

## Valores ausentes

Foram identificados valores ausentes em algumas variáveis do dataset.

A estratégia definida para o tratamento é:

* variáveis numéricas → **mediana**;
* variáveis categóricas → **moda**.

Essa estratégia evita a perda desnecessária de registros e utiliza valores representativos da própria distribuição dos dados.

---

## Outliers

Os possíveis outliers foram analisados utilizando o método do **Intervalo Interquartil (IQR)**.

A análise realizada não identificou valores que justificassem remoção automática.

Por isso, os valores foram mantidos, evitando a exclusão de observações potencialmente válidas.

---

## Correlação e multicolinearidade

A análise identificou correlações fortes entre algumas variáveis numéricas.

Exemplos:

```text
Anxiety_Score × Emotional_Stability = −0,983
Mood_Score × Productivity_Score = +0,974
Anxiety_Score × Mood_Score = −0,940
```

Essas relações indicam a existência de possível **multicolinearidade**, que deverá ser considerada durante uma futura etapa de modelagem.

---

## Codificação de variáveis categóricas

Para variáveis categóricas nominais será utilizado principalmente **One-Hot Encoding**.

Essa abordagem evita criar uma ordem artificial entre categorias que não possuem relação de ordem.

O Label Encoding poderá ser utilizado para representar o alvo ou categorias que possuam uma ordem natural, quando apropriado.

---

## Padronização

A padronização das variáveis numéricas será considerada para algoritmos sensíveis à escala, como:

* Regressão Logística;
* KNN;
* SVM;
* modelos lineares regularizados.

Modelos baseados em árvores, como Random Forest, não dependem da escala das variáveis da mesma maneira.

---

# ⚠️ Data Leakage

Durante a preparação dos problemas de Machine Learning também será considerado o risco de **vazamento de informação (data leakage)**.

Por exemplo, no problema de classificação, o `Burnout_Score` não deve ser utilizado diretamente como variável preditora de `Burnout_Risk`.

Também devem ser avaliadas variáveis como:

```text
Mental_Health_Status
AI_Wellness_Recommendation
```

caso tenham sido construídas a partir das mesmas informações utilizadas para determinar o risco.

---

# 💻 Tecnologias utilizadas

* Python 3.9+
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Jupyter Notebook

---

# 📓 Notebooks

Os notebooks da atividade de análise de dados são organizados na seguinte ordem:

```text
notebooks/
├── 01_eda.ipynb
├── 02_preprocessing.ipynb
├── 03_regression_analysis.ipynb
└── 04_classification_analysis.ipynb
```

### `01_eda.ipynb`

Realiza a análise exploratória dos dados, incluindo:

* carregamento;
* primeiras e últimas linhas;
* dimensões;
* tipos de dados;
* estatísticas;
* frequências;
* valores ausentes;
* outliers;
* correlações;
* visualizações.

### `02_preprocessing.ipynb`

Realiza a preparação dos dados:

* tratamento dos valores ausentes;
* codificação das categorias;
* padronização;
* preparação dos dados para Machine Learning.

### `03_regression_analysis.ipynb`

Analisa o problema de regressão utilizando:

```text
Burnout_Score
```

Inclui:

* justificativa do alvo;
* distribuição;
* correlações;
* análise das principais características;
* multicolinearidade;
* recomendações de algoritmos.

### `04_classification_analysis.ipynb`

Analisa o problema de classificação utilizando:

```text
Burnout_Risk
```

Inclui:

* distribuição das classes;
* análise das variáveis numéricas;
* ANOVA;
* análise das variáveis categóricas;
* teste qui-quadrado;
* identificação de possíveis vazamentos;
* seleção inicial de características;
* recomendações de algoritmos.

---

# 📁 Dados processados

Os dados utilizados durante o processamento podem ser armazenados em:

```text
data/
├── raw/
│   └── mental_health_burnout_prediction_dataset.csv
│
└── processed/
    ├── dataset_clean.csv
    ├── dataset_regression.csv
    └── dataset_classification.csv
```

### `raw`

Contém o dataset original, sem alterações.

### `processed`

Contém as versões utilizadas após as etapas de limpeza e preparação.

---

# 📊 Relatórios e gráficos

Os gráficos gerados durante as análises serão armazenados em:

```text
reports/
└── figures/
```

Entre as visualizações estão:

* valores ausentes;
* boxplots;
* matriz de correlação;
* distribuição do `Burnout_Score`;
* distribuição do `Burnout_Risk`;
* relação entre horas de trabalho e burnout;
* relação entre ansiedade e risco de burnout.

---

# ⚙️ Instalação

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
```

Entre na pasta:

```bash
cd NOME_DO_REPOSITORIO
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Ou, caso esteja utilizando um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual e depois execute:

```bash
pip install -r requirements.txt
```

---

# ▶️ Execução

Para executar os notebooks, abra o projeto no **VS Code** ou em outro ambiente compatível com Jupyter.

A atividade de análise de dados deve seguir a ordem:

```text
01_eda.ipynb
        ↓
02_preprocessing.ipynb
        ↓
03_regression_analysis.ipynb
        ↓
04_classification_analysis.ipynb
```

Cada notebook apresenta uma etapa específica do processo.

## 📚 Disciplina

**DCT1401 — Inteligência Artificial**

Repositório acadêmico destinado ao armazenamento e organização das atividades desenvolvidas durante a disciplina.
