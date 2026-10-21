"""Carregamento, limpeza e codificação dos datasets do projeto.

Este módulo concentra o pipeline de preparação de dados utilizado pelos
notebooks `02_preprocessing.ipynb` e `03/04_analysis`. Todas as decisões
(treatment de ausentes, outliers, codificação e exclusão por vazamento)
são documentadas nas docstrings e nos dataframes de log retornados, para que
o relatório técnico possa reproduzir exatamente os números apresentados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    BINARY_COLUMNS,
    CLASSIFICATION_DATASET,
    CLASSIFICATION_TARGET,
    CLEAN_DATASET,
    ID_COLUMN,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    NOMINAL_COLUMNS,
    ORDINAL_COLUMNS,
    RAW_DATASET,
    REGRESSION_DATASET,
    REGRESSION_TARGET,
    RISK_ENCODING,
    SEED,
    TEST_SIZE,
)

# --------------------------------------------------------------------------
# Justificativas por coluna (usadas nas tabelas do relatório)
# --------------------------------------------------------------------------

# Motivo pelo qual a mediana foi escolhida para cada coluna numérica
NUMERIC_IMPUTATION_REASON = {
    "Age": "mediana preserva a distribuição de uma variável demográfica "
           "simétrica e não desloca a média de idade da amostra.",
    "Monthly_Income_USD": "a distribuição de renda é fortemente assimétrica à "
                          "direita; a mediana é o estimador robusto (robusto a "
                          "valores extremos) e não é puxada por eles.",
    "Job_Satisfaction": "escala Likert de 1 a 10; a mediana evita deslocar a "
                        "centralidade da escala em direção a valores extremos.",
    "Work_Life_Balance": "escala Likert de 1 a 10; mesma justificativa da "
                         "satisfação no trabalho.",
    "Sleep_Hours": "a mediana mantém o valor típico de sono, que é o "
                   "quantitativo mais interpretável para a análise de "
                   "estresse.",
    "Anxiety_Score": "escore de 0 a 100; a mediana evita que registros "
                     "extremos alterem a escala de ansiedade do conjunto.",
    "Depression_Score": "escore de 0 a 100; mesma justificativa da ansiedade.",
    "Mood_Score": "escore de 0 a 100; mesma justificativa da ansiedade.",
    "Physical_Activity_Hours": "a distribuição é assimétrica à direita; a "
                               "mediana representa o padrão de atividade física "
                               "da amostra.",
    "Meditation_Minutes": "massa fortemente concentrada em zero; a mediana "
                           "representa o comportamento modal da variável.",
    "Screen_Time_Hours": "a mediana preserva a faixa típica de tempo de tela "
                          "diante do uso intensivo de certas subpopulações.",
    "Social_Media_Hours": "distribuição assimétrica; a mediana representa o "
                          "uso típico de redes sociais.",
    "Gaming_Hours": "massa concentrada em valores baixos; a mediana descreve o "
                    "padrão de consumo.",
    "Life_Satisfaction": "escala Likert de 1 a 10; a mediana mantém o nível "
                         "típico de satisfação.",
    "Productivity_Score": "escore de 0 a 100; a mediana evita viés por "
                           "valores extremos de produtividade.",
}

# Motivo pelo qual a moda foi escolhida para cada coluna categórica
CATEGORICAL_IMPUTATION_REASON = {
    "Therapy_Attendance": "variável binária Yes/No; a moda (Yes) é a "
                          "categoria majoritária e a imputação mantém a "
                          "proporção observada na amostra.",
}


@dataclass
class CleaningReport:
    """Resultado da etapa de limpeza, com os logs auditáveis."""

    clean: pd.DataFrame
    missing_log: pd.DataFrame
    outliers_log: pd.DataFrame
    encoding_log: pd.DataFrame


# --------------------------------------------------------------------------
# Carregamento
# --------------------------------------------------------------------------
def load_raw(path=RAW_DATASET) -> pd.DataFrame:
    """Carrega o dataset bruto (sem qualquer alteração)."""
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Diagnóstico de valores ausentes
# --------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Quantifica os valores ausentes de cada coluna e a estratégia aplicada.

    Retorna um dataframe com quantidade, percentual, tipo da variável,
    estratégia de imputação e a justificativa técnica da escolha.
    """
    rows = []
    for column in df.columns:
        n_missing = int(df[column].isna().sum())
        if n_missing == 0:
            continue

        if column in NUMERIC_IMPUTATION_REASON:
            strategy = "Mediana"
            reason = NUMERIC_IMPUTATION_REASON[column]
        elif column in CATEGORICAL_IMPUTATION_REASON:
            strategy = "Moda"
            reason = CATEGORICAL_IMPUTATION_REASON[column]
        else:
            strategy = "Moda"
            reason = (
                "Variável categórica: a moda é a categoria majoritária e "
                "preserva a distribuição observada."
            )

        rows.append(
            {
                "Variável": column,
                "Quantidade": n_missing,
                "Percentual (%)": round(100 * n_missing / len(df), 3),
                "Tipo": "Numérica" if pd.api.types.is_numeric_dtype(df[column]) else "Categórica",
                "Estratégia": strategy,
                "Justificativa": reason,
            }
        )

    return pd.DataFrame(rows).sort_values("Quantidade", ascending=False).reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> CleaningReport:
    """Executa a limpeza: imputação de ausentes e diagnóstico de outliers.

    Nenhuma linha é removida: a imputação preserva as 50.000 observações e
    os outliers diagnosticados são mantidos por serem plausíveis no domínio.
    """
    clean = df.copy()
    log = missing_report(df)

    # --- Imputação de variáveis numéricas pela mediana ---------------------
    numeric_columns = [
        c for c in clean.select_dtypes(include=np.number).columns if c != ID_COLUMN
    ]
    for column in numeric_columns:
        clean[column] = clean[column].fillna(clean[column].median())

    # --- Imputação de variáveis categóricas pela moda -----------------------
    categorical_columns = clean.select_dtypes(include="object").columns
    for column in categorical_columns:
        clean[column] = clean[column].fillna(clean[column].mode()[0])

    outliers = outliers_report(clean)

    encoding = encoding_report()

    return CleaningReport(
        clean=clean,
        missing_log=log,
        outliers_log=outliers,
        encoding_log=encoding,
    )


# --------------------------------------------------------------------------
# Diagnóstico de outliers (IQR * 1.5)
# --------------------------------------------------------------------------
def outliers_report(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a regra do Intervalo Interquartil (1,5 * IQR) em cada coluna.

    Para cada variável numérica são reportados os limites, a quantidade e o
    percentual de registros fora dos limites. A quantidade é a evidência
    numérica que sustenta a decisão de manter as observações.
    """
    rows = []
    numeric_columns = [
        c for c in df.select_dtypes(include=np.number).columns if c != ID_COLUMN
    ]

    for column in numeric_columns:
        series = df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_out = int(((series < lower) | (series > upper)).sum())

        rows.append(
            {
                "Variável": column,
                "Mínimo": round(float(series.min()), 3),
                "Q1": round(float(q1), 3),
                "Mediana": round(float(series.median()), 3),
                "Q3": round(float(q3), 3),
                "Máximo": round(float(series.max()), 3),
                "IQR": round(float(iqr), 3),
                "Limite inferior": round(float(lower), 3),
                "Limite superior": round(float(upper), 3),
                "Outliers (n)": n_out,
                "Outliers (%)": round(100 * n_out / len(series), 3),
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Codificação
# --------------------------------------------------------------------------
def encoding_report() -> pd.DataFrame:
    """Documenta a estratégia de codificação adotada para cada categórica."""
    rows = []

    for column, order in ORDINAL_COLUMNS.items():
        rows.append(
            {
                "Variável": column,
                "Natureza": "Ordinal",
                "Técnica": "OrdinalEncoder (ordem fixa do domínio)",
                "Justificativa": (
                    "Possui ordem natural de severidade/crescimento; o One-Hot "
                    "Encoding destruiria essa informação e o Label Encoding "
                    "arbitrário criaria uma ordem sem significado."
                ),
                "Ordem utilizada": " < ".join(order),
            }
        )

    for column in BINARY_COLUMNS:
        rows.append(
            {
                "Variável": column,
                "Natureza": "Binária",
                "Técnica": "Mapeamento fixo Yes=1 / No=0",
                "Justificativa": (
                    "Duas categorias mutuamente exclusivas: uma única coluna "
                    "binária é equivalente ao One-Hot e reduz a dimensionalidade."
                ),
                "Ordem utilizada": "No < Yes",
            }
        )

    for column in NOMINAL_COLUMNS:
        rows.append(
            {
                "Variável": column,
                "Natureza": "Nominal",
                "Técnica": "One-Hot Encoding",
                "Justificativa": (
                    "Categorias sem relação de ordem; o One-Hot Encoding evita "
                    "impor uma ordem artificial entre categorias."
                ),
                "Ordem utilizada": "-",
            }
        )

    return pd.DataFrame(rows)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a codificação às variáveis categóricas.

    - Nominais  -> One-Hot Encoding (drop_first=False, para manter a
      interpretabilidade das dummies e evitar dependencia da ordem das
      colunas na leitura dos coeficientes);
    - Ordinais  -> OrdinalEncoder com a ordem do domínio;
    - Binárias  -> 0/1.

    A normalização (StandardScaler) NÃO é aplicada aqui: ela é feita dentro
    de `Pipeline`, ajustada apenas com os dados de treino, para evitar
    vazamento de informação. Ver `src/models.py`.
    """
    encoded = df.copy()

    # --- Nominais: One-Hot Encoding ---------------------------------------
    encoded = pd.get_dummies(
        encoded, columns=NOMINAL_COLUMNS, drop_first=False, dtype=int
    )

    # --- Ordinais: ordem explícita do domínio -----------------------------
    for column, order in ORDINAL_COLUMNS.items():
        encoded[column] = pd.Categorical(
            encoded[column], categories=order, ordered=True
        ).codes.astype(int)

    # --- Binárias: Yes=1 / No=0 -------------------------------------------
    for column in BINARY_COLUMNS:
        encoded[column] = encoded[column].map({"No": 0, "Yes": 1}).astype(int)

    return encoded


# --------------------------------------------------------------------------
# Montagem dos datasets por problema
# --------------------------------------------------------------------------
def build_regression_dataset(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Dataset da regressão: features (sem vazamento) + Burnout_Score."""
    features = df_encoded.drop(
        columns=[
            REGRESSION_TARGET,
            CLASSIFICATION_TARGET,
            ID_COLUMN,
            *LEAKAGE_COLUMNS,
        ],
        errors="ignore",
    )
    dataset = features.copy()
    dataset[REGRESSION_TARGET] = df_encoded[REGRESSION_TARGET]
    return dataset


def build_classification_dataset(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Dataset da classificação: features (sem vazamento) + Burnout_Risk."""
    features = df_encoded.drop(
        columns=[
            CLASSIFICATION_TARGET,
            REGRESSION_TARGET,
            ID_COLUMN,
            *LEAKAGE_COLUMNS,
        ],
        errors="ignore",
    )
    dataset = features.copy()
    dataset[CLASSIFICATION_TARGET] = df_encoded[CLASSIFICATION_TARGET]
    return dataset


def save_processed(
    clean: pd.DataFrame,
    regression: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    """Grava os três datasets processados em `data/processed/`."""
    clean.to_csv(CLEAN_DATASET, index=False)
    regression.to_csv(REGRESSION_DATASET, index=False)
    classification.to_csv(CLASSIFICATION_DATASET, index=False)


# --------------------------------------------------------------------------
# Protocolo experimental (usado também na atividade de avaliação de modelos)
# --------------------------------------------------------------------------
def split_train_test(
    dataset: pd.DataFrame,
    target: str,
    stratify: bool = False,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
):
    """Divisão 80/20 com semente fixa (estratificada na classificação)."""
    X = dataset.drop(columns=[target])
    y = dataset[target]

    stratify_y = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_y
    )


def encode_target(labels: pd.Series) -> pd.Series:
    """Codifica o alvo de classificação (Low=0, Moderate=1, High=2)."""
    return labels.map(RISK_ENCODING)
