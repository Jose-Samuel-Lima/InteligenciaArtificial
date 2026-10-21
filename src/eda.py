"""Geração das tabelas e figuras da Análise Exploratória de Dados (EDA).

Cada função devolve um `pandas.DataFrame` (tabela) e/ou salva a figura
correspondente em `reports/figures/`. As tabelas são gravadas em
`reports/tables/` para serem referenciadas no relatório e no README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .config import (
    CLASSIFICATION_TARGET,
    FIGURES_DIR,
    FIGURE_DPI,
    ID_COLUMN,
    PALETTE,
    REGRESSION_TARGET,
    TABLES_DIR,
)

sns.set_theme(style="whitegrid", context="notebook")


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Salva uma tabela de resultados em `reports/tables/<name>.csv`."""
    _ensure_dirs()
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8")
    return df


def save_figure(fig, name: str) -> None:
    """Salva uma figura em `reports/figures/<name>.png`."""
    _ensure_dirs()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------
def inspection_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos de dados e dimensionalidade por coluna (inspeção inicial)."""
    rows = []
    for column in df.columns:
        rows.append(
            {
                "Variável": column,
                "Tipo (dtype)": str(df[column].dtype),
                "Classe": "Numérica"
                if pd.api.types.is_numeric_dtype(df[column])
                else "Categórica",
                "Valores únicos": int(df[column].nunique(dropna=True)),
                "Ausentes": int(df[column].isna().sum()),
            }
        )
    return save_table(pd.DataFrame(rows), "eda_inspecao_colunas")


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas das variáveis numéricas."""
    numeric = df.select_dtypes(include=np.number).drop(columns=[ID_COLUMN], errors="ignore")
    stats = numeric.describe().T
    stats["Variância"] = numeric.var()
    stats["Coef. de variação (%)"] = 100 * numeric.std() / numeric.mean().replace(0, np.nan)
    stats["Skew"] = numeric.skew()
    stats = stats.round(4)
    stats.index.name = "Variável"
    return save_table(stats.reset_index(), "eda_estatisticas_descritivas")


def categorical_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Frequências (contagem e percentual) de todas as variáveis categóricas."""
    frames = []
    for column in df.select_dtypes(include="object").columns:
        counts = df[column].value_counts(dropna=False)
        frame = pd.DataFrame(
            {
                "Variável": column,
                "Categoria": counts.index.astype(str),
                "Quantidade": counts.values,
                "Percentual (%)": (100 * counts.values / len(df)).round(3),
            }
        )
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return save_table(result, "eda_frequencias_categoricas")


def correlation_with_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Correlações de Pearson entre as variáveis numéricas e o alvo."""
    numeric = df.select_dtypes(include=np.number).drop(columns=[ID_COLUMN], errors="ignore")
    correlations = numeric.corr(numeric_only=True)[target].drop(target).sort_values(
        key=np.abs, ascending=False
    )
    frame = correlations.to_frame("Correlação (r)").reset_index().rename(
        columns={"index": "Variável"}
    )
    frame["Correlação (r)"] = frame["Correlação (r)"].round(4)
    frame["Força"] = frame["Correlação (r)"].abs().map(
        lambda x: "muito forte" if x >= 0.9 else ("forte" if x >= 0.7 else ("moderada" if x >= 0.4 else "fraca"))
    )
    frame["Sinal"] = frame["Correlação (r)"].map(
        lambda x: "positivo" if x > 0 else ("negativo" if x < 0 else "nulo")
    )
    return save_table(frame, f"eda_correlacao_{target}")


def multicollinearity_pairs(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Pares de variáveis com |r| >= limiar (risco de multicolinearidade)."""
    numeric = df.select_dtypes(include=np.number).drop(columns=[ID_COLUMN], errors="ignore")
    corr = numeric.corr(numeric_only=True)
    rows = []
    columns = corr.columns
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            r = corr.loc[col_a, col_b]
            if abs(r) >= threshold:
                rows.append(
                    {
                        "Par": f"{col_a} x {col_b}",
                        "Correlação (r)": round(float(r), 4),
                        "|r|": round(abs(float(r)), 4),
                    }
                )
    result = pd.DataFrame(rows).sort_values("|r|", ascending=False).reset_index(drop=True)
    return save_table(result, "eda_multicolinearidade")


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def plot_missing_values(missing_log: pd.DataFrame) -> None:
    """Gráfico de barras com a quantidade de ausentes por variável."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = missing_log.sort_values("Quantidade")
    colors = [PALETTE["secondary"] if p >= 1 else PALETTE["primary"] for p in data["Percentual (%)"]]
    ax.barh(data["Variável"], data["Quantidade"], color=colors)
    ax.set_xlabel("Registros ausentes")
    ax.set_title("Valores ausentes por variável (n = 50.000)")
    for y, (q, p) in enumerate(zip(data["Quantidade"], data["Percentual (%)"])):
        ax.text(q + 12, y, f"{q} ({p:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, data["Quantidade"].max() * 1.25)
    save_figure(fig, "missing_values")


def plot_outliers_boxplot(df: pd.DataFrame, columns: list[str]) -> None:
    """Boxplots das principais variáveis numéricas."""
    data = df[columns].melt(var_name="Variável", value_name="Valor")
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=data, x="Variável", y="Valor", ax=ax, color=PALETTE["primary"], fliersize=2)
    ax.set_title("Boxplots das variáveis numéricas (regra IQR × 1,5)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, "outliers_boxplot")


def plot_correlation_matrix(df: pd.DataFrame, columns: list[str]) -> None:
    """Mapa de calor da matriz de correlação de Pearson."""
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 8})
    ax.set_title("Matriz de correlação de Pearson (variáveis numéricas)")
    save_figure(fig, "correlation_matrix")


def plot_target_distribution(df: pd.DataFrame, target: str) -> None:
    """Histograma e boxplot da variável-alvo contínua."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    sns.histplot(df[target], bins=50, kde=True, ax=axes[0], color=PALETTE["primary"])
    axes[0].set_title(f"Distribuição de {target}")
    axes[0].set_xlabel(target)
    sns.boxplot(x=df[target], ax=axes[1], color=PALETTE["accent"])
    axes[1].set_title(f"Boxplot de {target}")
    axes[1].set_xlabel(target)
    fig.tight_layout()
    save_figure(fig, f"{target.lower()}_distribution")


def plot_target_vs_feature(df: pd.DataFrame, feature: str, target: str, kind: str = "reg") -> None:
    """Relação entre uma feature e o alvo."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if kind == "reg":
        sns.regplot(data=df, x=feature, y=target, ax=ax, scatter_kws={"s": 8, "alpha": 0.25},
                    line_kws={"color": PALETTE["secondary"]})
    else:
        sns.boxplot(data=df, x=feature, y=target, ax=ax, color=PALETTE["accent"])
    ax.set_title(f"{target} x {feature}")
    save_figure(fig, f"{target.lower()}_vs_{feature.lower()}")


def plot_class_distribution(df: pd.DataFrame, target: str) -> None:
    """Distribuição das classes do alvo categórico."""
    counts = df[target].value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    order = ["Low", "Moderate", "High"]
    order = [c for c in order if c in counts.index]
    colors = [PALETTE["accent"], "#e8a33d", PALETTE["danger"]][: len(order)]
    bars = ax.bar(counts.loc[order], [counts[c] for c in order], color=colors)
    ax.set_title(f"Distribuição das classes de {target}")
    ax.set_ylabel("Registros")
    for bar, value in zip(bars, [counts[c] for c in order]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                f"{value}\n({100 * value / counts.sum():.3f}%)", ha="center", fontsize=9)
    save_figure(fig, "burnout_risk_distribution")
