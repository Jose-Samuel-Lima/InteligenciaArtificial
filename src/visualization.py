"""Visualizações para o relatório técnico."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR, FIGURE_DPI, PALETTE


def _ensure():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_residuals(y_true, y_pred, name: str = "residuos_regressao"):
    """Gráfico de resíduos: real vs previsto e distribuição dos resíduos."""
    _ensure()
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuos = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.scatterplot(x=y_pred, y=residuos, ax=axes[0], s=8, alpha=0.25)
    axes[0].axhline(0, color=PALETTE["danger"], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Valores previstos")
    axes[0].set_ylabel("Resíduos (y - ŷ)")
    axes[0].set_title("Resíduos vs valores previstos")

    sns.histplot(residuos, bins=60, kde=True, ax=axes[1], color=PALETTE["primary"])
    axes[1].set_title("Distribuição dos resíduos")
    axes[1].set_xlabel("Resíduos")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm_df, name: str = "matriz_confusao"):
    """Heatmap da matriz de confusão."""
    _ensure()
    fig, ax = plt.subplots(figsize=(7, 6))
    data = cm_df.set_index(cm_df.columns[0]) if cm_df.columns[0] == "" else cm_df.set_index(cm_df.columns[0])
    # Garantir que seja numérico
    sns.heatmap(data.astype(int), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Matriz de confusão")
    ax.set_ylabel("Classe real")
    ax.set_xlabel("Classe prevista")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importances, features, name: str = "feature_importance"):
    """Top features por importância."""
    _ensure()
    df_imp = importances.copy()
    df_imp = df_imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(df_imp["Feature"], df_imp["Importance"], color=PALETTE["primary"])
    ax.set_title("Importância das variáveis (Random Forest)")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
