"""Funções de cálculo das métricas da Aula 08 (Avaliação de Modelos de IA).

Regressão: MAE, MSE, RMSE, MAPE, R² e MedAE.
Classificação: matriz de confusão, acurácia, precision, recall, F1,
ROC-AUC e PR-AUC (multiclasse com média macro e ponderada).

As funções retornam também o *gap de generalização*
(treinamento − teste), que é a evidência numérica de overfitting/underfitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_percentage_error as mape,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------
# Regressão
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """Conjunto completo de métricas de regressão."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)

    # MAPE: calculado manualmente, ignorando observações com valor real zero
    # (Burnout_Score = 0 existe na amostra e tornaria o denominador nulo).
    mask = y_true != 0
    if mask.any():
        ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        mape_value = float(np.mean(ape) * 100)
        coverage = float(mask.sum() / len(y_true) * 100)
    else:
        mape_value = float("nan")
        coverage = 0.0

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAPE (%)": mape_value,
        "R²": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "Cobertura MAPE (%)": coverage,
    }


def generalization_gap(metrics_train: dict, metrics_test: dict) -> dict:
    """Gap = desempenho no treino − desempenho no teste.

    Para métricas de erro (MAE, MSE, RMSE, MAPE, MedAE) um gap positivo
    indica erro maior no teste, isto é, overfitting. Para o R² (métrica de
    qualidade), um gap positivo indica queda de desempenho no teste.
    """
    error_metrics = ["MAE", "MSE", "RMSE", "MAPE (%)", "MedAE"]
    gap = {}
    for key, value_train in metrics_train.items():
        value_test = metrics_test[key]
        if not isinstance(value_train, (int, float, np.floating, np.integer)):
            continue
        if not isinstance(value_test, (int, float, np.floating, np.integer)):
            continue
        diff = float(value_train) - float(value_test)
        if key in error_metrics:
            gap[f"Gap {key}"] = diff
        else:
            gap[f"Gap {key}"] = diff
    return gap


# --------------------------------------------------------------------------
# Classificação
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_proba=None, classes=None) -> dict:
    """Conjunto completo de métricas de classificação multiclasse.

    A média macro é usada como métrica principal (todas as classes têm o
    mesmo peso), e a média ponderada é reportada como apoio.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (macro)": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall (macro)": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 (macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Precision (weighted)": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall (weighted)": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 (weighted)": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 1:
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_proba))
            metrics["PR-AUC"] = float(average_precision_score(y_true, y_proba))
        else:
            try:
                metrics["ROC-AUC (OVR macro)"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
                )
                metrics["PR-AUC (macro)"] = float(
                    average_precision_score(y_true, y_proba, average="macro")
                )
            except ValueError:
                metrics["ROC-AUC (OVR macro)"] = float("nan")
                metrics["PR-AUC (macro)"] = float("nan")

    return metrics


def per_class_report(y_true, y_pred, classes) -> pd.DataFrame:
    """Precision, recall e F1 por classe (usado para focar na classe High)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rows = []
    for idx, label in enumerate(classes):
        y_true_bin = (y_true == idx).astype(int)
        y_pred_bin = (y_pred == idx).astype(int)
        support = int(y_true_bin.sum())
        rows.append(
            {
                "Classe": label,
                "Suporte": support,
                "Precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                "Recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "F1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def confusion_matrix_frame(y_true, y_pred, classes) -> pd.DataFrame:
    """Matriz de confusão em formato de tabela (linhas = real)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    frame = pd.DataFrame(matrix, index=[f"Real {c}" for c in classes], columns=[f"Prev. {c}" for c in classes])
    return frame.reset_index().rename(columns={"index": ""})
