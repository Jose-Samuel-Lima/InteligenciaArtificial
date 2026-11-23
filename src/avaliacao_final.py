"""Avaliação final no conjunto de teste, generalização e validação cruzada.

Usa sempre o modelo **carregado do arquivo `.pt`**. O teste é consultado uma
única vez, depois de definidas arquitetura, hiperparâmetros, parada e limiar.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize

from . import metrics as mt
from .config import (CLASSIFICATION_DATASET, FIGURES_DIR, REPORTS_DIR,
                     RISK_CLASSES, ROOT, SEED, TABLES_DIR)
from .experimento import carregar_dados, treinar_config
from .persistencia import carregar_checkpoint, padronizar, reconstruir_modelo
from .split import carregar_split

MODELS_ROOT = ROOT / "models"


def _carregar_modelo(nome: str):
    ckpt = carregar_checkpoint(MODELS_ROOT / f"modelo_{nome}.pt")
    return ckpt, reconstruir_modelo(ckpt)


def _matriz_X(nome: str, ckpt: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    caminho = (CLASSIFICATION_DATASET if nome == "classificacao"
               else ROOT / "data" / "processed" / "dataset_regression.csv")
    df = pd.read_csv(caminho)
    X = padronizar(ckpt, df[ckpt["features"]].to_numpy(dtype=np.float32))
    return X[indices].numpy()


def _salvar(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(TABLES_DIR / name, index=False)


def avaliar_teste(nome: str) -> dict[str, Any]:
    """Métricas no teste com o modelo recarregado, além de curvas e gaps."""
    print(f"=== AVALIAÇÃO FINAL: {nome} ===", flush=True)
    ckpt, modelo = _carregar_modelo(nome)
    split = carregar_split(nome)
    Xtest = _matriz_X(nome, ckpt, split.test)
    Xtrain = _matriz_X(nome, ckpt, split.train)

    with torch.inference_mode():
        score_teste = modelo(torch.from_numpy(Xtest)).numpy()
        score_treino = modelo(torch.from_numpy(Xtrain)).numpy()

    resumo_final = json.loads((REPORTS_DIR / "models" / f"final_{nome}.json").read_text(encoding="utf-8"))
    linhas: list[dict[str, Any]] = []

    if nome == "regressao":
        y_test = split_test_y(nome, split.test)
        y_train = split_test_y(nome, split.train)
        inv = lambda a: a * ckpt["y_desvio"].numpy() + ckpt["y_media"].numpy()
        pred_teste = inv(score_teste.ravel())
        pred_treino = inv(score_treino.ravel())
        m_teste = mt.regression_metrics(y_test, pred_teste)
        m_treino = mt.regression_metrics(y_train, pred_treino)

        # figura: real vs previsto e resíduos
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(y_test, pred_teste, s=4, alpha=0.25, color="#1f4e79")
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        axes[0].set(xlabel="Real", ylabel="Previsto", title="Real vs previsto (teste)")
        resid = y_test - pred_teste
        axes[1].scatter(pred_teste, resid, s=4, alpha=0.25, color="#c55a11")
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set(xlabel="Previsto", ylabel="Resíduo", title="Resíduos (teste)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "07_residuos_regressao.png", dpi=150)
        plt.close(fig)

        for k, v in m_teste.items():
            linhas.append({"Conjunto": "Teste", "Métrica": k, "Valor": v})
        gaps = mt.generalization_gap(m_treino, m_teste)
        for k, v in gaps.items():
            linhas.append({"Conjunto": "Gap (treino-teste)", "Métrica": k, "Valor": v})
        _salvar("07_metricas_teste_regressao.csv", pd.DataFrame(linhas))
        return {"metricas_teste": m_teste, "metricas_treino": m_treino, "gaps": gaps}

    # ---------------- classificação ----------------
    y_test = split_test_y(nome, split.test).astype(int)
    y_train = split_test_y(nome, split.train).astype(int)
    limiar = float(ckpt.get("limiar", 0.0))
    prob = torch.softmax(torch.from_numpy(score_teste), dim=1).numpy()
    pred = prob.argmax(axis=1)
    if limiar > 0:
        pred = np.where(prob.max(axis=1) < limiar, 0, pred)

    m_teste = mt.classification_metrics(y_test, pred, prob, classes=RISK_CLASSES)
    prob_tr = torch.softmax(torch.from_numpy(score_treino), dim=1).numpy()
    pred_tr = prob_tr.argmax(axis=1)
    m_treino = mt.classification_metrics(y_train, pred_tr, prob_tr, classes=RISK_CLASSES)

    _salvar("07_matriz_confusao_classificacao.csv",
            mt.confusion_matrix_frame(y_test, pred, RISK_CLASSES))
    _salvar("07_por_classe_classificacao.csv",
            mt.per_class_report(y_test, pred, RISK_CLASSES))

    # curvas ROC e PR (uma curva por classe + micro/média)
    y_bin = label_binarize(y_test, classes=list(range(len(RISK_CLASSES))))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, classe in enumerate(RISK_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        axes[0].plot(fpr, tpr, label=f"{classe} (AUC={auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y_bin[:, i], prob[:, i])
        axes[1].plot(rec, prec, label=f"{classe} (AP={auc(rec, prec):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC — teste")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall — teste")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_curvas_roc_pr_mlp.png", dpi=150)
    plt.close(fig)

    for k, v in m_teste.items():
        linhas.append({"Conjunto": "Teste", "Métrica": k, "Valor": v})
    gaps = mt.generalization_gap(m_treino, m_teste)
    for k, v in gaps.items():
        linhas.append({"Conjunto": "Gap (treino-teste)", "Métrica": k, "Valor": v})
    _salvar("07_metricas_teste_classificacao.csv", pd.DataFrame(linhas))
    return {"metricas_teste": m_teste, "metricas_treino": m_treino,
            "gaps": gaps, "limiar": limiar}


def split_test_y(nome: str, indices: np.ndarray) -> np.ndarray:
    """Recupera o alvo em escala original para os índices pedidos."""
    from .config import CLASSIFICATION_TARGET, REGRESSION_DATASET, REGRESSION_TARGET
    caminho = (CLASSIFICATION_DATASET if nome == "classificacao" else REGRESSION_DATASET)
    df = pd.read_csv(caminho)
    alvo = CLASSIFICATION_TARGET if nome == "classificacao" else REGRESSION_TARGET
    valores = df[alvo].to_numpy()[indices]
    if nome == "classificacao":
        from .config import RISK_ENCODING
        return np.asarray([RISK_ENCODING[v] for v in valores], dtype=np.int64)
    return valores


def curvas_aprendizado(nome: str) -> None:
    """Figura com perda de treino/validação das 5 sementes da config final."""
    resumo = json.loads((REPORTS_DIR / "models" / f"final_{nome}.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in resumo["sementes"]:
        h = s["historico"]
        e = np.arange(1, len(h["treino_loss"]) + 1)
        ax.plot(e, h["val_loss"], alpha=0.55, linewidth=1,
                label=f"semente {s['semente']} (parou {s['epoca_parada']})")
    # Cada semente pode parar em uma época diferente: alinha pela menor série.
    n_min = min(len(s["historico"]["val_loss"]) for s in resumo["sementes"])
    medias = np.mean([s["historico"]["val_loss"][:n_min] for s in resumo["sementes"]], axis=0)
    ax.plot(np.arange(1, len(medias) + 1), medias, color="black", linewidth=2,
            label="média validação")
    ax.set(xlabel="Época", ylabel="Perda", title=f"Curvas de aprendizado — {nome} (config final)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"07_curvas_aprendizado_{nome}.png", dpi=150)
    plt.close(fig)


def validacao_cruzada(nome: str, n_epocas: int = 60, paciencia: int = 15) -> pd.DataFrame:
    """K=5 na configuração final, sobre os 40 mil registros de treino/validação."""
    print(f"=== K-FOLD 5: {nome} ===", flush=True)
    cfg = json.loads((REPORTS_DIR / "models" / f"config_final_{nome}.json").read_text(encoding="utf-8"))
    cfg["hidden"] = tuple(cfg["hidden"])
    dados = carregar_dados(nome)
    X = np.vstack([dados["Xtr"], dados["Xval"]])
    y = np.concatenate([dados["ytr"], dados["yval"]])
    splitter = (StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                if nome == "classificacao" else KFold(n_splits=5, shuffle=True, random_state=SEED))

    linhas = []
    for fold, (idx_tr, idx_va) in enumerate(splitter.split(X, y), start=1):
        dados_fold = dict(dados)
        dados_fold["Xtr"], dados_fold["ytr"] = X[idx_tr], y[idx_tr]
        dados_fold["Xval"], dados_fold["yval"] = X[idx_va], y[idx_va]
        r = treinar_config(dados_fold, cfg, semente=SEED, n_epocas=n_epocas,
                           paciencia=paciencia, batch_size=int(cfg["batch_size"]))
        linhas.append({"fold": fold, "metrica_validacao": r["metrica_validacao"],
                       "metrica_treino": r["metrica_treino"],
                       "epoca_parada": r["epoca_parada"], "tempo_s": r["tempo_s"]})
        print(f"  fold {fold}: {r['metrica_validacao']:.4f}", flush=True)

    df = pd.DataFrame(linhas)
    df.to_csv(TABLES_DIR / f"07_validacao_cruzada_{nome}.csv", index=False)
    print(f"  média={df['metrica_validacao'].mean():.4f} ± {df['metrica_validacao'].std():.4f}", flush=True)
    return df


def executar() -> None:
    for nome in ("regressao", "classificacao"):
        curvas_aprendizado(nome)
        avaliar_teste(nome)
        validacao_cruzada(nome)


if __name__ == "__main__":
    executar()
