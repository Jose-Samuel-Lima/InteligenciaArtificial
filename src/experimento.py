"""Núcleo compartilhado dos experimentos com MLP (3ª avaliação).

Carrega os datasets processados, reutiliza os splits salvos, aplica
padronização ajustada apenas no treino e fornece a função que treina uma
configuração com uma semente e devolve uma linha do log de hiperparâmetros.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSIFICATION_DATASET,
    CLASSIFICATION_TARGET,
    REGRESSION_DATASET,
    REGRESSION_TARGET,
    RISK_ENCODING,
    SEED,
)
from .models import MLP
from .split import carregar_split
from .train import criar_loader, set_seed, treinar_modelo

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def carregar_dados(nome: str, n_screening: int | None = None) -> dict[str, Any]:
    """Prepara arrays de treino/validação/teste para um dos dois problemas.

    ``n_screening`` limita o treino a uma subamostra determinística, usada
    apenas no estudo de hiperparâmetros. O scaler continua sendo ajustado no
    conjunto de treino completo, e a avaliação final usa todos os registros.
    """
    if nome == "regressao":
        caminho, alvo = REGRESSION_DATASET, REGRESSION_TARGET
    elif nome == "classificacao":
        caminho, alvo = CLASSIFICATION_DATASET, CLASSIFICATION_TARGET
    else:
        raise ValueError("nome deve ser 'regressao' ou 'classificacao'.")

    df = pd.read_csv(caminho)
    features = [c for c in df.columns if c != alvo]
    split = carregar_split(nome)

    X = df[features].to_numpy(dtype=np.float32)
    if nome == "classificacao":
        y = df[alvo].map(RISK_ENCODING).to_numpy(dtype=np.int64)
        classes = list(RISK_ENCODING.keys())
    else:
        y = df[alvo].to_numpy(dtype=np.float32)
        classes = []

    x_scaler = StandardScaler().fit(X[split.train])
    Xs = x_scaler.transform(X).astype(np.float32)

    y_scaler = None
    if nome == "regressao":
        y_scaler = StandardScaler().fit(y[split.train].reshape(-1, 1))
        ys = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()
    else:
        ys = y

    idx_tr = split.train if n_screening is None else split.train[:n_screening]
    return {
        "nome": nome,
        "features": features,
        "classes": classes,
        "Xtr": Xs[idx_tr], "ytr": ys[idx_tr],
        "Xval": Xs[split.val], "yval": ys[split.val],
        "Xtest": Xs[split.test], "ytest_raw": y[split.test],
        "y_scaler": y_scaler,
        "x_media": x_scaler.mean_, "x_desvio": x_scaler.scale_,
        "n_treino": len(idx_tr),
    }


def metrica_regressao(y_scaler: StandardScaler | None) -> MetricFn:
    """RMSE na escala original do ``Burnout_Score``."""
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = np.asarray(y_true).reshape(-1, 1)
        yp = np.asarray(y_pred).reshape(-1, 1)
        if y_scaler is not None:
            yt = y_scaler.inverse_transform(yt)
            yp = y_scaler.inverse_transform(yp)
        return float(np.sqrt(mean_squared_error(yt.ravel(), yp.ravel())))
    return _rmse


def metrica_classificacao(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """F1-macro a partir dos logits (argmax das 3 classes)."""
    return float(f1_score(np.asarray(y_true), np.asarray(y_score).argmax(axis=1),
                          average="macro", zero_division=0))


def obter_metrica(dados: dict[str, Any]) -> MetricFn:
    if dados["nome"] == "regressao":
        return metrica_regressao(dados["y_scaler"])
    return metrica_classificacao


def construir_modelo_otimizador(
    config: dict[str, Any], in_dim: int, out_dim: int
) -> tuple[MLP, torch.optim.Optimizer, torch.nn.Module]:
    """Monta MLP, otimizador e função de perda a partir da configuração."""
    modelo = MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden=tuple(config.get("hidden", (32, 32))),
        activation=config.get("activation", "ReLU"),
        dropout=config.get("dropout", 0.0),
        batch_norm=config.get("batch_norm", False),
        initialization=config.get("initialization", "padrao"),
    )

    lr = float(config.get("lr", 1e-3))
    wd = float(config.get("weight_decay", 0.0))
    otimizador_nome = config.get("optimizer", "Adam")
    params = modelo.parameters()
    if otimizador_nome == "SGD":
        opt = torch.optim.SGD(params, lr=lr, weight_decay=wd)
    elif otimizador_nome == "SGD_momentum":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    elif otimizador_nome == "RMSprop":
        opt = torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    elif otimizador_nome == "Adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif otimizador_nome == "AdamW":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Otimizador desconhecido: {otimizador_nome}")

    perda = torch.nn.MSELoss() if out_dim == 1 else torch.nn.CrossEntropyLoss()
    return modelo, opt, perda


def treinar_config(
    dados: dict[str, Any],
    config: dict[str, Any],
    semente: int = SEED,
    n_epocas: int = 30,
    paciencia: int = 6,
    batch_size: int | None = None,
    retornar_modelo: bool = False,
) -> dict[str, Any]:
    """Treina uma configuração com uma semente e devolve métricas do log."""
    tarefa = "regressao" if dados["nome"] == "regressao" else "classificacao"
    bs = batch_size or int(config.get("batch_size", 128))
    gen = set_seed(semente)
    dl_tr = criar_loader(dados["Xtr"], dados["ytr"], bs, True, tarefa, gen)
    dl_val = criar_loader(dados["Xval"], dados["yval"], max(bs, 256), False, tarefa)

    out_dim = 1 if tarefa == "regressao" else len(dados["classes"])
    modelo, opt, perda = construir_modelo_otimizador(config, len(dados["features"]), out_dim)
    resultado = treinar_modelo(
        modelo, dl_tr, dl_val, perda, opt,
        n_epocas=n_epocas, paciencia=paciencia,
        metric_fn=obter_metrica(dados),
    )

    tr = resultado.metricas_finais["treino"]
    va = resultado.metricas_finais["validacao"]
    linha = {
        "semente": semente,
        "metrica_treino": tr,
        "metrica_validacao": va,
        "gap_treino_menos_validacao": tr - va,
        "perda_validacao": resultado.best_val_loss,
        "parametros": modelo.n_parametros,
        "epoca_melhor": resultado.best_epoch,
        "epoca_parada": resultado.stop_epoch,
        "tempo_s": resultado.duration_s,
    }
    if retornar_modelo:
        linha.update({
            "modelo": resultado.model,
            "history": resultado.history,
            "best_state": resultado.best_state,
        })
    return linha
