"""Carregamento de checkpoints PyTorch e verificação de fidelidade.

O checkpoint contém pesos, arquitetura, estatísticas do StandardScaler, nomes
das features e (na classificação) classes e limiar de decisão. Sem esses itens
o modelo carregado produziria previsões erradas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ROOT
from .models import criar_modelo

MODELS_ROOT = ROOT / "models"


def carregar_checkpoint(caminho: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Carrega o checkpoint com ``weights_only=True`` e ``map_location='cpu'``."""
    return torch.load(caminho, map_location=map_location, weights_only=True)


def reconstruir_modelo(ckpt: dict[str, Any]) -> torch.nn.Module:
    """Recria a arquitetura a partir da configuração salva e restaura os pesos."""
    modelo = criar_modelo(ckpt["config"])
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo


def padronizar(ckpt: dict[str, Any], X: Any) -> torch.Tensor:
    """Aplica exatamente a mesma padronização usada no treino."""
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    return (Xt - ckpt["x_media"]) / ckpt["x_desvio"]


def prever(ckpt: dict[str, Any], modelo: torch.nn.Module, X: Any) -> dict[str, np.ndarray]:
    """Inferência: retorna probabilidades/rótulos (classificação) ou valores (regressão)."""
    x = padronizar(ckpt, X)
    with torch.inference_mode():
        score = modelo(x)

    if "classes" in ckpt:
        prob = torch.softmax(score, dim=1).numpy()
        classes = np.asarray(ckpt["classes"])
        pred = prob.argmax(axis=1)
        limiar = float(ckpt.get("limiar", 0.0))
        if limiar > 0:
            pred = np.where(prob.max(axis=1) < limiar, 0, pred)
        return {"probabilidades": prob, "classes": classes[pred], "indice": pred}

    y = score.squeeze(1).numpy()
    if "y_media" in ckpt:
        y = y * ckpt["y_desvio"].numpy() + ckpt["y_media"].numpy()
    return {"valores": y}


def verificar_fidelidade(nome: str) -> dict[str, float]:
    """Compara o modelo recarregado com os logits salvos no treino."""
    ckpt = carregar_checkpoint(MODELS_ROOT / f"modelo_{nome}.pt")
    modelo = reconstruir_modelo(ckpt)
    dados = np.load(ROOT / "reports" / "predictions" / f"final_{nome}_val_logits.npz")
    import pandas as pd
    from .config import (CLASSIFICATION_DATASET, CLASSIFICATION_TARGET,
                         REGRESSION_DATASET, REGRESSION_TARGET)

    caminho, alvo = ((REGRESSION_DATASET, REGRESSION_TARGET) if nome == "regressao"
                     else (CLASSIFICATION_DATASET, CLASSIFICATION_TARGET))
    df = pd.read_csv(caminho)
    features = ckpt["features"]

    # Os logits salvos seguiram a ordem dos índices de validação.
    from .split import carregar_split
    idx = carregar_split(nome).val
    X = padronizar(ckpt, df[features].to_numpy(dtype=np.float32))[idx]

    with torch.inference_mode():
        logits_novos = modelo(X).numpy()
    diff = float(np.max(np.abs(logits_novos - dados["logits"])))
    return {
        "diferenca_maxima_absoluta": diff,
        "allclose": bool(np.allclose(logits_novos, dados["logits"], atol=1e-6)),
        "n_amostras": int(len(logits_novos)),
    }
