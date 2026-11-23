"""Inferência em ambiente limpo a partir de um checkpoint ``.pt``.

O script não reutiliza nenhuma variável do treinamento: lê o checkpoint,
recria a arquitetura, aplica o pré-processamento salvo e devolve as previsões.

Exemplo::

    python src/inference.py --modelo models/modelo_classificacao.pt \
        --entrada data/processed/dataset_classification.csv --n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.persistencia import carregar_checkpoint, padronizar, reconstruir_modelo  # noqa: E402

import torch  # noqa: E402


def prever_arquivo(modelo_path: str, entrada: str, n: int | None = None) -> dict:
    """Carrega o modelo e prevê para as primeiras ``n`` linhas de um CSV."""
    ckpt = carregar_checkpoint(modelo_path)
    modelo = reconstruir_modelo(ckpt)

    df = pd.read_csv(entrada)
    faltantes = [c for c in ckpt["features"] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Features ausentes no CSV de entrada: {faltantes[:5]}")
    X = padronizar(ckpt, df[ckpt["features"]].to_numpy(dtype=np.float32))
    if n is not None:
        X = X[:n]

    with torch.inference_mode():
        score = modelo(X)

    if "classes" in ckpt:
        prob = torch.softmax(score, dim=1)
        classes = list(ckpt["classes"])
        indice = prob.argmax(dim=1)
        limiar = float(ckpt.get("limiar", 0.0))
        if limiar > 0:
            indice = torch.where(prob.max(dim=1).values < limiar,
                                 torch.zeros_like(indice), indice)
        return {
            "problema": "classificacao",
            "classes": classes,
            "limiar": limiar,
            "previsoes": [classes[i] for i in indice.numpy()],
            "confianca": [round(float(p), 6) for p in prob.max(dim=1).values.numpy()],
        }

    valores = score.squeeze(1)
    if "y_media" in ckpt:
        valores = valores * ckpt["y_desvio"] + ckpt["y_media"]
    return {
        "problema": "regressao",
        "previsoes": [round(float(v), 4) for v in valores.numpy()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência com modelo PyTorch salvo.")
    parser.add_argument("--modelo", required=True, help="Caminho do arquivo .pt")
    parser.add_argument("--entrada", required=True, help="CSV com as features")
    parser.add_argument("--n", type=int, default=5, help="Número de amostras (padrão: 5)")
    args = parser.parse_args()

    resultado = prever_arquivo(args.modelo, args.entrada, args.n)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
