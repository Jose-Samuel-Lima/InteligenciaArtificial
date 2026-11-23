"""Divisão treino / validação / teste da 3ª avaliação.

Reaproveita **exatamente** o protocolo da atividade anterior: os arquivos
``dataset_regression.csv`` e ``dataset_classification.csv``, ``SEED=42``,
``TEST_SIZE=0.20`` e divisão estratificada na classificação. A novidade é o
conjunto de validação, separado de dentro do treino (20% dele) para comparar
hiperparâmetros e fazer early stopping.

Os índices são salvos uma única vez em ``data/processed/splits/`` e podem ser
carregados por todos os notebooks, garantindo que nenhuma configuração
reembaralhe os dados.

Uso::

    python -m src.split                 # gera os dois problemas
    python -m src.split --problema classificacao
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CLASSIFICATION_DATASET,
    CLASSIFICATION_TARGET,
    PROCESSED_DIR,
    REGRESSION_DATASET,
    REGRESSION_TARGET,
    SEED,
    TEST_SIZE,
)

SPLITS_DIR = PROCESSED_DIR / "splits"
VAL_SIZE = 0.20  # 20% do conjunto de treino -> 10% do total


@dataclass
class Split:
    """Índices posicionais dos três conjuntos."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    n_amostras: int
    info: dict[str, Any] = field(default_factory=dict)

    def save(self, nome: str) -> tuple[Any, Any]:
        """Grava os índices (``.npz``) e os metadados (``.json``)."""
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        caminho_npz = SPLITS_DIR / f"indices_{nome}.npz"
        caminho_json = SPLITS_DIR / f"info_{nome}.json"
        np.savez(caminho_npz, train=self.train, val=self.val, test=self.test)
        caminho_json.write_text(
            json.dumps(self.info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return caminho_npz, caminho_json


def dividir_indices(
    n_amostras: int,
    y: np.ndarray | pd.Series | None = None,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Split:
    """Gera índices de treino/validação/teste com semente fixa.

    Parameters
    ----------
    n_amostras : int
        Número de linhas do dataset processado.
    y : array-like, optional
        Alvo usado apenas para estratificar. Se ``None``, a divisão é
        aleatória simples (caso da regressão).
    """
    indices = np.arange(n_amostras)
    estrato = np.asarray(y) if y is not None else None

    # Divisão idêntica à da atividade anterior (src.data.split_train_test).
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=estrato
    )

    # Validação separada do treino, estratificada quando há alvo categórico.
    estrato_treino = estrato[idx_train] if estrato is not None else None
    idx_train, idx_val = train_test_split(
        idx_train, test_size=val_size, random_state=seed, stratify=estrato_treino
    )

    return Split(
        train=idx_train,
        val=idx_val,
        test=idx_test,
        n_amostras=n_amostras,
        info={
            "semente": seed,
            "test_size": test_size,
            "val_size_dentro_do_treino": val_size,
            "n_total": int(n_amostras),
            "n_treino": int(len(idx_train)),
            "n_validacao": int(len(idx_val)),
            "n_teste": int(len(idx_test)),
            "estratificada": estrato is not None,
        },
    )


def preparar_split_regressao() -> Split:
    """Gera e salva o split do problema de ``Burnout_Score``."""
    df = pd.read_csv(REGRESSION_DATASET)
    split = dividir_indices(len(df))  # sem estratificação
    split.info["alvo"] = REGRESSION_TARGET
    split.save("regressao")
    return split


def preparar_split_classificacao() -> Split:
    """Gera e salva o split estratificado de ``Burnout_Risk``."""
    df = pd.read_csv(CLASSIFICATION_DATASET)
    split = dividir_indices(len(df), y=df[CLASSIFICATION_TARGET])
    split.info["alvo"] = CLASSIFICATION_TARGET
    split.info["classes"] = sorted(df[CLASSIFICATION_TARGET].unique().tolist())
    split.save("classificacao")
    return split


def carregar_split(nome: str) -> Split:
    """Carrega os índices salvos (``nome`` = ``regressao``/``classificacao``)."""
    dados = np.load(SPLITS_DIR / f"indices_{nome}.npz")
    info = json.loads((SPLITS_DIR / f"info_{nome}.json").read_text(encoding="utf-8"))
    return Split(
        train=dados["train"],
        val=dados["val"],
        test=dados["test"],
        n_amostras=info["n_total"],
        info=info,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera os splits da 3ª avaliação.")
    parser.add_argument(
        "--problema",
        choices=("ambos", "regressao", "classificacao"),
        default="ambos",
    )
    args = parser.parse_args()

    problemas: dict[str, Any] = {}
    if args.problema in ("ambos", "regressao"):
        problemas["regressao"] = preparar_split_regressao()
    if args.problema in ("ambos", "classificacao"):
        problemas["classificacao"] = preparar_split_classificacao()

    for nome, split in problemas.items():
        print(
            f"{nome}: treino={len(split.train)}, val={len(split.val)}, "
            f"teste={len(split.test)} (total={split.n_amostras})"
        )
    print(f"Índices salvos em {SPLITS_DIR}")


if __name__ == "__main__":
    main()
