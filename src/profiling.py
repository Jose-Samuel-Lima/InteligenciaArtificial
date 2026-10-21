"""Medição de eficiência computacional (tempo, memória, vazão).

- Sistema especialista: apenas inferência (não há treinamento computacional).
- Árvore de decisão e ensembles: treinamento e inferência.
- Medições repetidas (recomenda-se n_rep=30) para obter média e desvio padrão.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class InferenceProfile:
    """Resultado da medição da inferência."""
    n_amostras: int
    n_repeticoes: int
    tempo_medio_s: float
    tempo_desvio_s: float
    tempo_por_amostra_ms: float
    vazao_amostras_por_s: float
    pico_memoria_kb: float
    pico_memoria_mb: float


def profile_inference(funcao_predicao: Callable, X, n_rep: int = 30) -> InferenceProfile:
    """Mede tempo e pico de memória exclusivamente da etapa de inferência.

    O contador de alocações é iniciado antes do loop e parado após a última
    repetição; o valor reportado é o pico de memória registrado durante as
    execuções (tracealloc).
    """
    tempos = []
    tracemalloc.start()
    try:
        for _ in range(n_rep):
            inicio = time.perf_counter()
            funcao_predicao(X)
            tempos.append(time.perf_counter() - inicio)
        _, pico_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    tempos_arr = np.array(tempos)
    media = float(tempos_arr.mean())
    desvio = float(tempos_arr.std(ddof=1)) if len(tempos_arr) > 1 else 0.0
    n = len(X)

    return InferenceProfile(
        n_amostras=n,
        n_repeticoes=n_rep,
        tempo_medio_s=media,
        tempo_desvio_s=desvio,
        tempo_por_amostra_ms=(media / n) * 1000.0 if n > 0 else 0.0,
        vazao_amostras_por_s=(n / media) if media > 0 else 0.0,
        pico_memoria_kb=pico_bytes / 1024.0,
        pico_memoria_mb=pico_bytes / (1024.0 ** 2),
    )


def profile_training(funcao_treinamento: Callable) -> dict:
    """Mede tempo e pico de memória da etapa de treinamento."""
    tracemalloc.start()
    try:
        inicio = time.perf_counter()
        modelo = funcao_treinamento()
        tempo_treino = time.perf_counter() - inicio
        _, pico_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "tempo_treinamento_s": float(tempo_treino),
        "pico_memoria_treinamento_kb": float(pico_bytes / 1024.0),
        "pico_memoria_treinamento_mb": float(pico_bytes / (1024.0 ** 2)),
        "modelo": modelo,
    }


def format_profile_row(familia: str, modelo: str, perfil: InferenceProfile, treino: dict | None = None) -> dict:
    """Formata linha para a tabela comparativa de eficiência."""
    row = {
        "Família": familia,
        "Modelo": modelo,
        "Tempo treino (s)": treino["tempo_treinamento_s"] if treino else "Não se aplica",
        "Tempo inferência (média ± dp) s": f"{perfil.tempo_medio_s:.6f} ± {perfil.tempo_desvio_s:.6f}",
        "Tempo por amostra (ms)": round(perfil.tempo_por_amostra_ms, 6),
        "Vazão (amostras/s)": round(perfil.vazao_amostras_por_s, 2),
        "Pico memória inferência (KB)": round(perfil.pico_memoria_kb, 2),
        "Pico memória treino (KB)": treino["pico_memoria_treinamento_kb"] if treino else "Não se aplica",
    }
    return row
