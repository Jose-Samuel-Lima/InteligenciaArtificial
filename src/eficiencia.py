"""Medição de eficiência computacional dos modelos finais.

Tempo com ``perf_counter`` (30 repetições, 5 de warm-up), memória residente com
``psutil`` (o ``tracemalloc`` não captura alocações nativas do PyTorch),
energia estimada por potência média × tempo e CO₂ com o fator do Brasil
(45,8 gCO₂/kWh).
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import psutil
import torch

from .config import REPORTS_DIR, ROOT
from .persistencia import carregar_checkpoint, padronizar, reconstruir_modelo
from .split import carregar_split

MODELS_ROOT = ROOT / "models"
POTENCIA_W = 45.0
FATOR_CO2_G_POR_KWH = 45.8

# Mantém o mesmo regime de CPU usado no treino (medido localmente: 2 threads
# são mais rápidas que o default de 4 neste processador).
torch.set_num_threads(2)


def _dados_X(nome: str, ckpt: dict) -> torch.Tensor:
    import pandas as pd
    from .config import CLASSIFICATION_DATASET, REGRESSION_DATASET
    caminho = CLASSIFICATION_DATASET if nome == "classificacao" else REGRESSION_DATASET
    df = pd.read_csv(caminho)
    idx = carregar_split(nome).test
    return padronizar(ckpt, df[ckpt["features"]].to_numpy(dtype="float32"))[idx]


def _macs(modelo: torch.nn.Module) -> int:
    """Multiplicações-acumulações das camadas lineares por amostra."""
    total = 0
    for modulo in modelo.modules():
        if isinstance(modulo, torch.nn.Linear):
            total += int(modulo.in_features) * int(modulo.out_features)
    return total


def medir(nome: str) -> dict:
    caminho = MODELS_ROOT / f"modelo_{nome}.pt"
    ckpt = carregar_checkpoint(caminho)
    modelo = reconstruir_modelo(ckpt)
    X = _dados_X(nome, ckpt)
    proc = psutil.Process()
    n_rep, n_warm = 30, 5

    with torch.inference_mode():
        for _ in range(n_warm):
            modelo(X[:256])
        mem_antes = proc.memory_info().rss
        tempos = []
        for _ in range(n_rep):
            inicio = time.perf_counter()
            modelo(X)
            tempos.append(time.perf_counter() - inicio)
        mem_depois = proc.memory_info().rss

        tempos_1 = []
        for _ in range(n_rep):
            inicio = time.perf_counter()
            modelo(X[:1])
            tempos_1.append(time.perf_counter() - inicio)

    medio = statistics.mean(tempos)
    desvio = statistics.stdev(tempos)
    media_1 = statistics.mean(tempos_1)
    n = len(X)
    final = json.loads((REPORTS_DIR / "models" / f"final_{nome}.json").read_text(encoding="utf-8"))
    tempo_treino = final["tempo_s"]
    energia_treino = POTENCIA_W * tempo_treino / 3_600_000
    energia_inf = POTENCIA_W * medio / 3_600_000

    return {
        "problema": nome,
        "tempo_treino_wall_clock_s": round(tempo_treino, 3),
        "tempo_inferencia_lote_media_s": round(medio, 6),
        "tempo_inferencia_lote_desvio_s": round(desvio, 6),
        "tempo_por_amostra_ms": round(1000 * medio / n, 6),
        "tempo_1_amostra_ms": round(1000 * media_1, 6),
        "vazao_amostras_por_s": round(n / medio, 1),
        "delta_memoria_rss_kb": round((mem_depois - mem_antes) / 1024, 1),
        "memoria_rss_final_mb": round(mem_depois / 1024**2, 1),
        "tamanho_arquivo_kb": round(caminho.stat().st_size / 1024, 2),
        "parametros": modelo.n_parametros,
        "camadas_lineares": sum(isinstance(m, torch.nn.Linear) for m in modelo.modules()),
        "macs_por_amostra": _macs(modelo),
        "threads_cpu": torch.get_num_threads(),
        "energia_treino_kwh": round(energia_treino, 8),
        "energia_inferencia_kwh": round(energia_inf, 8),
        "co2_treino_g": round(energia_treino * FATOR_CO2_G_POR_KWH, 5),
        "co2_inferencia_g": round(energia_inf * FATOR_CO2_G_POR_KWH, 5),
    }


def executar() -> None:
    linhas = [medir(nome) for nome in ("regressao", "classificacao")]
    import pandas as pd
    pd.DataFrame(linhas).to_csv(REPORTS_DIR / "tables" / "08_eficiencia.csv", index=False)
    hardware = {
        "processador": platform.processor() or "Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz",
        "sistema": platform.platform(),
        "memoria_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "threads_cpu": torch.get_num_threads(),
        "pytorch": torch.__version__,
        "potencia_media_w": POTENCIA_W,
        "fator_co2_g_por_kwh": FATOR_CO2_G_POR_KWH,
        "metodo_memoria": "psutil Process().memory_info().rss",
    }
    (REPORTS_DIR / "models" / "08_hardware.json").write_text(
        json.dumps(hardware, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(linhas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    executar()
