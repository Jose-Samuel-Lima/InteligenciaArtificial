"""Estudo controlado de hiperparâmetros e seleção da configuração final.

Executa, para os dois problemas:

1. configuração de referência (baseline);
2. variação de um fator por vez (6 obrigatórios + 2 opcionais), 3 sementes;
3. varredura 2D taxa de aprendizado × batch (mapa de calor);
4. random search de 20 configurações na validação;
5. treino da configuração final no conjunto completo, 5 sementes, e salvamento
   do checkpoint PyTorch com todo o pré-processamento.

O conjunto de teste não é consultado em nenhum ponto deste módulo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import FIGURES_DIR, LOGS_DIR, REPORTS_DIR, ROOT, SEED
from .experimento import carregar_dados, treinar_config

MODELS_ROOT = ROOT / "models"
PRED_DIR = REPORTS_DIR / "predictions"
TABLES_DIR = REPORTS_DIR / "tables"

BASELINE: dict[str, Any] = {
    "hidden": (32, 32), "activation": "ReLU", "optimizer": "Adam",
    "lr": 1e-3, "batch_size": 128, "dropout": 0.0,
    "weight_decay": 0.0, "initialization": "padrao", "batch_norm": False,
}

# (hiperparâmetro, rótulo, valor da configuração)
GRID: list[tuple[str, Any, dict[str, Any]]] = [
    ("Neurônios por camada", (8, 8), {"hidden": (8, 8)}),
    ("Neurônios por camada", (16, 16), {"hidden": (16, 16)}),
    ("Neurônios por camada", (64, 64), {"hidden": (64, 64)}),
    ("Camadas ocultas", 1, {"hidden": (32,)}),
    ("Camadas ocultas", 3, {"hidden": (32, 32, 32)}),
    ("Camadas ocultas", 5, {"hidden": (32, 32, 32, 32, 32)}),
    ("Ativação", "LeakyReLU", {"activation": "LeakyReLU"}),
    ("Ativação", "Tanh", {"activation": "Tanh"}),
    ("Ativação", "Sigmoid", {"activation": "Sigmoid"}),
    ("Otimizador", "SGD", {"optimizer": "SGD"}),
    ("Otimizador", "SGD_momentum", {"optimizer": "SGD_momentum"}),
    ("Otimizador", "RMSprop", {"optimizer": "RMSprop"}),
    ("Otimizador", "AdamW", {"optimizer": "AdamW"}),
    ("Taxa de aprendizado", 1e-4, {"lr": 1e-4}),
    ("Taxa de aprendizado", 1e-2, {"lr": 1e-2}),
    ("Taxa de aprendizado", 1e-1, {"lr": 1e-1}),
    ("Tamanho do batch", 8, {"batch_size": 8}),
    ("Tamanho do batch", 32, {"batch_size": 32}),
    ("Tamanho do batch", 512, {"batch_size": 512}),
    ("Regularização L2", 1e-5, {"weight_decay": 1e-5}),
    ("Regularização L2", 1e-4, {"weight_decay": 1e-4}),
    ("Regularização L2", 1e-3, {"weight_decay": 1e-3}),
    ("Dropout", 0.1, {"dropout": 0.1}),
    ("Dropout", 0.3, {"dropout": 0.3}),
    ("Dropout", 0.5, {"dropout": 0.5}),
]

N_SEEDS_STUDY = 3
N_SEEDS_FINAL = 5
EPOCHAS_STUDY = 12
PACIENCIA_STUDY = 3
N_SCREENING = 3000

# Em CPUs com poucos núcleos físicos, 2 threads são mais rápidas que o default
# do PyTorch (medido localmente: 4,9 s vs 6,8 s por época em 40 mil amostras).
torch.set_num_threads(2)


def _config_com(overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(BASELINE)
    cfg.update(overrides)
    return cfg


def _rodar(cfg: dict[str, Any], dados: dict[str, Any], sementes: list[int]) -> list[dict[str, Any]]:
    linhas = []
    for s in sementes:
        row = treinar_config(dados, cfg, semente=s, n_epocas=EPOCHAS_STUDY,
                             paciencia=PACIENCIA_STUDY,
                             batch_size=int(cfg["batch_size"]))
        row.update({k: v for k, v in cfg.items() if k != "batch_size"})
        row["batch_size"] = int(cfg["batch_size"])
        row["configuracao"] = json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                          for k, v in cfg.items()})
        linhas.append(row)
    return linhas


def _agregar(linhas: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(linhas)
    g = df.groupby(["hiperparametro", "valor"], dropna=False)["metrica_validacao"]
    agg = g.agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "metrica_media", "std": "metrica_desvio",
                              "count": "n_sementes"})
    return agg


def _grafico_hiperparametros(agg: pd.DataFrame, caminho: Path, titulo: str) -> None:
    hps = list(dict.fromkeys(agg["hiperparametro"].tolist()))
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    for ax, hp in zip(axes.flat, hps):
        sub = agg[agg["hiperparametro"] == hp]
        x = np.arange(len(sub))
        valores = sub["metrica_media"].to_numpy()
        erros = sub["metrica_desvio"].fillna(0).to_numpy()
        ax.bar(x, valores, yerr=erros, capsize=4, color="#1f4e79", alpha=0.85)
        ax.set_xticks(x, [str(v) for v in sub["valor"]], rotation=30, ha="right", fontsize=7)
        ax.set_title(hp, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def _mapa_calor(dados: dict[str, Any], nome: str) -> tuple[pd.DataFrame, Path]:
    lrs = [1e-4, 1e-3, 1e-2]
    batches = [32, 128, 512]
    linhas = []
    for lr in lrs:
        for bs in batches:
            cfg = _config_com({"lr": lr, "batch_size": bs})
            for s in range(SEED, SEED + N_SEEDS_STUDY):
                row = treinar_config(dados, cfg, semente=s, n_epocas=EPOCHAS_STUDY,
                                     paciencia=PACIENCIA_STUDY, batch_size=bs)
                row.update({"lr": lr, "batch_size": bs, "semente": s})
                linhas.append(row)

    df = pd.DataFrame(linhas)
    mat = df.pivot_table(index="lr", columns="batch_size", values="metrica_validacao", aggfunc="mean")
    caminho_csv = TABLES_DIR / f"05_mapa_calor_{nome}.csv"
    mat.to_csv(caminho_csv)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat.to_numpy(), cmap="RdYlGn" if nome == "classificacao" else "RdYlGn_r",
                   aspect="auto")
    ax.set_xticks(range(len(batches)), [str(b) for b in batches])
    ax.set_yticks(range(len(lrs)), [f"1e{int(np.log10(lr))}" for lr in lrs])
    ax.set_xlabel("Tamanho do batch")
    ax.set_ylabel("Taxa de aprendizado")
    for i in range(len(lrs)):
        for j in range(len(batches)):
            ax.text(j, i, f"{mat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Métrica de validação")
    ax.set_title(f"Interação lr × batch — {nome}")
    fig.tight_layout()
    caminho_png = FIGURES_DIR / f"05_mapa_calor_{nome}.png"
    fig.savefig(caminho_png, dpi=150)
    plt.close(fig)
    return mat, caminho_png


def _random_search(dados: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[int, dict[str, Any]]]:
    rng = np.random.default_rng(SEED)
    linhas: list[dict[str, Any]] = []
    configuracoes: dict[int, dict[str, Any]] = {}
    opcoes_hidden = [(16, 16), (32, 32), (64, 64), (32, 32, 32), (16, 16, 16)]
    for i in range(20):
        cfg = {
            "hidden": opcoes_hidden[int(rng.integers(len(opcoes_hidden)))],
            "activation": str(rng.choice(["ReLU", "LeakyReLU", "Tanh", "ELU"])),
            "optimizer": str(rng.choice(["Adam", "AdamW", "RMSprop", "SGD_momentum"])),
            "lr": float(10 ** rng.uniform(-4, -1)),
            "batch_size": int(rng.choice([32, 64, 128, 256])),
            "dropout": float(rng.choice([0.0, 0.1, 0.3])),
            "weight_decay": float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
            "initialization": str(rng.choice(["padrao", "xavier", "he"])),
            "batch_norm": False,
        }
        configuracoes[i] = cfg
        for s in range(SEED, SEED + N_SEEDS_STUDY):
            row = treinar_config(dados, cfg, semente=s, n_epocas=EPOCHAS_STUDY,
                                 paciencia=PACIENCIA_STUDY, batch_size=cfg["batch_size"])
            row.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
            row["id_config"] = i
            linhas.append(row)
        print(f"  random search {i + 1}/20 concluída", flush=True)

    df = pd.DataFrame(linhas)
    agg = (df.groupby("id_config")
             .agg(metrica_media=("metrica_validacao", "mean"),
                  metrica_desvio=("metrica_validacao", "std"))
             .reset_index()
             .sort_values("metrica_media", ascending=(dados["nome"] == "regressao")))
    return linhas, agg, configuracoes


def _escolher_limiar(modelo: torch.nn.Module, dados: dict[str, Any], batch_size: int) -> float:
    """Escolhe a confiança mínima na validação (fallback para a classe majoritária)."""
    Xt = torch.from_numpy(dados["Xval"])
    modelo.eval()
    with torch.inference_mode():
        logits = modelo(Xt).numpy()
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = logits.argmax(axis=1)
    verdades = dados["yval"]
    from sklearn.metrics import f1_score
    melhor, melhor_f1 = 0.0, -1.0
    for limiar in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ajustado = np.where(prob.max(axis=1) < limiar, 0, pred)
        f1 = f1_score(verdades, ajustado, average="macro", zero_division=0)
        if f1 > melhor_f1:
            melhor, melhor_f1 = limiar, f1
    return float(melhor)


def _treinar_final(dados: dict[str, Any], cfg: dict[str, Any], nome: str) -> dict[str, Any]:
    """Treina a configuração vencedora no conjunto completo e salva o checkpoint."""
    print(f"  treino final: {N_SEEDS_FINAL} sementes no conjunto completo", flush=True)
    resultados = []
    for s in range(SEED, SEED + N_SEEDS_FINAL):
        linha = treinar_config(dados, cfg, semente=s, n_epocas=80, paciencia=15,
                               batch_size=int(cfg["batch_size"]), retornar_modelo=True)
        resultados.append(linha)
        print(f"    semente {s}: val={linha['metrica_validacao']:.4f}", flush=True)

    melhor = max(resultados, key=lambda r: r["metrica_validacao"]) if nome == "classificacao" \
        else min(resultados, key=lambda r: r["metrica_validacao"])
    modelo = melhor["modelo"]
    modelo.load_state_dict(melhor["best_state"])
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    ckpt: dict[str, Any] = {
        "state_dict": melhor["best_state"],
        "config": modelo.configuracao(),
        "features": dados["features"],
        "x_media": torch.tensor(dados["x_media"], dtype=torch.float32),
        "x_desvio": torch.tensor(dados["x_desvio"], dtype=torch.float32),
        "lr": float(cfg["lr"]),
        "optimizer": cfg["optimizer"],
        "weight_decay": float(cfg["weight_decay"]),
        "batch_size": int(cfg["batch_size"]),
        "semente": int(melhor["semente"]),
    }
    if nome == "regressao":
        ckpt["y_media"] = torch.tensor(dados["y_scaler"].mean_, dtype=torch.float32)
        ckpt["y_desvio"] = torch.tensor(dados["y_scaler"].scale_, dtype=torch.float32)
    else:
        ckpt["classes"] = dados["classes"]
        ckpt["limiar"] = _escolher_limiar(modelo, dados, int(cfg["batch_size"]))

    caminho = MODELS_ROOT / f"modelo_{nome}.pt"
    torch.save(ckpt, caminho)

    # logits no conjunto de validação: usados para verificar fidelidade após recarregar
    Xt = torch.from_numpy(dados["Xval"])
    modelo.eval()
    with torch.inference_mode():
        logits = modelo(Xt).numpy()
    np.savez(PRED_DIR / f"final_{nome}_val_logits.npz",
             logits=logits, y_true=dados["yval"])

    resumo = {
        "problema": nome,
        "configuracao_final": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "semente_escolhida": int(melhor["semente"]),
        "metrica_validacao": melhor["metrica_validacao"],
        "metrica_treino": melhor["metrica_treino"],
        "gap": melhor["gap_treino_menos_validacao"],
        "epoca_melhor": melhor["epoca_melhor"],
        "epoca_parada": melhor["epoca_parada"],
        "tempo_s": melhor["tempo_s"],
        "parametros": melhor["parametros"],
        "sementes": [
            {"semente": int(r["semente"]), "metrica_treino": r["metrica_treino"],
             "metrica_validacao": r["metrica_validacao"], "tempo_s": r["tempo_s"],
             "epoca_melhor": r["epoca_melhor"], "epoca_parada": r["epoca_parada"],
             "historico": r["history"]}
            for r in resultados
        ],
        "checkpoint": str(caminho),
        "tamanho_checkpoint_kb": round(caminho.stat().st_size / 1024, 2),
        "limiar": ckpt.get("limiar"),
    }
    (REPORTS_DIR / "models" / f"final_{nome}.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumo


def executar_estudo(nome: str) -> dict[str, Any]:
    """Executa o estudo completo para um problema e grava todos os artefatos."""
    print(f"=== ESTUDO: {nome} ===", flush=True)
    dados = carregar_dados(nome, n_screening=N_SCREENING)
    sementes = list(range(SEED, SEED + N_SEEDS_STUDY))
    linhas = _rodar(_config_com({}), dados, sementes)
    for row in linhas:
        row["hiperparametro"], row["valor"] = "Baseline", "-"

    for hp, valor, overrides in GRID:
        print(f"  {hp} = {valor}", flush=True)
        novas = _rodar(_config_com(overrides), dados, sementes)
        for row in novas:
            row["hiperparametro"], row["valor"] = hp, str(valor)
        linhas.extend(novas)

    df = pd.DataFrame(linhas)
    df.insert(0, "problema", nome)
    caminho_log = LOGS_DIR / "hiperparametros.csv"
    header = not caminho_log.exists()
    df.to_csv(caminho_log, mode="a", header=header, index=False)

    agg = _agregar(linhas)
    agg["impacto"] = agg.groupby("hiperparametro")["metrica_media"].transform(
        lambda s: s.max() - s.min())
    sens = agg.sort_values(["impacto", "hiperparametro"], ascending=[False, True])
    sens.to_csv(TABLES_DIR / f"05_sensibilidade_{nome}.csv", index=False)
    _grafico_hiperparametros(agg, FIGURES_DIR / f"05_hiperparametros_{nome}.png",
                             f"Sensibilidade aos hiperparâmetros — {nome}")

    print("  mapa de calor lr × batch", flush=True)
    _mapa_calor(dados, nome)

    print("  random search (20 configurações)", flush=True)
    _, busca, configuracoes = _random_search(dados)
    busca.to_csv(TABLES_DIR / f"05_busca_final_{nome}.csv", index=False)
    melhor_id = int(busca.iloc[0]["id_config"])
    cfg_final = dict(configuracoes[melhor_id])
    (REPORTS_DIR / "models" / f"config_final_{nome}.json").write_text(
        json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg_final.items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    resumo = _treinar_final(carregar_dados(nome), cfg_final, nome)
    print(f"=== CONCLUÍDO: {nome} ===", flush=True)
    return resumo


def main() -> None:
    for nome in ("regressao", "classificacao"):
        executar_estudo(nome)


if __name__ == "__main__":
    main()
