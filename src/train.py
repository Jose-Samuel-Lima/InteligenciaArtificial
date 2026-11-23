"""Treinamento de MLPs com early stopping e sementes controladas.

O loop segue o padrão do enunciado da 3ª avaliação: separa-se um conjunto de
validação dentro do treino, a perda de validação decide a parada e o melhor
``state_dict`` é restaurado ao final. O conjunto de teste nunca entra aqui.

Também concentra a criação dos ``DataLoader`` (com gerador semeado) e a
reprodutibilidade (``random``, ``numpy`` e ``torch``).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class TrainResult:
    """Resultado de uma execução de treinamento (uma semente)."""

    model: nn.Module
    history: dict[str, list[float]] = field(default_factory=dict)
    best_state: dict[str, torch.Tensor] = field(default_factory=dict)
    best_epoch: int = 0
    stop_epoch: int = 0
    best_val_loss: float = float("inf")
    duration_s: float = 0.0
    metricas_finais: dict[str, float] = field(default_factory=dict)


def set_seed(seed: int = 42) -> torch.Generator:
    """Fixa as sementes de ``random``, ``numpy`` e ``torch``.

    Retorna um gerador do PyTorch para ser passado aos ``DataLoader`` com
    ``shuffle=True``; sem ele, a ordem dos batches também varia entre sementes.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def criar_loader(
    X: Any,
    y: Any,
    batch_size: int,
    shuffle: bool,
    tarefa: str,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Cria um ``DataLoader`` a partir de arrays/DataFrames.

    ``tarefa="classificacao"`` converte o alvo para ``torch.long`` (esperado
    por ``CrossEntropyLoss``); ``tarefa="regressao"`` converte para ``float32``
    e mantém o formato ``(n, 1)``, compatível com a saída linear da MLP.
    """
    if tarefa not in ("classificacao", "regressao"):
        raise ValueError("tarefa deve ser 'classificacao' ou 'regressao'.")

    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    if tarefa == "classificacao":
        yt = torch.as_tensor(np.asarray(y, dtype=np.int64))
    else:
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)

    dataset = TensorDataset(Xt, yt)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,  # no Windows, workers adicionam custo de spawn
    )


def avaliar_modelo(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str | torch.device = "cpu",
    metric_fn: MetricFn | None = None,
) -> dict[str, float]:
    """Avalia o modelo em um loader e retorna perda e métrica opcional.

    ``metric_fn`` recebe ``(y_true, y_score)`` já em NumPy: ``y_score`` são os
    **logits** brutos na classificação e as previsões na escala modelada na
    regressão (cabe a quem define a função aplicar softmax/argmax ou inverter
    a padronização do alvo).
    """
    model.to(device)
    model.eval()
    perda_total = 0.0
    verdades: list[np.ndarray] = []
    scores: list[np.ndarray] = []

    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            score = model(xb)
            perda_total += float(loss_fn(score, yb)) * len(xb)
            verdades.append(yb.cpu().numpy())
            scores.append(score.cpu().numpy())

    resultado = {"loss": perda_total / len(loader.dataset)}
    if metric_fn is not None:
        y_true = np.concatenate(verdades)
        y_score = np.concatenate(scores)
        resultado["metrica"] = float(metric_fn(y_true, y_score))
    return resultado


def treinar_modelo(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epocas: int = 200,
    paciencia: int = 20,
    min_delta: float = 1e-6,
    device: str | torch.device = "cpu",
    metric_fn: MetricFn | None = None,
) -> TrainResult:
    """Treina com early stopping na perda de validação.

    A cada época, o melhor ``state_dict`` é copiado; ao parar, ele é
    restaurado. Assim o modelo devolvido corresponde ao ponto de menor perda
    na validação — não à última época executada.
    """
    model.to(device)
    history: dict[str, list[float]] = {"treino_loss": [], "val_loss": []}
    melhor_loss = float("inf")
    melhor_estado: dict[str, torch.Tensor] = {}
    melhor_epoca = 0
    espera = 0
    epoca_parada = 0

    inicio = time.perf_counter()
    for epoca in range(1, n_epocas + 1):
        model.train()
        soma = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            perda = loss_fn(model(xb), yb)
            perda.backward()
            optimizer.step()
            soma += float(perda.item()) * len(xb)

        treino_loss = soma / len(train_loader.dataset)
        val_loss = avaliar_modelo(model, val_loader, loss_fn, device)["loss"]
        history["treino_loss"].append(treino_loss)
        history["val_loss"].append(val_loss)
        epoca_parada = epoca

        if val_loss < melhor_loss - min_delta:
            melhor_loss = val_loss
            melhor_epoca = epoca
            melhor_estado = copy.deepcopy(model.state_dict())
            espera = 0
        else:
            espera += 1
            if espera >= paciencia:
                break

    duracao = time.perf_counter() - inicio
    if melhor_estado:
        model.load_state_dict(melhor_estado)

    metricas: dict[str, float] = {}
    if metric_fn is not None:
        for nome, loader in (("treino", train_loader), ("validacao", val_loader)):
            metricas[nome] = avaliar_modelo(model, loader, loss_fn, device, metric_fn)["metrica"]

    return TrainResult(
        model=model,
        history=history,
        best_state=melhor_estado,
        best_epoch=melhor_epoca,
        stop_epoch=epoca_parada,
        best_val_loss=melhor_loss,
        duration_s=duracao,
        metricas_finais=metricas,
    )
