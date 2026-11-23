"""Arquiteturas de redes neurais (MLP) em PyTorch.

A classe :class:`MLP` é parametrizável: todos os hiperparâmetros estudados na
3ª avaliação (neurônios, camadas, ativação, dropout, normalização e
inicialização) podem ser alterados sem reescrever o código.

A camada de saída produz **logits**, sem ativação:

* regressão: ``out_dim=1`` + ``nn.MSELoss``;
* classificação binária: ``out_dim=1`` + ``nn.BCEWithLogitsLoss``;
* classificação multiclasse (caso deste projeto, ``Low/Moderate/High``):
  ``out_dim=3`` + ``nn.CrossEntropyLoss`` com rótulos inteiros.

As probabilidades só são obtidas na inferência (``softmax`` para multiclasse,
``sigmoid`` para binário), nunca dentro do modelo.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

ATIVACOES_DISPONIVEIS = ("ReLU", "LeakyReLU", "Tanh", "Sigmoid", "ELU", "GELU")
INICIALIZACOES_DISPONIVEIS = ("padrao", "xavier", "he", "zeros")


class MLP(nn.Module):
    """Perceptron multicamadas com hiperparâmetros configuráveis.

    Parameters
    ----------
    in_dim : int
        Número de features de entrada (já padronizadas).
    out_dim : int
        1 para regressão/classificação binária; número de classes para
        classificação multiclasse.
    hidden : tuple[int, ...]
        Quantidade de neurônios em cada camada oculto. Ex.: ``(32, 32)``.
    activation : str
        Nome de uma ativação de :mod:`torch.nn` (ex.: ``"ReLU"``).
    dropout : float
        Probabilidade de desligamento aplicada após cada camada oculto.
        0 desativa a regularização.
    batch_norm : bool
        Se ``True``, insere ``BatchNorm1d`` entre a linear e a ativação.
    initialization : str
        ``"padrao"`` (PyTorch), ``"xavier"``, ``"he"`` ou ``"zeros"``.
        A opção ``"zeros"`` existe apenas para o estudo de inicialização;
        na prática ela quebra a simetria e impede o aprendizado.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: tuple[int, ...] = (32, 32),
        activation: str = "ReLU",
        dropout: float = 0.0,
        batch_norm: bool = False,
        initialization: str = "padrao",
    ) -> None:
        super().__init__()

        if activation not in ATIVACOES_DISPONIVEIS:
            raise ValueError(
                f"Ativação '{activation}' não suportada. "
                f"Escolha entre {ATIVACOES_DISPONIVEIS}."
            )
        if initialization not in INICIALIZACOES_DISPONIVEIS:
            raise ValueError(
                f"Inicialização '{initialization}' não suportada. "
                f"Escolha entre {INICIALIZACOES_DISPONIVEIS}."
            )

        act = getattr(nn, activation)
        camadas: list[nn.Module] = []
        dim = in_dim

        for neuronios in hidden:
            camadas.append(nn.Linear(dim, neuronios))
            if batch_norm:
                camadas.append(nn.BatchNorm1d(neuronios))
            camadas.append(act())
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            dim = neuronios

        # Saída linear: logits. Sem ativação, sem dropout e sem BatchNorm.
        camadas.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*camadas)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = tuple(hidden)
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.initialization = initialization

        self._inicializar_pesos(initialization, activation)

    def _inicializar_pesos(self, metodo: str, activation: str) -> None:
        """Aplica a inicialização escolhida apenas às camadas lineares."""
        if metodo == "padrao":
            return

        for modulo in self.modules():
            if not isinstance(modulo, nn.Linear):
                continue
            if metodo == "xavier":
                nn.init.xavier_uniform_(modulo.weight)
                nn.init.zeros_(modulo.bias)
            elif metodo == "he":
                neg_slope = 0.01 if activation == "LeakyReLU" else 0.0
                nn.init.kaiming_uniform_(
                    modulo.weight, a=neg_slope, nonlinearity="leaky_relu" if neg_slope else "relu"
                )
                nn.init.zeros_(modulo.bias)
            elif metodo == "zeros":
                nn.init.zeros_(modulo.weight)
                nn.init.zeros_(modulo.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna os logits de saída."""
        return self.net(x)

    @property
    def n_parametros(self) -> int:
        """Número de parâmetros treináveis (para a tabela de eficiência)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configuracao(self) -> dict[str, Any]:
        """Configuração serializável, usada no checkpoint ``.pt``."""
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "hidden": list(self.hidden),
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "initialization": self.initialization,
        }


def criar_modelo(config: dict[str, Any]) -> MLP:
    """Recria uma MLP a partir do dicionário salvo no checkpoint."""
    return MLP(
        in_dim=config["in_dim"],
        out_dim=config["out_dim"],
        hidden=tuple(config["hidden"]),
        activation=config["activation"],
        dropout=config["dropout"],
        batch_norm=config.get("batch_norm", False),
        initialization=config.get("initialization", "padrao"),
    )
