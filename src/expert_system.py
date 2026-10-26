"""Sistema especialista baseado em regras SE-ENTÃO (Aula 05).

Separação explícita entre:
- Base de conhecimento (KB): regras estruturadas, tipo e justificativa;
- Motor de inferência: avaliação vetorizada das condições e agregação das
  conclusões (votação ponderada na classificação e soma de deltas na
  regressão).

O tempo de elaboração das regras é medido com cronômetro (esforço humano)
e registrado em `logs/cronometro_regras.csv`, separado do tempo
computacional de inferência.

Formato das regras (todas com conjunções E, nunca E/OU misturados):

    Rule(
        rule_id=1,
        problem="classificacao",
        tipo="heurística",
        conditions=[("Anxiety_Score", ">=", 75), ("Depression_Score", ">=", 75)],
        consequent="High",
        weight=3.0,
        justification="...",
    )

Na classificação o peso é um voto para a classe consequente; na regressão o
peso é o delta (em pontos de Burnout_Score) somado ao escore base.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import LOGS_DIR, RULES_DIR

# Operadores aceitos nas condições (todos vetorizáveis no pandas)
_OPERATORS = {
    ">=": lambda series, value: series >= value,
    ">": lambda series, value: series > value,
    "<=": lambda series, value: series <= value,
    "<": lambda series, value: series < value,
    "==": lambda series, value: series == value,
    "!=": lambda series, value: series != value,
}


@dataclass
class Condition:
    """Uma condição atômica de uma regra (feature operador valor)."""

    feature: str
    operator: str
    value: float

    def __post_init__(self) -> None:
        if self.operator not in _OPERATORS:
            raise ValueError(f"Operador inválido: {self.operator}")

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        if self.feature not in frame.columns:
            raise KeyError(
                f"Variável '{self.feature}' não existe no conjunto de dados."
            )
        return _OPERATORS[self.operator](frame[self.feature], self.value)

    def to_dict(self) -> dict:
        return {"feature": self.feature, "operator": self.operator, "value": self.value}

    def __str__(self) -> str:
        return f"{self.feature} {self.operator} {self.value:g}"


@dataclass
class Rule:
    """Representação de uma regra SE-ENTÃO estruturada."""

    rule_id: int
    problem: str  # "classificacao" | "regressao"
    tipo: str  # relação, recomendação, diretriz, estratégia, heurística
    conditions: list[Condition]
    consequent: str
    weight: float
    justification: str
    active: bool = True

    # ---------------------------------------------------------------- utils
    @property
    def antecedent(self) -> str:
        """Antecedente legível (condições unidas por E)."""
        return " AND ".join(str(c) for c in self.conditions)

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """Máscara booleana das linhas que satisfazem TODAS as condições."""
        mask = pd.Series(True, index=frame.index)
        for condition in self.conditions:
            mask &= condition.mask(frame)
        return mask

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "problem": self.problem,
            "tipo": self.tipo,
            "antecedent": self.antecedent,
            "conditions": json.dumps(
                [c.to_dict() for c in self.conditions], ensure_ascii=False
            ),
            "consequent": self.consequent,
            "weight": self.weight,
            "justification": self.justification,
            "active": self.active,
        }

    def to_text(self) -> str:
        return (
            f"REGRA-{self.rule_id:02d} | tipo: {self.tipo}\n"
            f"SE ({self.antecedent}) ENTÃO ({self.consequent}) "
            f"[peso={self.weight:g}]\n"
            f"Justificativa: {self.justification}\n"
        )

    @classmethod
    def from_dict(cls, row: dict) -> "Rule":
        raw = row.get("conditions")
        if raw:
            conditions = [Condition(**c) for c in json.loads(raw)]
        else:  # compatibilidade com o formato antigo (texto livre)
            conditions = []
        return cls(
            rule_id=int(row["rule_id"]),
            problem=row["problem"],
            tipo=row["tipo"],
            conditions=conditions,
            consequent=row["consequent"],
            weight=float(row.get("weight", 1.0) or 1.0),
            justification=row.get("justification", ""),
            active=str(row.get("active", "True")).lower() == "true",
        )


@dataclass
class KnowledgeBase:
    """Base de conhecimento separada do motor de inferência."""

    rules: list[Rule] = field(default_factory=list)
    # Calibração da regressão (reta delta -> Burnout_Score), ajustada no treino
    regression_calibration: dict = field(
        default_factory=lambda: {"intercept": 0.0, "slope": 1.0}
    )

    # ---------------------------------------------------------------- rules
    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def active_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.active]

    def by_problem(self, problem: str) -> list[Rule]:
        return [r for r in self.active_rules() if r.problem == problem]

    # ------------------------------------------------------------ io (csv)
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in self.rules]
        fieldnames = [
            "rule_id",
            "problem",
            "tipo",
            "antecedent",
            "conditions",
            "consequent",
            "weight",
            "justification",
            "active",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        rules: list[Rule] = []
        if not path.exists():
            return cls(rules)
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rules.append(Rule.from_dict(row))
        return cls(rules)

    def save_text(self, path: Path, title: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Base de regras SE-ENTÃO — {title}\n"
            "# Tipos: relação, recomendação, diretriz, estratégia, heurística\n"
            "# Regras elaboradas exclusivamente com o conjunto de TREINO\n"
            "# Conjunções sempre com E (sem misturar E/OU na mesma regra)\n\n"
        )
        body = "\n".join(r.to_text() for r in self.rules)
        path.write_text(header + body, encoding="utf-8")

    # ------------------------------------------------------ io (calibração)
    def save_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"regression_calibration": self.regression_calibration},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_metadata(cls, kb: "KnowledgeBase", path: Path) -> "KnowledgeBase":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            kb.regression_calibration = data.get(
                "regression_calibration", kb.regression_calibration
            )
        return kb


class InferenceEngine:
    """Motor de inferência por encadeamento para frente (forward chaining)."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    # -------------------------------------------------------- classificação
    def vote_matrix(
        self,
        frame: pd.DataFrame,
        classes: Sequence[str] = ("Low", "Moderate", "High"),
    ) -> pd.DataFrame:
        """Matriz de votos (linhas x classes) das regras de classificação."""
        votes = pd.DataFrame(0.0, index=frame.index, columns=list(classes))
        for rule in self.kb.by_problem("classificacao"):
            if rule.consequent not in votes.columns:
                continue
            mask = rule.mask(frame)
            votes.loc[mask, rule.consequent] += rule.weight
        return votes

    def predict_classification(
        self,
        frame: pd.DataFrame,
        classes: Sequence[str] = ("Low", "Moderate", "High"),
    ) -> np.ndarray:
        """Classe prevista = argmax dos votos; empate resolve para a menor
        severidade (Low < Moderate < High)."""
        votes = self.vote_matrix(frame, classes)
        return votes.to_numpy().argmax(axis=1)

    def predict_proba_classification(
        self,
        frame: pd.DataFrame,
        classes: Sequence[str] = ("Low", "Moderate", "High"),
    ) -> np.ndarray:
        """Converte votos em pseudo-probabilidades (normalização por linha).

        Usado apenas para as curvas ROC/PR do especialista. Quando todos os
        votos são zero, aplica-se distribuição uniforme.
        """
        votes = self.vote_matrix(frame, classes).to_numpy(dtype=float)
        totals = votes.sum(axis=1, keepdims=True)
        uniform = np.full_like(votes, 1.0 / votes.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            proba = np.where(totals > 0, votes / totals, uniform)
        return proba

    # ------------------------------------------------------------ regressão
    def regression_delta(self, frame: pd.DataFrame) -> np.ndarray:
        """Soma dos deltas das regras de regressão disparadas."""
        delta = np.zeros(len(frame), dtype=float)
        for rule in self.kb.by_problem("regressao"):
            delta += rule.mask(frame).to_numpy() * rule.weight
        return delta

    def predict_regression(self, frame: pd.DataFrame) -> np.ndarray:
        """Escore = intercepto + inclinação * delta, truncado em [0, 100]."""
        delta = self.regression_delta(frame)
        calibr = self.kb.regression_calibration
        score = calibr.get("intercept", 0.0) + calibr.get("slope", 1.0) * delta
        return np.clip(score, 0.0, 100.0)

    # -------------------------------------------------------------- análise
    def fired_rules(self, row: pd.Series, problem: str) -> list[dict]:
        """Lista as regras disparadas para UMA observação (explicabilidade)."""
        frame = row.to_frame().T
        fired = []
        for rule in self.kb.by_problem(problem):
            try:
                if bool(rule.mask(frame).iloc[0]):
                    fired.append(
                        {
                            "regra": f"REGRA-{rule.rule_id:02d}",
                            "tipo": rule.tipo,
                            "se": rule.antecedent,
                            "entao": rule.consequent,
                            "peso": rule.weight,
                        }
                    )
            except KeyError:
                continue
        return fired

    def rule_firing_counts(self, frame: pd.DataFrame, problem: str) -> pd.DataFrame:
        """Quantas observações cada regra dispara (suporte da regra)."""
        rows = []
        for rule in self.kb.by_problem(problem):
            mask = rule.mask(frame)
            rows.append(
                {
                    "regra": f"REGRA-{rule.rule_id:02d}",
                    "tipo": rule.tipo,
                    "antecedente": rule.antecedent,
                    "consequente": rule.consequent,
                    "peso": rule.weight,
                    "disparos": int(mask.sum()),
                    "cobertura (%)": round(100 * mask.mean(), 3),
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cronometragem do esforço humano
# --------------------------------------------------------------------------
def timer_session() -> dict:
    """Cria um registro de sessão de cronometragem das regras."""
    return {
        "data": datetime.now().strftime("%Y-%m-%d"),
        "horario_inicio": None,
        "horario_fim": None,
        "interrupcoes": "",
        "tempo_total_min": 0.0,
        "n_regras_criadas": 0,
        "problema": None,
        "observacoes": "",
    }


class RuleTimer:
    """Cronômetro para medir o esforço humano na elaboração das regras."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()
        return datetime.now().isoformat()

    def stop(self):
        self.end_time = time.perf_counter()
        return datetime.now().isoformat()

    def elapsed_minutes(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) / 60.0


def save_timer_log(sessions: list[dict], path: Path | None = None) -> Path:
    """Salva o log de cronometragem em `logs/cronometro_regras.csv`."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (LOGS_DIR / "cronometro_regras.csv")

    fieldnames = [
        "data",
        "horario_inicio",
        "horario_fim",
        "interrupcoes",
        "tempo_total_min",
        "n_regras_criadas",
        "problema",
        "tempo_medio_por_regra_min",
        "observacoes",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for session in sessions:
            n = session.get("n_regras_criadas", 0) or 1
            tempo_total = session.get("tempo_total_min", 0.0)
            row = {**session, "tempo_medio_por_regra_min": round(tempo_total / n, 4)}
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return path


def summarize_timer_log(path: Path | None = None) -> pd.DataFrame:
    """Média, desvio-padrão e tempo médio por regra, por problema."""
    path = path or (LOGS_DIR / "cronometro_regras.csv")
    frame = pd.read_csv(path)
    rows = []
    for problem, group in frame.groupby("problema"):
        total = float(group["tempo_total_min"].sum())
        regras = int(group["n_regras_criadas"].sum())
        rows.append(
            {
                "Problema": problem,
                "Sessões": int(len(group)),
                "Tempo total (min)": round(total, 2),
                "Média por sessão (min)": round(float(group["tempo_total_min"].mean()), 2),
                "Desvio-padrão (min)": round(
                    float(group["tempo_total_min"].std(ddof=1)), 2
                )
                if len(group) > 1
                else 0.0,
                "Regras": regras,
                "Tempo médio por regra (min)": round(total / regras, 2) if regras else 0.0,
            }
        )
    return pd.DataFrame(rows)
