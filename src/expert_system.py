"""Sistema especialista baseado em regras SE-ENTÃO (Aula 05).

Separação explícita entre:
- Base de conhecimento (KB): regras, metadata (tipo, justificativa)
- Motor de inferência: avaliação de condições e agregação de conclusões

O tempo de elaboração das regras é medido com cronômetro (esforço humano)
e registrado em `logs/cronometro_regras.csv` (separado do tempo computacional).
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LOGS_DIR, RULES_DIR


@dataclass
class Rule:
    """Representação de uma regra SE-ENTÃO."""
    rule_id: int
    problem: str  # "regressao" | "classificacao"
    tipo: str      # relação, recomendação, diretriz, estratégia, heurística
    antecedent: str
    consequent: str
    justification: str
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "problem": self.problem,
            "tipo": self.tipo,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "justification": self.justification,
            "active": self.active,
        }


class KnowledgeBase:
    """Base de conhecimento (separada do motor de inferência)."""
    def __init__(self, rules: list[Rule] | None = None):
        self.rules: list[Rule] = rules or []

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def active_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.active]

    def by_problem(self, problem: str) -> list[Rule]:
        return [r for r in self.active_rules() if r.problem == problem]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            if not self.rules:
                return
            writer = csv.DictWriter(f, fieldnames=list(self.rules[0].to_dict().keys()))
            writer.writeheader()
            for rule in self.rules:
                writer.writerow(rule.to_dict())

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        rules = []
        if not path.exists():
            return cls(rules)
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rules.append(
                    Rule(
                        rule_id=int(row["rule_id"]),
                        problem=row["problem"],
                        tipo=row["tipo"],
                        antecedent=row["antecedent"],
                        consequent=row["consequent"],
                        justification=row["justification"],
                        active=row.get("active", "True").lower() == "true",
                    )
                )
        return cls(rules)


class InferenceEngine:
    """Motor de inferência por encadeamento para frente (forward chaining)."""
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def evaluate(self, facts: dict[str, Any], rules: list[Rule] | None = None) -> dict:
        """Avalia regras com base em fatos e retorna conclusões.

        As condições são expressas em linguagem natural na base; a função
        `infer_classification`/`infer_regression` implementa a lógica concreta
        para o dataset utilizado. Este motor provê a interface e a estrutura
        pedidas pela atividade.
        """
        if rules is None:
            rules = self.kb.active_rules()

        fired = []
        conclusions = []

        for rule in rules:
            if self._matches(rule.antecedent, facts):
                fired.append(rule.to_dict())
                conclusions.append(rule.consequent)

        return {
            "fired_rules": fired,
            "conclusions": conclusions,
            "n_rules_evaluated": len(rules),
        }

    def _matches(self, antecedent: str, facts: dict[str, Any]) -> bool:
        # Implementação genérica: lógica concreta em infer_* 
        return False


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


def save_timer_log(sessions: list[dict], problem: str) -> Path:
    """Salva o log de cronometragem em `logs/cronometro_regras.csv`."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "cronometro_regras.csv"

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
            # garante apenas as colunas esperadas
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return path


# Templates para documentação das regras
CLASSIFICATION_RULES_TEMPLATE = """
# Base de regras SE-ENTÃO — Classificação (Burnout_Risk)
# 
# Tipo: relação | recomendação | diretriz | estratégia | heurística
# Regra: SEMPRE com conjunções E/OU separadas, nunca misturar na mesma regra
# Fonte: elaboradas a partir do conjunto de TREINO (sem consulta ao teste)

# Exemplo (formato):
# REGRA-01 | tipo: heurística
# SE (Anxiety_Score > p75 AND Sleep_Quality == Poor) ENTÃO Burnout_Risk = High
# Justificativa: evidências estatísticas do treino + coerência com literatura
""".strip()


REGRESSION_RULES_TEMPLATE = """
# Base de regras SE-ENTÃO — Regressão (Burnout_Score)
#
# Tipo: relação | recomendação | diretriz | estratégia | heurística
# Abordagem: mapeamento por faixas (intervalos) com peso heurístico
# Fonte: elaboradas a partir do conjunto de TREINO (sem consulta ao teste)

# Exemplo (formato):
# REGRA-01 | tipo: relação
# SE (Anxiety_Score >= faixas_altas AND Depression_Score >= faixas_altas) ENTÃO +delta ao Burnout_Score
# Justificativa: correlação forte observada (r>0,90) mas sem duplo-contagem
""".strip()


def initialize_rule_files() -> dict:
    """Cria os arquivos modelo das regras em `rules/`."""
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    class_path = RULES_DIR / "regras_classificacao.txt"
    reg_path = RULES_DIR / "regras_regressao.txt"
    class_path.write_text(CLASSIFICATION_RULES_TEMPLATE, encoding="utf-8")
    reg_path.write_text(REGRESSION_RULES_TEMPLATE, encoding="utf-8")
    return {"classificacao": class_path, "regressao": reg_path}
