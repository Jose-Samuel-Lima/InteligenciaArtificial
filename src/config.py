"""Configurações centralizadas do projeto (DCT1401 — Inteligência Artificial).

Centraliza caminhos, sementes aleatórias, listas de variáveis e decisões de
codificação para que todos os notebooks e scripts produzam resultados
reproduzíveis.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = ROOT / "notebooks"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DOCS_DIR = ROOT / "docs"
RULES_DIR = ROOT / "rules"
LOGS_DIR = ROOT / "logs"

RAW_DATASET = RAW_DIR / "mental_health_burnout_prediction_dataset.csv"
CLEAN_DATASET = PROCESSED_DIR / "dataset_clean.csv"
REGRESSION_DATASET = PROCESSED_DIR / "dataset_regression.csv"
CLASSIFICATION_DATASET = PROCESSED_DIR / "dataset_classification.csv"

# --------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Variáveis do dataset
# --------------------------------------------------------------------------
ID_COLUMN = "Person_ID"

REGRESSION_TARGET = "Burnout_Score"
CLASSIFICATION_TARGET = "Burnout_Risk"

# Classes do alvo de classificação (ordem natural de severidade)
RISK_CLASSES = ["Low", "Moderate", "High"]
RISK_ENCODING = {label: idx for idx, label in enumerate(RISK_CLASSES)}

# --------------------------------------------------------------------------
# Codificação
#
# Variáveis ORDINAIS: possuem ordem natural -> OrdinalEncoder.
# A ordem abaixo é a ordem real de severidade/crescimento do domínio.
# --------------------------------------------------------------------------
ORDINAL_COLUMNS = {
    "Education_Level": ["High School", "Diploma", "Bachelor", "Master", "PhD"],
    "Sleep_Quality": ["Poor", "Average", "Good", "Excellent"],
    "Support_System": ["Poor", "Average", "Good", "Excellent"],
    "Stress_Level": ["Low", "Moderate", "High"],
    "Exercise_Frequency": ["Never", "Rarely", "Weekly", "Daily"],
}

# Variáveis BINÁRIAS: Yes/No -> 1/0 (mapeamento fixo, sem ordem inerente)
BINARY_COLUMNS = [
    "Alcohol_Consumption",
    "Smoking",
    "Healthy_Diet",
    "Chronic_Stress",
    "Family_History_Mental_Illness",
    "Therapy_Attendance",
]

# Variáveis NOMINAIS: sem ordem -> One-Hot Encoding
NOMINAL_COLUMNS = [
    "Gender",
    "Country",
    "Occupation",
    "Employment_Status",
    "Remote_Work",
]

# Variáveis numéricas que entram na modelagem como features
NUMERIC_FEATURES = [
    "Age",
    "Monthly_Income_USD",
    "Work_Hours_Per_Week",
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Sleep_Hours",
    "Anxiety_Score",
    "Depression_Score",
    "Mood_Score",
    "Emotional_Stability",
    "Physical_Activity_Hours",
    "Meditation_Minutes",
    "Screen_Time_Hours",
    "Social_Media_Hours",
    "Gaming_Hours",
    "Coffee_Cups_Per_Day",
    "Life_Satisfaction",
    "Productivity_Score",
    "Absenteeism_Days",
]

# --------------------------------------------------------------------------
# Data leakage
#
# Variáveis derivadas do próprio alvo (ou de um judgment clínico derivado do
# alvo). Mantidas no dataset limpo para análise, mas NUNCA como features.
# --------------------------------------------------------------------------
LEAKAGE_COLUMNS = [
    "Mental_Health_Status",
    "AI_Wellness_Recommendation",
]

# Justificativa individual por variável (usada no relatório)
LEAKAGE_JUSTIFICATION = {
    "Burnout_Risk": (
        "Variável-alvo do problema de classificação:Determina diretamente a "
        "rótulo e, portanto, constitui vazamento trivial de informação."
    ),
    "Burnout_Score": (
        "Variável-alvo do problema de regressão e resumo contínuo do mesmo "
        "conceito usado para discretizar Burnout_Risk."
    ),
    "Mental_Health_Status": (
        "Categoriza o estado de saúde mental (Critical/Needs Attention/Healthy) "
        "a partir dos mesmos sintomas que definem o escore de burnout."
    ),
    "AI_Wellness_Recommendation": (
        "Recomendação gerada por um modelo de IA treinado sobre o próprio "
        "Burnout_Score; carrega a informação do alvo dentro do texto."
    ),
    "Person_ID": (
        "Identificador único sequencial, sem valor preditivo (risco de "
        "memorização do conjunto de treino)."
    ),
}

# --------------------------------------------------------------------------
# Temas gráficos
# --------------------------------------------------------------------------
PALETTE = {
    "primary": "#1f4e79",
    "secondary": "#c55a11",
    "accent": "#2e7d32",
    "danger": "#b00020",
    "grey": "#6b6b6b",
}

FIGURE_DPI = 200
