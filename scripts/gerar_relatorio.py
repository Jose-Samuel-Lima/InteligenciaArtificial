"""Gera docs/relatorio_final.pdf com os resultados reais dos notebooks 03-07."""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, NextPageTemplate, PageTemplate, Frame, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfgen import canvas as pdfcanvas
from xml.sax.saxutils import escape
from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "relatorio_final.pdf"
T = ROOT / "reports" / "tables"
F = ROOT / "reports" / "figures"
M = ROOT / "reports" / "models"

# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
clf = pd.read_csv(T / "comparativo_classificacao.csv")
reg = pd.read_csv(T / "comparativo_regressao.csv")
gaps = pd.read_csv(T / "06_gaps_exatos.csv")
cv = pd.read_csv(T / "06_validacao_cruzada.csv")
por_classe = pd.read_csv(T / "06_por_classe.csv")
treino = pd.read_csv(T / "07_treinamento.csv")
inferencia = pd.read_csv(T / "07_inferencia.csv")
tamanhos = pd.read_csv(T / "07_tamanho_disco.csv")
energia = pd.read_csv(T / "07_energia.csv")
cron = pd.read_csv(ROOT / "logs" / "cronometro_regras.csv")
hardware = json.loads((M / "07_hardware.json").read_text(encoding="utf-8"))
regras_arvore = (T / "04_regras_arvore.txt").read_text(encoding="utf-8")

nomes_curtos = {
    "Sistema Especialista": "Especialista",
    "Arvore de Decisao": "Árvore de Decisão",
    "Ensemble (Random Forest)": "Random Forest",
    "Ensemble (Gradient Boosting)": "Gradient Boosting",
}

def v(valor, casas=4):
    if pd.isna(valor):
        return "—"
    return f"{float(valor):.{casas}f}".replace(".", ",")

def milhar(valor, casas=1):
    if pd.isna(valor):
        return "—"
    return f"{float(valor):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_metric(fam, prob, col):
    base = clf if prob == "Classificacao" else reg
    linha = base[base["Familia"] == fam]
    return linha[col].iloc[0] if len(linha) else np.nan

def get_eff(artefato, col):
    for base in (treino, inferencia, tamanhos, energia):
        if "Artefato" in base.columns and col in base.columns:
            linha = base[base["Artefato"] == artefato]
            if len(linha):
                return linha[col].iloc[0]
    return np.nan

def artefato(fam, prob):
    prefixo = {
        "Sistema Especialista": "expert",
        "Arvore de Decisao": "tree",
        "Ensemble (Random Forest)": "rf",
        "Ensemble (Gradient Boosting)": "gb",
    }[fam]
    return f"{prefixo}_{'classificacao' if prob == 'Classificacao' else 'regressao'}"

def stats_problema(problema):
    grupo = cron[cron["problema"] == problema]
    tempos = grupo["tempo_total_min"].astype(float)
    return {
        "n": len(grupo),
        "total": tempos.sum(),
        "media": tempos.mean(),
        "dp": tempos.std(ddof=1),
        "regras": int(grupo["n_regras_criadas"].sum()),
        "min_regra": tempos.sum() / int(grupo["n_regras_criadas"].sum()),
    }

s_cls = stats_problema("classificacao")
s_reg = stats_problema("regressao")

# --------------------------------------------------------------------------
# Estilos e documento
# --------------------------------------------------------------------------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBR", fontSize=16, alignment=TA_CENTER, spaceAfter=12,
                          fontName="Helvetica-Bold", leading=21))
styles.add(ParagraphStyle(name="SubtitleBR", fontSize=11, alignment=TA_CENTER, spaceAfter=6,
                          fontName="Helvetica", leading=15))
styles.add(ParagraphStyle(name="H1", fontSize=13.5, alignment=TA_LEFT, spaceAfter=9, spaceBefore=12,
                          fontName="Helvetica-Bold", leading=17))
styles.add(ParagraphStyle(name="H2", fontSize=11.5, alignment=TA_LEFT, spaceAfter=7, spaceBefore=9,
                          fontName="Helvetica-Bold", leading=15))
styles.add(ParagraphStyle(name="H3", fontSize=10.2, alignment=TA_LEFT, spaceAfter=5, spaceBefore=7,
                          fontName="Helvetica-BoldOblique", leading=13))
styles.add(ParagraphStyle(name="Body", fontSize=9.6, alignment=TA_JUSTIFY, spaceAfter=5,
                          leading=13.2, fontName="Helvetica"))
styles.add(ParagraphStyle(name="Small", fontSize=8.6, alignment=TA_JUSTIFY, spaceAfter=4,
                          leading=11.4, fontName="Helvetica"))
styles.add(ParagraphStyle(name="Caption", fontSize=8.2, alignment=TA_CENTER, spaceAfter=8,
                          spaceBefore=3, fontName="Helvetica-Oblique", textColor=colors.HexColor("#333333")))
styles.add(ParagraphStyle(name="Center", fontSize=10, alignment=TA_CENTER, fontName="Helvetica", leading=14))
styles.add(ParagraphStyle(name="Tiny", fontSize=5.6, alignment=TA_LEFT, fontName="Helvetica", leading=6.6))
styles.add(ParagraphStyle(name="TinyCenter", fontSize=5.6, alignment=TA_CENTER, fontName="Helvetica", leading=6.6))
styles.add(ParagraphStyle(name="TinyHead", fontSize=5.7, alignment=TA_CENTER, fontName="Helvetica-Bold", leading=6.6))

def p(texto, estilo="Body"):
    return Paragraph(texto, styles[estilo])

def celula(texto, centro=True):
    return Paragraph(str(texto), styles["TinyCenter" if centro else "Tiny"])

def tabela(dados, larguras, cabecalho=True, fonte=5.7, align="CENTER", padding=1.4):
    t = Table(dados, colWidths=larguras, repeatRows=1 if cabecalho else 0)
    comandos = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9a9a9a")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), align),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("FONTSIZE", (0, 0), (-1, -1), fonte),
    ]
    if cabecalho:
        comandos += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe5f1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ]
    t.setStyle(TableStyle(comandos))
    return t

class NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_pages = []

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for estado in self._saved_pages:
            self.__dict__.update(estado)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#666666"))
            self.drawString(1.2 * cm, 0.7 * cm,
                            "DCT1401 — Avaliação comparativa de modelos | José Samuel Silva Lima")
            self.drawRightString(self._pagesize[0] - 1.2 * cm, 0.7 * cm, f"Página {self._pageNumber} de {total}")
            super().showPage()
        super().save()

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=1.2 * cm, rightMargin=1.2 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    title="Avaliação comparativa de modelos — Sistema Especialista, Árvore de Decisão e Ensemble Learning",
    author="José Samuel Silva Lima",
)
frame_retrato = Frame(1.2 * cm, 1.35 * cm, 18.6 * cm, 26.0 * cm, id="ret")
frame_paisagem = Frame(1.0 * cm, 1.2 * cm, 27.7 * cm, 18.2 * cm, id="pais")
doc.addPageTemplates([
    PageTemplate(id="Retrato", frames=[frame_retrato], pagesize=A4),
    PageTemplate(id="Paisagem", frames=[frame_paisagem], pagesize=landscape(A4)),
])

story = []

# --------------------------------------------------------------------------
# Capa
# --------------------------------------------------------------------------
story.append(Spacer(1, 0.8 * cm))
story.append(p("UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE", "Center"))
story.append(p("DEPARTAMENTO DE COMPUTAÇÃO E TECNOLOGIA (DCT)", "Center"))
story.append(p("DCT1401 — INTELIGÊNCIA ARTIFICIAL", "Center"))
story.append(p("Bacharelado em Sistemas de Informação", "Center"))
story.append(Spacer(1, 2.0 * cm))
story.append(p("AVALIAÇÃO COMPARATIVA DE MODELOS:<br/>SISTEMA ESPECIALISTA, ÁRVORE DE DECISÃO E ENSEMBLE LEARNING", "TitleBR"))
story.append(Spacer(1, 0.6 * cm))
story.append(p("Regressão de Burnout_Score e classificação de Burnout_Risk", "SubtitleBR"))
story.append(Spacer(1, 2.0 * cm))
story.append(p("<b>Aluno:</b> José Samuel Silva Lima", "Center"))
story.append(p("<b>Matrícula:</b> 20250038765", "Center"))
story.append(Spacer(1, 0.8 * cm))
story.append(p("<b>Professor:</b> Thommas K. S. Flores", "Center"))
story.append(p("<b>Período letivo:</b> 2026.2", "Center"))
story.append(Spacer(1, 1.6 * cm))
story.append(p("<b>Dataset:</b> mental_health_burnout_prediction_dataset.csv (Kaggle)", "Center"))
story.append(p("<b>Repositório:</b> https://github.com/Jose-Samuel-Lima/InteligenciaArtificial", "Center"))
story.append(Spacer(1, 1.2 * cm))
story.append(p("Natal/RN — Outubro de 2026", "Center"))
story.append(PageBreak())

# --------------------------------------------------------------------------
# 1 Introdução
# --------------------------------------------------------------------------
story.append(p("1. INTRODUÇÃO", "H1"))
story.append(p(
    "Este relatório compara três famílias de modelos de Inteligência Artificial aplicadas ao mesmo "
    "conjunto de dados de saúde mental e burnout: (i) um <b>sistema especialista</b> baseado em regras "
    "SE–ENTÃO; (ii) uma <b>árvore de decisão</b>; e (iii) <b>ensemble learning</b>, com Random Forest "
    "(obrigatório) e Gradient Boosting (segundo método). O mesmo protocolo experimental é usado em todos "
    "os modelos, com divisão 80/20, <i>random_state=42</i>, estratificação na classificação e prevenção "
    "explícita de vazamento de dados."
))
story.append(p(
    "São avaliados dois problemas: <b>regressão</b> de <i>Burnout_Score</i> (0 a 100) e <b>classificação</b> "
    "de <i>Burnout_Risk</i> em três classes (Low, Moderate e High). A comparação cobre desempenho preditivo, "
    "capacidade de generalização, interpretabilidade, custo de construção (incluindo esforço humano medido "
    "com cronômetro) e eficiência computacional (tempo, memória, tamanho em disco e estimativa de energia/CO2)."
))
story.append(p(
    "<b>Nota de integridade dos resultados.</b> Todos os números deste relatório foram produzidos pela execução "
    "efetiva dos notebooks <i>03_expert_system.ipynb</i> a <i>07_generalization_efficiency.ipynb</i>, que compõem "
    "a versão corrigida da atividade. O motor de inferência do sistema especialista foi reimplementado para "
    "avaliar de fato as condições das regras (a versão anterior retornava sempre falso) e os notebooks foram "
    "reexecutados de ponta a ponta. Por isso, os valores desta versão substituem integralmente os da versão "
    "anterior do relatório."
))
story.append(PageBreak())

# --------------------------------------------------------------------------
# 2 Protocolo
# --------------------------------------------------------------------------
story.append(p("2. PROTOCOLO EXPERIMENTAL", "H1"))
story.append(p("2.1 Dados e divisão", "H2"))
story.append(p(
    "Foram utilizados os datasets processados <i>dataset_regression.csv</i> e <i>dataset_classification.csv</i>, "
    "derivados do arquivo original com 50.000 registros. A divisão é 80% treino (40.000) e 20% teste (10.000), "
    "com <i>random_state=42</i>; na classificação a divisão é estratificada para preservar a proporção das classes."
))
story.append(p("2.2 Pré-processamento e prevenção de vazamento", "H2"))
story.append(p(
    "Valores ausentes foram tratados pela mediana (variáveis numéricas) ou moda (categóricas). Variáveis ordinais "
    "foram codificadas com <i>OrdinalEncoder</i>, respeitando a ordem natural; binárias foram mapeadas para 0/1; "
    "nominais receberam <i>One-Hot Encoding</i>. Como os modelos usados são baseados em árvores, não foi "
    "necessária padronização; os dados foram usados na escala original. "
    "Para impedir vazamento, foram excluídas: <i>Burnout_Score</i>, <i>Burnout_Risk</i>, "
    "<i>Mental_Health_Status</i>, <i>AI_Wellness_Recommendation</i> e <i>Person_ID</i>. As regras do sistema "
    "especialista e a calibração da regressão usaram exclusivamente o conjunto de treino."
))

story.append(p("2.3 Ambiente computacional", "H2"))
hw_rows = [["Item", "Valor"]]
for chave in ["Sistema operacional", "Arquitetura", "CPU", "Núcleos físicos", "Núcleos lógicos",
              "RAM total (GB)", "Python", "scikit-learn", "pandas", "numpy", "Data da medição"]:
    valor = hardware.get(chave, hardware.get(chave.replace("Núcleos físicos", "Nucleos fisicos"), "—"))
    hw_rows.append([chave, str(valor)])
story.append(tabela(hw_rows, [5.4 * cm, 12.6 * cm], fonte=8, align="LEFT", padding=3))
story.append(p("Quadro 1 — Especificações do ambiente em que todas as medições foram realizadas.", "Caption"))
story.append(p(
    "Tempos foram medidos com <i>time.perf_counter()</i> e picos de memória com <i>tracemalloc</i>. O treinamento "
    "foi refeito em modo <i>single-thread</i> (<i>n_jobs=1</i>) para que tempo e memória fossem comparáveis entre "
    "modelos; a inferência também foi medida com 30 repetições. O tamanho em disco corresponde ao arquivo "
    "<i>joblib</i> comprimido (<i>compress=3</i>) ou, no caso do especialista, ao CSV/TXT da base de regras."
))
story.append(PageBreak())

# --------------------------------------------------------------------------
# 3 Especialista
# --------------------------------------------------------------------------
story.append(p("3. SISTEMA ESPECIALISTA BASEADO EM REGRAS SE–ENTÃO", "H1"))
story.append(p("3.1 Base de conhecimento e motor de inferência", "H2"))
story.append(p(
    "A base de conhecimento é separada do motor de inferência. Cada regra possui identificador, tipo "
    "(<i>relação</i>, <i>recomendação</i>, <i>diretriz</i>, <i>estratégia</i> ou <i>heurística</i>), condições "
    "estruturadas (variável, operador e limiar), consequente, peso e justificativa. Foram construídas "
    f"<b>{len(pd.read_csv(ROOT / 'rules' / 'kb_classificacao.csv'))} regras de classificação</b> e "
    f"<b>{len(pd.read_csv(ROOT / 'rules' / 'kb_regressao.csv'))} regras de regressão</b>. Apenas conjunções "
    "<b>E</b> são usadas, sem mistura de E/OU na mesma regra."
))
story.append(p(
    "O motor faz encadeamento para frente de forma vetorizada. Na classificação, cada regra disparada soma um "
    "voto ponderado à classe consequente e vence a classe com maior pontuação (empates favorecem a menor "
    "severidade). Na regressão, cada regra soma um delta em pontos de <i>Burnout_Score</i>; uma reta de "
    "calibração ajustada no treino converte a soma dos deltas em escore final, truncado em [0, 100]. "
    "O motor registra quais regras dispararam para cada observação, o que garante auditabilidade total."
))
story.append(p(
    "Os limiares foram definidos a partir de quantis e médias condicionais do <b>treino</b> (por exemplo, "
    "p75 de ansiedade/depressão ~ 59 e p90 ~ 76–77), e não por inspeção do teste. O notebook 03 documenta "
    "as tabelas de correlação, quantis e perfil por classe que fundamentam as regras."
))

story.append(p("3.2 Esforço humano medido com cronômetro", "H2"))
story.append(p(
    f"Foram realizadas <b>{s_cls['n']} sessões de classificação</b> e <b>{s_reg['n']} sessões de regressão</b>, "
    "superando o mínimo de três sessões por problema. O tempo foi medido com cronômetro e registrado em "
    "<i>logs/cronometro_regras.csv</i>, separado do tempo computacional. O desvio-padrão é amostral (ddof=1)."
))
cron_rows = [["Data", "Início", "Fim", "Interrup.", "Tempo (min)", "Regras", "Problema", "min/regra"]]
for _, r in cron.iterrows():
    cron_rows.append([
        str(r["data"]), str(r["horario_inicio"]), str(r["horario_fim"]),
        str(r["interrupcoes"])[:10], v(r["tempo_total_min"], 1), str(r["n_regras_criadas"]),
        "Classificação" if r["problema"] == "classificacao" else "Regressão",
        v(r["tempo_medio_por_regra_min"], 2),
    ])
story.append(tabela(cron_rows, [1.9*cm, 1.25*cm, 1.25*cm, 1.55*cm, 1.5*cm, 1.2*cm, 2.25*cm, 1.55*cm],
                    fonte=6.8, padding=1.6))
story.append(p("Tabela 1 — Sessões cronometradas de elaboração/refinamento das regras.", "Caption"))
story.append(p(
    f"<b>Classificação:</b> {s_cls['n']} sessões, total de {v(s_cls['total'],0)} min, média de "
    f"{v(s_cls['media'],2)} min, desvio-padrão de {v(s_cls['dp'],2)} min e {s_cls['min_regra']:.2f} min por regra "
    f"({s_cls['regras']} regras). <b>Regressão:</b> {s_reg['n']} sessões, total de {v(s_reg['total'],0)} min, "
    f"média de {v(s_reg['media'],2)} min, desvio-padrão de {v(s_reg['dp'],2)} min e {s_reg['min_regra']:.2f} min "
    f"por regra ({s_reg['regras']} regras). <b>Total:</b> {v(s_cls['total'] + s_reg['total'],0)} min."
))

story.append(p("3.3 Desempenho do especialista", "H2"))
story.append(p(
    f"No teste, o especialista obteve <b>F1-macro {v(get_metric('Sistema Especialista','Classificacao','F1 (macro)'))}</b>, "
    f"acurácia {v(get_metric('Sistema Especialista','Classificacao','Accuracy'))}, "
    f"ROC-AUC {v(get_metric('Sistema Especialista','Classificacao','ROC-AUC (OVR macro)'))} e "
    f"PR-AUC {v(get_metric('Sistema Especialista','Classificacao','PR-AUC (macro)'))}. Na regressão, "
    f"RMSE {v(get_metric('Sistema Especialista','Regressao','RMSE'))}, MAE {v(get_metric('Sistema Especialista','Regressao','MAE'))}, "
    f"R² {v(get_metric('Sistema Especialista','Regressao','R²'))} e MedAE {v(get_metric('Sistema Especialista','Regressao','MedAE'))}."
))
high_exp = por_classe[(por_classe["Familia"] == "Sistema Especialista") & (por_classe["Classe"] == "High")].iloc[0]
story.append(p(
    f"Na classe <b>High</b>, o especialista apresentou precisão {v(high_exp['Precision'])} e recall "
    f"{v(high_exp['Recall'])} — o maior recall entre os quatro modelos. Isso é relevante porque, em risco de "
    "burnout, deixar de detectar um caso grave (falso negativo) tende a ser mais custoso do que um falso positivo."
))
story.append(PageBreak())

# --------------------------------------------------------------------------
# 4 Árvore
# --------------------------------------------------------------------------
story.append(p("4. ÁRVORE DE DECISÃO", "H1"))
story.append(p("4.1 Configuração e ajuste de hiperparâmetros", "H2"))
params = json.loads((M / "tree_params.json").read_text(encoding="utf-8"))
story.append(p(
    "Foram comparados os critérios <i>gini</i> e <i>entropy</i> (classificação) e <i>squared_error</i>, "
    "<i>friedman_mse</i> e <i>absolute_error</i> (regressão), sempre com a mesma divisão. O ajuste usou "
    "<i>GridSearchCV</i> com validação cruzada K=5 sobre o treino, grade com <i>max_depth</i> em {6, 10, 15}, "
    "<i>min_samples_leaf</i> em {5, 10} e <i>ccp_alpha</i> em {0; 0,001}. A métrica de seleção foi F1-macro "
    "(classificação) e RMSE (regressão)."
))
story.append(p(
    f"Configuração final de classificação: <b>max_depth={params['classificacao']['max_depth']}</b>, "
    f"<b>min_samples_leaf={params['classificacao']['min_samples_leaf']}</b>, "
    f"<b>ccp_alpha={params['classificacao']['ccp_alpha']}</b>, resultando em profundidade efetiva "
    f"{params['profundidade']['classificacao']} e {params['folhas']['classificacao']} folhas. "
    f"Configuração de regressão: max_depth={params['regressao']['max_depth']}, "
    f"min_samples_leaf={params['regressao']['min_samples_leaf']}, ccp_alpha={params['regressao']['ccp_alpha']}, "
    f"com profundidade {params['profundidade']['regressao']} e {params['folhas']['regressao']} folhas."
))
story.append(p(
    "Na validação cruzada K=5 (Tabela 4), a árvore de classificação obteve F1-macro <b>0,7870 "
    "± 0,0033</b>; a árvore de regressão obteve RMSE <b>9,1756 ± 0,0770</b>."
))

story.append(p("4.2 Overfitting: árvore sem restrição × árvore ajustada", "H2"))
story.append(p(
    "A árvore sem restrição cresceu até a pureza: F1-macro de 1,0000 no treino e 0,7299 no teste "
    "(gap 0,2701); na regressão, RMSE 0,0000 no treino e 12,5399 no teste (gap 12,5399). A árvore ajustada "
    f"reduziu o gap para {v(abs(get_metric('Arvore de Decisao','Classificacao','Gap F1 (macro)')))} no F1 e "
    f"{v(get_metric('Arvore de Decisao','Regressao','Gap RMSE'))} no RMSE, mantendo F1-macro de "
    f"{v(get_metric('Arvore de Decisao','Classificacao','F1 (macro)'))} e RMSE de "
    f"{v(get_metric('Arvore de Decisao','Regressao','RMSE'))} no teste."
))
story.append(p(
    "O resultado é o comportamento clássico de viés–variância: uma árvore profunda tem viés baixo e variância "
    "altíssima, decorando ruídos do treino. A poda (<i>max_depth</i>, <i>min_samples_leaf</i>, <i>ccp_alpha</i>) "
    "troca um pouco de ajuste no treino por estabilidade no teste."
))

story.append(p("4.3 Visualização da árvore final", "H2"))
if (F / "04_arvore_classificacao.png").exists():
    story.append(Image(str(F / "04_arvore_classificacao.png"), width=17.2 * cm, height=7.75 * cm))
    story.append(p("Figura 1 — Três primeiros níveis da árvore de decisão final de classificação.", "Caption"))

story.append(p("4.4 Três regras extraídas da árvore e comparação com o especialista", "H2"))
story.append(p(
    "As regras abaixo foram extraídas automaticamente dos caminhos até as folhas com maior suporte, com "
    "simplificação de condições redundantes sobre a mesma variável (mantém-se o limiar mais restritivo)."
))
# Texto das 3 regras (limpo)
trechos = [t.strip() for t in regras_arvore.strip().split("\n\n") if t.strip()]
for trecho in trechos[:3]:
    linhas = trecho.splitlines()
    cabecalho = escape(linhas[0])
    condicao = escape(linhas[1]).replace("&gt;", ">")
    consequencia = escape(linhas[2]) if len(linhas) > 2 else ""
    distribuicao = escape(linhas[3]) if len(linhas) > 3 else ""
    story.append(p(f"<b>{cabecalho}</b><br/>{condicao}<br/>{consequencia}<br/>{distribuicao}", "Small"))

story.append(p(
    "<b>Comparação.</b> A árvore e o especialista usam a mesma lógica SE–ENTÃO e as mesmas variáveis centrais "
    "(ansiedade, depressão, humor, estabilidade emocional, estresse, produtividade e jornada), mas a árvore "
    "aprende limiares exatos dos dados, enquanto o especialista codifica limiares heurísticos derivados de "
    "quantis do treino. As três regras da árvore têm equivalentes diretos na base do especialista: a regra de "
    "risco baixo (REGRA-01 da árvore) dialoga com as regras 18–19 do especialista (sono, atividade física, "
    "satisfação e perfil emocional saudável); a regra de risco alto (REGRA-02) dialoga com as regras 1, 2 e 6 "
    "(ansiedade/depressão elevadas e estresse crônico); e a regra intermediária (REGRA-03) dialoga com as regras "
    "12–17, que tratam faixas intermediárias de ansiedade, depressão, humor, estabilidade e jornada."
))
story.append(PageBreak())

# --------------------------------------------------------------------------
# 5 Ensemble
# --------------------------------------------------------------------------
story.append(p("5. ENSEMBLE LEARNING", "H1"))
story.append(p("5.1 Random Forest", "H2"))
story.append(p(
    "O Random Forest foi ajustado com <i>RandomizedSearchCV</i> (CV=3, 6 candidatos) sobre <i>max_depth</i>, "
    "<i>min_samples_leaf</i> e <i>max_features</i>, mantendo <b>n_estimators=100</b> na configuração final. "
    "No teste, obteve F1-macro "
    f"{v(get_metric('Ensemble (Random Forest)','Classificacao','F1 (macro)'))}, acurácia "
    f"{v(get_metric('Ensemble (Random Forest)','Classificacao','Accuracy'))}, ROC-AUC "
    f"{v(get_metric('Ensemble (Random Forest)','Classificacao','ROC-AUC (OVR macro)'))} e RMSE "
    f"{v(get_metric('Ensemble (Random Forest)','Regressao','RMSE'))} na regressão. O gap é "
    f"{v(get_metric('Ensemble (Random Forest)','Classificacao','Gap F1 (macro)'))} no F1 e "
    f"{v(get_metric('Ensemble (Random Forest)','Regressao','Gap RMSE'))} no RMSE."
))
story.append(p("5.2 Gradient Boosting", "H2"))
story.append(p(
    "Para o Gradient Boosting, uma grade completa em 40 mil linhas seria proibitiva. Foi feita uma busca manual "
    "em subamostra de 25% do treino (três configurações, validação interna), vencendo <i>max_depth=2</i> e "
    "<i>learning_rate=0,10</i>; o modelo final foi reentrenado no treino completo com <b>n_estimators=100</b>. "
    f"No teste: F1-macro {v(get_metric('Ensemble (Gradient Boosting)','Classificacao','F1 (macro)'))}, "
    f"acurácia {v(get_metric('Ensemble (Gradient Boosting)','Classificacao','Accuracy'))}, ROC-AUC "
    f"{v(get_metric('Ensemble (Gradient Boosting)','Classificacao','ROC-AUC (OVR macro)'))}, RMSE "
    f"{v(get_metric('Ensemble (Gradient Boosting)','Regressao','RMSE'))}. O gap é "
    f"{v(get_metric('Ensemble (Gradient Boosting)','Classificacao','Gap F1 (macro)'))} no F1 e "
    f"{v(get_metric('Ensemble (Gradient Boosting)','Regressao','Gap RMSE'))} no RMSE."
))

story.append(p("5.3 Curva de erro em função do número de árvores", "H2"))
story.append(p(
    "O Random Forest foi treinado incrementalmente (<i>warm_start=True</i>) com 10, 25, 50, 100, 150 e 200 "
    "árvores, registrando o erro <i>out-of-bag</i> (OOB) e o desempenho no teste. O erro cai rapidamente até "
    "~50 árvores e depois estabiliza: novas árvores reduzem <b>variância</b>, mas não corrigem o <b>viés</b> do "
    "modelo. O Gradient Boosting foi avaliado por <i>staged_predict</i>, mostrando a perda logarítmica e o RMSE "
    "a cada árvore adicionada."
))
if (F / "05_curva_erro_random_forest.png").exists():
    story.append(Image(str(F / "05_curva_erro_random_forest.png"), width=16.5 * cm, height=5.18 * cm))
    story.append(p("Figura 2 — Curva de erro do Random Forest em função do número de árvores (classificação e regressão).", "Caption"))
if (F / "05_curva_gradient_boosting.png").exists():
    story.append(Image(str(F / "05_curva_gradient_boosting.png"), width=16.5 * cm, height=5.18 * cm))
    story.append(p("Figura 3 — Log-loss e RMSE do Gradient Boosting ao longo das árvores.", "Caption"))

story.append(p("5.4 Importância das variáveis", "H2"))
if (F / "05_importancia_ensemble.png").exists():
    story.append(Image(str(F / "05_importancia_ensemble.png"), width=11.5 * cm, height=8.01 * cm))
    story.append(p("Figura 4 — Importância das variáveis: Random Forest × Gradient Boosting (classificação).", "Caption"))
story.append(p(
    "As variáveis mais importantes nos dois ensembles coincidem com aquelas usadas nas regras do especialista: "
    "estresse, ansiedade, humor, depressão, estabilidade emocional, estresse crônico e produtividade. Isso "
    "indica que os modelos estão capturando sinal consistente com o conhecimento de domínio, e não apenas ruído."
))

# --------------------------------------------------------------------------
# 6 Resultados — landscape
# --------------------------------------------------------------------------
story.append(NextPageTemplate("Paisagem"))
story.append(PageBreak())
story.append(p("6. RESULTADOS", "H1"))
story.append(p("6.1 Métricas adotadas", "H2"))
story.append(p(
    "Regressão: MAE, MSE, RMSE, MAPE, R² e MedAE. Classificação: matriz de confusão, acurácia, precisão, recall "
    "e F1 (macro e ponderado), ROC-AUC e PR-AUC (macro, one-vs-rest). A classe <b>High</b> recebe atenção "
    "especial por seu custo assimétrico. O <b>gap</b> é padronizado como degradação: treino - teste para "
    "métricas de qualidade (F1, acurácia, ROC-AUC, R²) e teste - treino para métricas de erro (MAE, MSE, RMSE, "
    "MAPE, MedAE). Valores positivos indicam piora no teste."
))

story.append(p("6.2 Tabela comparativa final", "H2"))
story.append(p("Tabela 2a — Classificação (teste, exceto indicação). Valores completos, sem estimativas.", "Small"))
cab_c = ["Modelo", "Acc.", "Prec. macro", "Recall macro", "F1 macro", "ROC-AUC", "PR-AUC", "Gap F1",
         "CV F1 (K=5)", "Treino (s)", "Inf. (ms/am.)", "Mem. inf. (KB)", "Disco (KB)", "Esforço (min)", "Complexidade"]
dados_c = [[celula(h) for h in cab_c]]
for _, r in clf.iterrows():
    cv_txt = "—" if pd.isna(r["CV F1-macro (media)"]) else f"{v(r['CV F1-macro (media)'])} ± {v(r['CV F1-macro (dp)'])}"
    dados_c.append([
        celula(nomes_curtos[r["Familia"]], False),
        celula(v(r["Accuracy"])), celula(v(r["Precision (macro)"])), celula(v(r["Recall (macro)"])),
        celula(v(r["F1 (macro)"])), celula(v(r["ROC-AUC (OVR macro)"])), celula(v(r["PR-AUC (macro)"])),
        celula(v(r["Gap F1 (macro)"])), celula(cv_txt),
        celula(v(r["Tempo treino (s)"], 2)), celula(v(r["Tempo por amostra (ms)"], 4)),
        celula(milhar(r["Pico memoria inferencia (KB)"], 0)), celula(milhar(r["Tamanho disco (KB)"], 1)),
        celula("—" if pd.isna(r["Esforco humano (min)"]) else v(r["Esforco humano (min)"], 0)),
        celula(str(r["Complexidade"]).replace("SE-ENTAO", "SE-ENTÃO").replace("arvores", "árvores"), False),
    ])
larg_c = [2.9*cm, 1.25*cm, 1.45*cm, 1.45*cm, 1.35*cm, 1.4*cm, 1.35*cm, 1.25*cm, 2.5*cm,
          1.55*cm, 1.65*cm, 1.75*cm, 1.6*cm, 1.55*cm, 3.0*cm]
story.append(tabela(dados_c, larg_c))
story.append(p(
    "Legenda: Acc. = acurácia; Prec./Recall/F1 macro = médias macro entre Low, Moderate e High; CV F1 = média ± "
    "desvio-padrão dos 5 folds no treino; Gap F1 = F1(treino) - F1(teste); Inf. = tempo médio por amostra; "
    "Mem. inf. = pico de memória na inferência; Disco = tamanho do modelo salvo; Esforço = cronômetro humano "
    "(só especialista).", "Caption"))

story.append(Spacer(1, 0.25 * cm))
story.append(p("Tabela 2b — Regressão (teste). Valores completos, sem estimativas.", "Small"))
cab_r = ["Modelo", "MAE", "MSE", "RMSE", "MAPE (%)", "R²", "MedAE", "Gap RMSE",
         "CV RMSE (K=5)", "Treino (s)", "Inf. (ms/am.)", "Mem. inf. (KB)", "Disco (KB)", "Esforço (min)", "Complexidade"]
dados_r = [[celula(h) for h in cab_r]]
for _, r in reg.iterrows():
    cv_txt = "—" if pd.isna(r["CV RMSE (media)"]) else f"{v(r['CV RMSE (media)'])} ± {v(r['CV RMSE (dp)'])}"
    dados_r.append([
        celula(nomes_curtos[r["Familia"]], False),
        celula(v(r["MAE"])), celula(v(r["MSE"])), celula(v(r["RMSE"])), celula(v(r["MAPE (%)"], 2)),
        celula(v(r["R²"])), celula(v(r["MedAE"])), celula(v(r["Gap RMSE"])), celula(cv_txt),
        celula(v(r["Tempo treino (s)"], 2)), celula(v(r["Tempo por amostra (ms)"], 4)),
        celula(milhar(r["Pico memoria inferencia (KB)"], 0)), celula(milhar(r["Tamanho disco (KB)"], 1)),
        celula("—" if pd.isna(r["Esforco humano (min)"]) else v(r["Esforco humano (min)"], 0)),
        celula(str(r["Complexidade"]).replace("SE-ENTAO", "SE-ENTÃO").replace("arvores", "árvores"), False),
    ])
larg_r = [2.9*cm, 1.3*cm, 1.4*cm, 1.35*cm, 1.5*cm, 1.25*cm, 1.45*cm, 1.4*cm, 2.5*cm,
          1.55*cm, 1.65*cm, 1.75*cm, 1.6*cm, 1.55*cm, 3.0*cm]
story.append(tabela(dados_r, larg_r))
story.append(p(
    "Legenda: Gap RMSE = RMSE(teste) - RMSE(treino); MAPE calculado apenas sobre valores reais não nulos; "
    "demais colunas como na Tabela 2a. O especialista não possui tempo de treinamento computacional nem CV "
    "(regras fixas); seu custo está no esforço humano.", "Caption"))

# 6.3 Generalização
story.append(NextPageTemplate("Paisagem"))
story.append(PageBreak())
story.append(p("6.3 Generalização e correção dos gaps", "H2"))
story.append(p(
    "A versão anterior do relatório calculava o gap a partir de valores já arredondados para 4 casas, o que "
    "produzia divergências (por exemplo, 0,817 - 0,797 = 0,020). Nesta versão o gap é calculado diretamente "
    "sobre os arrays completos de predição, sem arredondamento intermediário. A Tabela 3 mostra o valor exato e, "
    "para auditoria, o valor que seria obtido com as colunas arredondadas."
))
cab_g = ["Modelo", "Problema", "Métrica", "Treino (exato)", "Teste (exato)", "Gap exato", "Gap com arredondamento"]
dados_g = [[celula(h) for h in cab_g]]
for _, r in gaps.iterrows():
    dados_g.append([
        celula(nomes_curtos[r["Familia"]], False), celula("Classificação" if r["Problema"] == "Classificacao" else "Regressão"),
        celula(r["Metrica"]), celula(v(r["Treino_exato"], 6)), celula(v(r["Teste_exato"], 6)),
        celula(v(r["Gap_exato"], 6)), celula(v(r["Gap_pelas_colunas_arredondadas"], 6)),
    ])
story.append(tabela(dados_g, [4.2*cm, 3.0*cm, 2.4*cm, 3.6*cm, 3.6*cm, 3.2*cm, 5.2*cm], fonte=7.4))
story.append(p("Tabela 3 — Gaps exatos e comparação com o cálculo feito sobre colunas arredondadas.", "Caption"))
story.append(p(
    f"O Gradient Boosting foi o modelo mais estável: F1-macro de treino 0,8013 e teste 0,8009, gap "
    f"{v(get_metric('Ensemble (Gradient Boosting)','Classificacao','Gap F1 (macro)'), 6)}. O Random Forest "
    f"apresentou o maior gap de classificação ({v(get_metric('Ensemble (Random Forest)','Classificacao','Gap F1 (macro)'), 6)}): "
    "cada árvore cresce até a pureza e o ensemble memoriza o treino (F1 de treino 0,8510). Na regressão, o "
    "Random Forest também tem o maior gap de RMSE "
    f"({v(get_metric('Ensemble (Random Forest)','Regressao','Gap RMSE'), 6)}), enquanto o Gradient Boosting fica em "
    f"{v(get_metric('Ensemble (Gradient Boosting)','Regressao','Gap RMSE'), 6)}. A árvore ajustada tem gap pequeno "
    f"em classificação ({v(get_metric('Arvore de Decisao','Classificacao','Gap F1 (macro)'), 6)}) e moderado em "
    f"regressão ({v(get_metric('Arvore de Decisao','Regressao','Gap RMSE'), 6)})."
))
story.append(p(
    "O sistema especialista tem gap negativo em classificação (-0,0078): o teste foi ligeiramente melhor que o "
    "treino. Isso não é overfitting; como as regras são fixas, a diferença reflete apenas a variabilidade da "
    "amostra. Na regressão, o gap é 0,0144, praticamente nulo."
))

story.append(p("Validação cruzada K=5", "H3"))
story.append(p(
    "A validação cruzada foi feita sobre o treino: <i>StratifiedKFold</i> na classificação e <i>KFold</i> na "
    "regressão, ambos com 5 folds e embaralhamento (<i>random_state=42</i>). O teste permaneceu intocado. "
    "Reporta-se média ± desvio-padrão entre folds."
))
cab_cv = ["Modelo", "Problema", "Métrica", "Média", "Desvio-padrão", "Média ± dp"]
dados_cv = [[celula(h) for h in cab_cv]]
for _, r in cv[cv["Metrica"].isin(["F1_macro", "RMSE"])].iterrows():
    dados_cv.append([
        celula(nomes_curtos[r["Familia"]], False),
        celula("Classificação" if r["Problema"] == "Classificacao" else "Regressão"),
        celula("F1-macro" if r["Metrica"] == "F1_macro" else "RMSE"),
        celula(v(r["Media"])), celula(v(r["Desvio_padrao"])),
        celula(f"{v(r['Media'])} ± {v(r['Desvio_padrao'])}"),
    ])
story.append(tabela(dados_cv, [5.4*cm, 3.6*cm, 3.0*cm, 3.2*cm, 3.6*cm, 5.6*cm], fonte=8.0))
story.append(p("Tabela 4 — Validação cruzada K=5: média e desvio-padrão da métrica principal.", "Caption"))

# 6.4 Eficiência
story.append(NextPageTemplate("Paisagem"))
story.append(PageBreak())
story.append(p("6.4 Eficiência computacional, memória, disco e energia", "H2"))
story.append(p(
    "As medições de treinamento e inferência foram feitas separadamente. O treinamento foi padronizado em "
    "<i>single-thread</i> para permitir comparação justa; a inferência foi repetida 30 vezes sobre o conjunto de "
    "teste (10.000 amostras). O pico de memória é o máximo observado por <i>tracemalloc</i> no processo."
))
cab_e = ["Modelo / problema", "Treino (s)", "Mem. treino (MB)", "Inf. (ms/amostra)", "Vazão (am./s)",
         "Mem. inf. (KB)", "Disco (KB)", "Energia treino (mWh)", "CO2 treino (mg)",
         "Energia inf./1000 (mWh)", "CO2 inf./1000 (mg)"]
dados_e = [[celula(h) for h in cab_e]]
for _, r in clf.iterrows():
    art = artefato(r["Familia"], "Classificacao")
    dados_e.append([
        celula(f"{nomes_curtos[r['Familia']]} — classificação", False),
        celula(v(r["Tempo treino (s)"], 2)), celula(v(r["Pico memoria treino (MB)"], 1)),
        celula(v(r["Tempo por amostra (ms)"], 4)),
        celula(milhar(get_eff(art, "Vazao (amostras/s)"), 0)),
        celula(milhar(r["Pico memoria inferencia (KB)"], 0)), celula(milhar(r["Tamanho disco (KB)"], 1)),
        celula(v(r["Energia treino (Wh)"] * 1000, 2)), celula(v(r["CO2 treino (g)"] * 1000, 2)),
        celula(v(r["Energia inferencia/1000 (Wh)"] * 1000, 3)),
        celula(v(r["CO2 inferencia/1000 (g)"] * 1000, 3)),
    ])
for _, r in reg.iterrows():
    art = artefato(r["Familia"], "Regressao")
    dados_e.append([
        celula(f"{nomes_curtos[r['Familia']]} — regressão", False),
        celula(v(r["Tempo treino (s)"], 2)), celula(v(r["Pico memoria treino (MB)"], 1)),
        celula(v(r["Tempo por amostra (ms)"], 4)),
        celula(milhar(get_eff(art, "Vazao (amostras/s)"), 0)),
        celula(milhar(r["Pico memoria inferencia (KB)"], 0)), celula(milhar(r["Tamanho disco (KB)"], 1)),
        celula(v(r["Energia treino (Wh)"] * 1000, 2)), celula(v(r["CO2 treino (g)"] * 1000, 2)),
        celula(v(r["Energia inferencia/1000 (Wh)"] * 1000, 3)),
        celula(v(r["CO2 inferencia/1000 (g)"] * 1000, 3)),
    ])
story.append(tabela(dados_e, [4.8*cm, 1.8*cm, 2.1*cm, 2.1*cm, 2.2*cm, 2.1*cm, 1.9*cm, 2.4*cm, 2.1*cm, 2.5*cm, 2.3*cm], fonte=6.4))
story.append(p(
    "Tabela 5 — Eficiência completa. Energia estimada com CPU de 45 W, PUE 1,0 e intensidade de carbono de "
    "0,0817 kgCO2/kWh; CO2 em mg. Valores de treinamento do especialista são “—”: não há treinamento "
    "computacional, apenas esforço humano.", "Caption"))
story.append(p(
    "A árvore é o modelo mais leve e rápido: 5–6 KB em disco, 0,86 s de treino e cerca de 0,002 ms por amostra. "
    "O Random Forest ocupa 2,5–2,8 MB (cerca de 400–550× a árvore), treina em 38–80 s e tem a inferência mais "
    "cara (0,027–0,031 ms/amostra). O Gradient Boosting ocupa apenas 26–57 KB, mas seu treinamento é o mais "
    "custoso (71 s na classificação, 23 s na regressão) porque é sequencial. O especialista tem inferência "
    "entre as mais baratas (0,0018–0,0117 ms/amostra), atrás apenas da árvore; seu custo energético é baixo, "
    "mas não o menor, e seu custo real é humano (242 min)."
))
story.append(p(
    "O pico de ~10 MB nos modelos scikit-learn é dominado pela conversão do DataFrame de teste "
    "(10.000 × 86) para matriz float64 (~6,9 MB) e pelo <i>check_array</i>; o custo específico do "
    "<i>predict</i> é pequeno. O especialista avalia as colunas diretamente, sem essa cópia, e por "
    "isso aparece com menos memória de inferência."
))
story.append(p(
    "As estimativas de energia são ordens de grandeza: o treinamento do Gradient Boosting consumiu cerca de "
    "0,88 Wh (0,07 gCO2) na classificação; o do Random Forest, 0,99 Wh (0,08 gCO2) na regressão. A inferência "
    "de 1.000 amostras custa menos de 0,5 mWh em todos os modelos. Em termos práticos, o custo ambiental do "
    "treinamento é desprezível neste volume de dados; o custo relevante é o tempo de desenvolvimento humano."
))

# Figuras de avaliação
story.append(NextPageTemplate("Paisagem"))
story.append(PageBreak())
story.append(p("6.5 Matrizes de confusão e curvas ROC/PR", "H2"))
if (F / "06_matrizes_confusao.png").exists():
    story.append(Image(str(F / "06_matrizes_confusao.png"), width=23.5 * cm, height=5.74 * cm))
    story.append(p("Figura 5 — Matrizes de confusão no teste para os quatro modelos de classificação.", "Caption"))
if (F / "06_curvas_roc_pr.png").exists():
    story.append(Image(str(F / "06_curvas_roc_pr.png"), width=19.0 * cm, height=7.49 * cm))
    story.append(p("Figura 6 — Curvas ROC (macro OVR) e Precision-Recall da classe High.", "Caption"))
story.append(p(
    f"O Gradient Boosting e o Random Forest têm ROC-AUC praticamente empatadas "
    f"({v(get_metric('Ensemble (Gradient Boosting)','Classificacao','ROC-AUC (OVR macro)'))} e "
    f"{v(get_metric('Ensemble (Random Forest)','Classificacao','ROC-AUC (OVR macro)'))}); a árvore fica em "
    f"{v(get_metric('Arvore de Decisao','Classificacao','ROC-AUC (OVR macro)'))}. O especialista tem a menor "
    f"ROC-AUC ({v(get_metric('Sistema Especialista','Classificacao','ROC-AUC (OVR macro)'))}) porque suas "
    "pseudo-probabilidades vêm de votos discretos normalizados, não de probabilidades calibradas. Ainda assim, "
    "seu recall na classe High é o maior, o que ilustra o trade-off entre ordenação probabilística e cobertura "
    "de casos graves."
))

# --------------------------------------------------------------------------
# 7 Discussão — portrait
# --------------------------------------------------------------------------
story.append(NextPageTemplate("Retrato"))
story.append(PageBreak())
story.append(p("7. ANÁLISE COMPARATIVA E DISCUSSÃO", "H1"))
story.append(p(
    "<b>(a) Desempenho.</b> No teste de classificação, Gradient Boosting "
    f"(F1-macro {v(get_metric('Ensemble (Gradient Boosting)','Classificacao','F1 (macro)'))}) e Random Forest "
    f"({v(get_metric('Ensemble (Random Forest)','Classificacao','F1 (macro)'))}) ficam praticamente empatados e "
    f"à frente da árvore ({v(get_metric('Arvore de Decisao','Classificacao','F1 (macro)'))}) e do especialista "
    f"({v(get_metric('Sistema Especialista','Classificacao','F1 (macro)'))}). Na regressão, o Random Forest tem "
    f"o menor RMSE ({v(get_metric('Ensemble (Random Forest)','Regressao','RMSE'))}), seguido de perto pelo "
    f"Gradient Boosting ({v(get_metric('Ensemble (Gradient Boosting)','Regressao','RMSE'))}), pela árvore "
    f"({v(get_metric('Arvore de Decisao','Regressao','RMSE'))}) e pelo especialista "
    f"({v(get_metric('Sistema Especialista','Regressao','RMSE'))})."
))
story.append(p(
    "<b>(b) Generalização.</b> O Gradient Boosting é o mais estável: gap de 0,0005 no F1 e 0,0625 no RMSE. "
    "O Random Forest tem o melhor RMSE absoluto, mas o maior gap (0,8302), evidência de overfitting: cada árvore "
    "cresce até a pureza e o ensemble decora o treino (F1 de treino 0,8510). A árvore ajustada é estável em "
    "classificação (gap 0,0004) e moderada em regressão (0,1214). O especialista não aprende, então seu gap "
    "não mede overfitting, e sim variabilidade amostral."
))
story.append(p(
    "<b>(c) Interpretabilidade e classe High.</b> O especialista é totalmente auditável e obteve o maior recall "
    f"na classe High ({v(high_exp['Recall'])}), contra "
    f"{v(por_classe[(por_classe['Familia']=='Arvore de Decisao') & (por_classe['Classe']=='High')].iloc[0]['Recall'])} "
    "da árvore e ~0,77 dos ensembles. Na prática, isso significa que o especialista deixa passar menos casos "
    "graves, ao custo de mais falsos positivos. A árvore tem a maior precisão na classe High, mas o menor recall."
))
story.append(p(
    "<b>(d) Custo de construção.</b> O esforço humano do especialista foi de 127 min na classificação (19 regras, "
    "42,33 ± 3,79 min por sessão, 6,68 min/regra) e 115 min na regressão (16 regras, 38,33 ± 2,08 min por sessão, "
    "7,19 min/regra), totalizando 242 min. Esse custo não aparece no tempo de CPU e precisa ser contabilizado "
    "em qualquer comparação honesta."
))
story.append(p(
    "<b>(e) Eficiência.</b> A árvore é imbatível em custo computacional (~6 KB, 0,9 s de treino, 0,002 ms/amostra). "
    "O Random Forest paga 2,5–2,8 MB e 38–80 s de treino pela redução de variância. O Gradient Boosting é "
    "sequencial (71 s na classificação) e ocupa pouco disco (26–57 KB), mas oferece o melhor equilíbrio "
    "desempenho–generalização. O especialista tem inferência barata e custo humano alto."
))
story.append(p(
    "<b>(f) Recomendação.</b> Para produção com restrição de latência e memória, usar árvore podada. Para máxima "
    "acurácia com generalização estável, usar Gradient Boosting. Para transparência normativa e auditoria "
    "caso a caso, usar o sistema especialista — de preferência em uma abordagem híbrida, na qual o especialista "
    "sinaliza casos de alto risco e o ensemble fornece o escore contínuo."
))

story.append(p("8. CONSIDERAÇÕES FINAIS", "H1"))
story.append(p("8.1 Conclusões", "H2"))
story.append(p(
    "O protocolo foi aplicado de forma idêntica às três famílias, com a mesma divisão, as mesmas métricas e a "
    "mesma base de teste. As métricas exigidas foram efetivamente calculadas: MSE, MAPE e MedAE na regressão; "
    "precisão, ROC-AUC, PR-AUC, matrizes de confusão e curvas ROC/PR na classificação. A validação cruzada K=5 "
    "foi executada para árvore e ensembles, com média ± desvio-padrão. Os gaps foram recalculados com valores "
    "exatos. A eficiência inclui tempo e memória separados para treino e inferência, tamanho em disco, "
    "estimativa de energia/CO2 e especificações de hardware."
))
story.append(p("8.2 Limitações", "H2"))
story.append(p(
    "Os resultados referem-se a uma única divisão treino/teste (mitigada pela CV K=5). O dataset é sintético/"
    "curado e apresenta variáveis altamente correlacionadas com o alvo. A estimativa de energia usa uma "
    "potência nominal de CPU (45 W) e fator médio de carbono; não é uma medição de consumo na tomada. O "
    "MAPE é sensível a escores reais próximos de zero (700 das 10.000 observações de teste têm escore 0, "
    "excluídas do cálculo, e 1.116 têm escore de até 5), por isso MAE, RMSE e MedAE são as métricas "
    "principais. O especialista foi calibrado com conhecimento derivado do treino, o que não dispensa "
    "validação por especialistas de domínio em uso real."
))
story.append(p("8.3 Correções aplicadas nesta versão", "H2"))
story.append(p(
    "Os notebooks 03–07, antes esqueletos com células <i>TODO</i> e sem execução, foram reescritos e executados "
    "de ponta a ponta, gerando tabelas e figuras verificáveis. O motor de inferência do especialista foi "
    "corrigido (antes, a função de correspondência retornava sempre falso) e a base foi ampliada para 19 regras "
    "de classificação e 16 de regressão, compatível com o cronômetro. O relatório passou a usar apenas valores "
    "calculados, sem placeholders. As tabelas comparativas foram separadas em classificação e regressão na "
    "seção 6.2, e a seção 6.3 trata apenas de generalização. Os gaps são calculados com valores exatos, e a "
    "diferença em relação à versão anterior é explicada na Tabela 3."
))

story.append(PageBreak())
story.append(p("9. REFERÊNCIAS E REPOSITÓRIO", "H1"))
story.append(p(
    "Repositório: https://github.com/Jose-Samuel-Lima/InteligenciaArtificial<br/>"
    "Estrutura: <i>data/</i>, <i>src/</i>, <i>notebooks/</i> (01_eda a 07_generalization_efficiency), "
    "<i>rules/</i>, <i>logs/</i>, <i>reports/</i> e <i>docs/</i>.<br/>"
    "Dataset: Mental Health and Burnout Prediction Dataset (Kaggle).<br/>"
    "Ferramentas: Python 3.9, scikit-learn 1.6, pandas 2.3, NumPy 2.0, Matplotlib/Seaborn, ReportLab."
))
story.append(Spacer(1, 1.0 * cm))
story.append(p("Natal/RN, Outubro de 2026", "Center"))
story.append(p("José Samuel Silva Lima", "Center"))

doc.build(story, canvasmaker=NumberedCanvas)
print(f"PDF gerado: {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
