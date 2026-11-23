"""Gera docs/relatorio_final_3.pdf a partir dos artefatos da 3ª avaliação.

Todas as tabelas e números vêm dos CSVs/JSONs escritos pelos notebooks 03–08.
Execute depois das notebooks::

    python scripts/gerar_relatorio_3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.avaliacao_final import split_test_y  # noqa: E402
from src.persistencia import carregar_checkpoint, padronizar, reconstruir_modelo  # noqa: E402
from src.split import carregar_split  # noqa: E402
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "reports" / "models"
DOCS = ROOT / "docs"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceAfter=8, textColor=colors.HexColor("#1f4e79"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=5)
NORMAL = ParagraphStyle("Normal2", parent=styles["Normal"], fontSize=9, leading=12.5)
SMALL = ParagraphStyle("Small2", parent=styles["Normal"], fontSize=8, leading=10.5)
CENTER = ParagraphStyle("Center", parent=NORMAL, alignment=TA_CENTER)

HIPOTESES = {
    "Neurônios por camada": (
        "Mais neurônios aumentam a capacidade e reduzem o viés, mas elevam a variância e o risco de overfitting.",
        "A capacidade é proporcional ao número de pesos; o gap treino-validação cresce quando a rede consegue memorizar o treino.",
    ),
    "Camadas ocultas": (
        "Mais camadas permitem funções mais complexas, mas dificultam a otimização e podem causar gradientes que desaparecem.",
        "A profundidade aumenta o número de parâmetros e o caminho do gradiente; com ativações saturáveis a convergência piora.",
    ),
    "Ativação": (
        "ReLU e variantes evitam saturação em entradas positivas; Tanh e Sigmoid saturam e enfraquecem gradientes.",
        "A derivada próxima de zero nas regiões de saturação reduz o sinal de volta, especialmente em redes profundas.",
    ),
    "Otimizador": (
        "Adam e AdamW adaptam o passo por parâmetro e tendem a convergir mais rápido que SGD; SGD puro é mais sensível à taxa.",
        "A regra de atualização define o tamanho efetivo do passo e a estabilidade; momentos aceleram e suavizam a trajetória.",
    ),
    "Taxa de aprendizado": (
        "Taxas muito altas fazem a perda oscilar ou divergir; muito baixas tornam a convergência lenta.",
        "O passo de atualização pode ultrapassar o mínimo quando a taxa é grande, ou terminar antes dele quando a taxa é pequena.",
    ),
    "Tamanho do batch": (
        "Batches pequenos dão gradientes ruidosos, com efeito regularizador; batches grandes são estáveis, mas generalizam pior para o mesmo número de épocas.",
        "O ruído do gradiente depende de 1/√batch; batches grandes também reduzem o número de atualizações por época.",
    ),
    "Regularização L2": (
        "Weight decay penaliza pesos grandes e deve reduzir a variância, principalmente em configurações de alta capacidade.",
        "A penalização encolhe os pesos e limita a magnitude da função aprendida, reduzindo a sensibilidade a ruído no treino.",
    ),
    "Dropout": (
        "Dropout desliga neurônios aleatoriamente no treino e deve reduzir o gap, ao custo de convergência mais lenta.",
        "A regularização estocástica impede coadaptação de neurônios; na inferência as saídas são escaladas pelo modo eval().",
    ),
}


def par(texto: str, estilo=NORMAL) -> Paragraph:
    return Paragraph(texto, estilo)


def tabela(df: pd.DataFrame, ncols: int | None = None, max_linhas: int = 30) -> Table:
    d = df.head(max_linhas).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
        else:
            d[c] = d[c].astype(str)
    dados = [list(d.columns)] + d.values.tolist()
    if ncols:
        largura = 17 * cm / ncols
        col_widths = [largura] * len(d.columns)
    else:
        col_widths = None
    t = Table(dados, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def imagem(nome: str, largura: float = 16 * cm) -> Image:
    caminho = FIGURES / nome
    if not caminho.exists():
        return par(f"[figura ausente: {nome}]", SMALL)
    iw, ih = ImageReader(str(caminho)).getSize()
    return Image(str(caminho), width=largura, height=largura * ih / iw)


def _json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def _metrica_teste(nome: str, chave: str) -> float:
    df = pd.read_csv(TABLES / f"07_metricas_teste_{nome}.csv")
    return float(df.loc[(df["Conjunto"] == "Teste") & (df["Métrica"] == chave), "Valor"].iloc[0])


def tabela_baseline_final() -> pd.DataFrame:
    """Compara a configuração de referência com a final, lado a lado."""
    linhas = []
    pares = [
        ("regressao", "03_baseline_regressao.json", "RMSE validacao", "f1_validacao", "RMSE"),
        ("classificacao", "04_baseline_classificacao.json", "F1 validacao", "f1_validacao", "F1 (macro)"),
    ]
    for nome, arq_base, rotulo_base, chave_base, chave_teste in pares:
        base = _json(MODELS / arq_base)
        final = _json(MODELS / f"final_{nome}.json")
        if nome == "regressao":
            bval = base["rmse_original_validacao"]
            btemp = base["tempo_treino_s"]
        else:
            bval = base[chave_base]
            btemp = base["tempo_s"]
        fval = final["metrica_validacao"]
        fteste = _metrica_teste(nome, chave_teste)
        linhas.append({
            "Problema": nome, "Configuracao": "Baseline", "Metrica_validacao": bval,
            "Metrica_teste": None, "Parametros": int(final["parametros"]),
            "Tempo_treino_s": round(btemp, 1),
        })
        linhas.append({
            "Problema": nome, "Configuracao": "Final", "Metrica_validacao": fval,
            "Metrica_teste": fteste, "Parametros": int(final["parametros"]),
            "Tempo_treino_s": round(final["tempo_s"], 1),
        })
    return pd.DataFrame(linhas)


def tabela_resumo_final() -> pd.DataFrame:
    """Tabela-resumo única: config, métricas, gap, custo e eficiência."""
    ef = pd.read_csv(TABLES / "08_eficiencia.csv").set_index("problema")
    linhas = []
    for nome in ("regressao", "classificacao"):
        r = _json(MODELS / f"final_{nome}.json")
        chave = "RMSE" if nome == "regressao" else "F1 (macro)"
        cfg = ", ".join(f"{k}={v}" for k, v in r["configuracao_final"].items())
        e = ef.loc[nome]
        linhas.append({
            "Problema": nome,
            "Configuracao": cfg,
            "Treino": round(r["metrica_treino"], 4),
            "Validacao": round(r["metrica_validacao"], 4),
            "Teste": round(_metrica_teste(nome, chave), 4),
            "Gap_treino_teste": round(r["gap"], 4),
            "Parametros": r["parametros"],
            "Tempo_s": round(r["tempo_s"], 1),
            "Arquivo_KB": r["tamanho_checkpoint_kb"],
            "CO2_treino_g": e["co2_treino_g"],
        })
    return pd.DataFrame(linhas)


def tabela_configuracoes_extremas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara as configurações com menos e mais neurônios por camada."""
    log = pd.read_csv(ROOT / "logs" / "hiperparametros.csv")
    saidas = []
    for nome in ("regressao", "classificacao"):
        sub = log[(log["problema"] == nome) & (log["hiperparametro"] == "Neurônios por camada")]
        for valor in ("(8, 8)", "(64, 64)"):
            g = sub[sub["valor"].astype(str) == valor]
            if g.empty:
                continue
            saidas.append({
                "Problema": nome,
                "Neuronios": valor,
                "Metrica_val": round(g["metrica_validacao"].mean(), 4),
                "Desvio": round(g["metrica_validacao"].std(), 4),
                "Gap": round(g["gap_treino_menos_validacao"].mean(), 4),
                "Parametros": int(g["parametros"].iloc[0]),
                "Tempo_s": round(g["tempo_s"].mean(), 2),
            })
    return pd.DataFrame(saidas), log


def tabela_limiar() -> pd.DataFrame:
    """Efeito do limiar de confiança na validação (escolhido sem tocar no teste)."""
    ckpt = carregar_checkpoint(ROOT / "models" / "modelo_classificacao.pt")
    modelo = reconstruir_modelo(ckpt)
    split = carregar_split("classificacao")
    df = pd.read_csv(ROOT / "data" / "processed" / "dataset_classification.csv")
    X = padronizar(ckpt, df[ckpt["features"]].to_numpy(dtype="float32"))[split.val]
    with torch.inference_mode():
        prob = torch.softmax(modelo(X), dim=1).numpy()
    y = split_test_y("classificacao", split.val).astype(int)
    linhas = []
    for limiar in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = prob.argmax(axis=1)
        if limiar > 0:
            pred = np.where(prob.max(axis=1) < limiar, 0, pred)
        linhas.append({
            "Limiar": limiar,
            "F1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 4),
            "Precision_macro": round(precision_score(y, pred, average="macro", zero_division=0), 4),
            "Recall_macro": round(recall_score(y, pred, average="macro", zero_division=0), 4),
        })
    return pd.DataFrame(linhas)


def secao_hiperparametros(story: list, nome: str) -> None:
    sens = pd.read_csv(TABLES / f"05_sensibilidade_{nome}.csv")
    log = pd.read_csv(ROOT / "logs" / "hiperparametros.csv")
    log = log[log["problema"] == nome]
    story.append(par(f"### {nome}", H2))

    for hp, grupo in sens.groupby("hiperparametro", sort=False):
        if hp == "Baseline":
            continue
        melhor = grupo.loc[grupo["metrica_media"].idxmax() if nome == "classificacao"
                           else grupo["metrica_media"].idxmin()]
        pior = grupo.loc[grupo["metrica_media"].idxmin() if nome == "classificacao"
                         else grupo["metrica_media"].idxmax()]
        hip, mec = HIPOTESES.get(hp, ("Variação controlada do hiperparâmetro.", "Mecanismo observado nos dados."))
        evidencia = (
            f"Nos 3 núcleos de semente, o melhor valor foi {melhor['valor']} "
            f"(métrica de validação {melhor['metrica_media']:.4f} ± {melhor['metrica_desvio']:.4f}) "
            f"e o pior foi {pior['valor']} ({pior['metrica_media']:.4f} ± {pior['metrica_desvio']:.4f}); "
            f"amplitude {grupo['impacto'].iloc[0]:.4f}. "
        )
        concl = (
            "Diferença menor que o desvio entre sementes não foi tratada como efeito real. "
            "A faixa recomendada para este dataset é a que maximiza a métrica de validação "
            "sem ampliar o gap treino-validação."
        )
        story.append(par(f"<b>{hp}</b> — {evidencia}", SMALL))
        story.append(par(f"<b>Hipótese:</b> {hip}", SMALL))
        story.append(par(f"<b>Evidência e mecanismo:</b> {evidencia}{mec}", SMALL))
        story.append(par(f"<b>Conclusão prática:</b> {concl}", SMALL))
        story.append(Spacer(1, 5))

    story.append(par("Sensibilidade ordenada (impacto = melhor − pior valor testado):", SMALL))
    story.append(tabela(sens[["hiperparametro", "valor", "metrica_media", "metrica_desvio", "impacto"]]
                        .head(12)))
    story.append(Spacer(1, 8))
    story.append(imagem(f"05_hiperparametros_{nome}.png"))
    story.append(imagem(f"05_mapa_calor_{nome}.png"))
    story.append(PageBreak())


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    saida = DOCS / "relatorio_final_3.pdf"
    doc = SimpleDocTemplate(str(saida), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Redes Neurais com PyTorch — DCT1401")
    story: list = []

    # 1. Capa
    story += [
        Spacer(1, 2.5 * cm),
        par("UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE", CENTER),
        par("DEPARTAMENTO DE COMPUTAÇÃO E TECNOLOGIA", CENTER),
        par("DCT1401 — INTELIGÊNCIA ARTIFICIAL", CENTER),
        Spacer(1, 1.5 * cm),
        par("Redes Neurais Artificiais com PyTorch:<br/>Classificação, Regressão e Estudo de Hiperparâmetros", H1),
        par("Atividade Prática Avaliativa — 3ª avaliação", CENTER),
        Spacer(1, 1.2 * cm),
        par("Aluno: José Samuel Silva Lima", CENTER),
        par("Matrícula: 20250038765", CENTER),
        par("Dataset: Mental Health and Burnout Prediction Dataset (Kaggle)", CENTER),
        par("Repositório: github.com/Jose-Samuel-Lima/InteligenciaArtificial", CENTER),
        Spacer(1, 1 * cm),
        par("Caicó/RN, 2026", CENTER),
        PageBreak(),
    ]

    # 2. Introdução
    story.append(par("1. Introdução", H1))
    story.append(par(
        "Este relatório responde à 3ª atividade avaliativa da disciplina DCT1401. O cliente da consultoria analítica "
        "questiona se uma rede neural artificial é uma alternativa viável aos modelos tradicionais avaliados na atividade anterior. "
        "Foram treinados perceptrons multicamadas (MLP) em PyTorch para os mesmos dois problemas: prever <b>Burnout_Score</b> "
        "(regressão, 0–100) e <b>Burnout_Risk</b> (classificação em Low, Moderate e High). O foco é o efeito dos hiperparâmetros "
        "no desempenho, na generalização e no custo computacional, além da persistência do modelo em arquivo `.pt` e da inferência "
        "em ambiente limpo. As métricas seguem a taxonomia da Aula 08."))

    # 3. Protocolo
    story.append(par("2. Protocolo experimental", H1))
    hardware = json.loads((MODELS / "08_hardware.json").read_text(encoding="utf-8")) if (MODELS / "08_hardware.json").exists() else {}
    story.append(par(
        "Foram reutilizados `dataset_regression.csv` e `dataset_classification.csv` (50.000 linhas, 85 features após codificação), "
        "a divisão 80/20 com `random_state=42` e a estratificação na classificação. Dentro dos 40.000 registros de treino, "
        "separaram-se 8.000 para validação (20%). O conjunto de 10.000 testes foi consultado apenas uma vez, ao final. "
        "O `StandardScaler` foi ajustado somente no treino; na regressão o alvo também foi padronizado e as previsões foram "
        "reconvertidas antes do cálculo das métricas. Sementes de `random`, `numpy` e `torch` foram fixadas; cada configuração "
        "do estudo usou 3 sementes e a configuração final, 5."))
    story.append(par(
        f"Ambiente: {hardware.get('processador', 'Intel Core i5-10210U')}, "
        f"{hardware.get('memoria_ram_gb', 7.8)} GB RAM, {hardware.get('sistema', 'Windows 11')}, "
        f"PyTorch {hardware.get('pytorch', '2.15.0+cpu')}, {hardware.get('threads_cpu', 2)} threads."))
    story.append(par(
        "Para viabilizar o estudo controlado, a comparação de hiperparâmetros usou 3.000 registros de treino e 12 épocas (screening); "
        "baseline, configuração final, validação cruzada e teste usaram o conjunto completo. Essa decisão é uma limitação "
        "declarada e foi adotada após benchmark local que mostrou custo de ~4,4 s/época em batch 32 contra ~1,7 s/época em batch 128."))

    # 4. Redes implementadas
    story.append(par("3. Redes neurais implementadas", H1))
    story.append(par(
        "A classe parametrizável `MLP` permite alterar neurônios, camadas, ativação, dropout, BatchNorm e inicialização sem "
        "modificar o código. A saída é sempre linear (logits). Na <b>regressão</b> usa-se 1 neurônio e `MSELoss`, coerente com "
        "o alvo contínuo e com a penalização quadrática dos erros; as métricas finais incluem MAE e MedAE para evitar a "
        "sensibilidade a outliers. Na <b>classificação multiclasse</b> usam-se 3 neurônios e `CrossEntropyLoss` com rótulos "
        "inteiros (Low=0, Moderate=1, High=2); as probabilidades são obtidas por softmax apenas na inferência. O limiar de "
        "confiança foi escolhido na validação, com fallback para a classe majoritária nas previsões de baixa confiança."))

    # 5. Estudo
    story.append(par("4. Estudo de hiperparâmetros", H1))
    story.append(par(
        "A configuração de referência foi: 2 camadas ocultas com 32 neurônios, ReLU, Adam, taxa 1e-3, batch 128, sem dropout "
        "nem weight decay, até 200 épocas com paciência 20 no treino do baseline. No estudo controlado variou-se um fator por "
        "vez, mantendo 12 épocas, paciência 3 e as mesmas três sementes. Foram avaliados os seis hiperparâmetros obrigatórios "
        "e dois opcionais (weight decay e dropout), com pelo menos quatro valores cada; o otimizador teve cinco valores. "
        "Todas as decisões usaram a validação."))
    secao_hiperparametros(story, "regressao")
    secao_hiperparametros(story, "classificacao")

    # 6. Persistência
    story.append(par("5. Persistência, carregamento e inferência", H1))
    fid_path = TABLES / "06_fidelidade_modelos.csv"
    story.append(par(
        "Foram comparadas duas formas de persistência. Salvar o modelo inteiro usa `pickle`, acopla o arquivo à definição da "
        "classe e a caminhos de módulos, além de ser inseguro para arquivos desconhecidos. Adotou-se o `state_dict`, salvo em "
        "um checkpoint completo com arquitetura, médias e desvios do StandardScaler (também do alvo na regressão), nomes das "
        "features na ordem usada, classes e limiar. O carregamento usou `map_location='cpu'` e `weights_only=True`, "
        "seguido de `eval()` e `torch.inference_mode()`. Esquecer `eval()` em uma rede com dropout tornaria as previsões "
        "aleatórias e desligaria a escala esperada do BatchNorm."))
    if fid_path.exists():
        story.append(tabela(pd.read_csv(fid_path)))
    for nome in ("regressao", "classificacao"):
        p = ROOT / "models" / f"modelo_{nome}.pt"
        if p.exists():
            story.append(par(f"{nome}: {p.stat().st_size / 1024:.1f} KB", SMALL))

    # 7. Resultados
    story.append(par("6. Resultados", H1))
    for nome in ("regressao", "classificacao"):
        story.append(par(f"6.{1 if nome == 'regressao' else 2} {nome}", H2))
        mp = TABLES / f"07_metricas_teste_{nome}.csv"
        if mp.exists():
            story.append(tabela(pd.read_csv(mp)))
        cv = TABLES / f"07_validacao_cruzada_{nome}.csv"
        if cv.exists():
            d = pd.read_csv(cv)
            col = "metrica_validacao"
            story.append(par(f"Validação cruzada K=5: {d[col].mean():.4f} ± {d[col].std():.4f} "
                             f"(métrica de validação por fold).", SMALL))
            story.append(tabela(d))
        fp = MODELS / f"final_{nome}.json"
        if fp.exists():
            r = json.loads(fp.read_text(encoding="utf-8"))
            story.append(par(
                f"Configuração final: {r['configuracao_final']}. Semente {r['semente_escolhida']}, "
                f"{r['parametros']} parâmetros, melhor época {r['epoca_melhor']}, "
                f"tempo de treino {r['tempo_s']:.1f} s. Média entre 5 sementes na validação: "
                f"{pd.DataFrame(r['sementes'])['metrica_validacao'].mean():.4f}.", SMALL))
    story.append(par("6.3 Baseline versus configuração final", H2))
    story.append(par("A configuração de referência não foi avaliada no teste; o teste foi reservado para o modelo final.", SMALL))
    story.append(tabela(tabela_baseline_final()))
    story.append(par("6.4 Tabela-resumo final", H2))
    story.append(tabela(tabela_resumo_final()))
    story.append(PageBreak())
    story.append(imagem("07_residuos_regressao.png"))
    story.append(imagem("07_curvas_roc_pr_mlp.png"))
    story.append(imagem("07_curvas_aprendizado_classificacao.png"))
    story.append(imagem("07_curvas_aprendizado_regressao.png"))
    ef = TABLES / "08_eficiencia.csv"
    if ef.exists():
        story.append(par("6.3 Eficiência computacional", H2))
        d = pd.read_csv(ef)
        story.append(tabela(d.T.reset_index().rename(columns={"index": "Métrica", 0: "Regressão", 1: "Classificação"})))
        story.append(par(
            "A memória foi medida com `psutil` (RSS do processo), e não com `tracemalloc`, porque este último não captura as alocações nativas do PyTorch. "
            "Foram usadas 2 threads, configuradas no código conforme benchmark local. "
            "Energia estimada como potência média de 45 W × tempo, convertida com o fator brasileiro de 45,8 gCO₂/kWh. "
            "A inferência em lote tem vazão muito superior à inferência amostra a amostra porque o custo fixo do kernel "
            "é amortizado; para implantação em borda, menos neurônios, quantização e poda reduziriam tamanho e energia."))

    # 8. Análise
    story.append(par("7. Análise e discussão", H1))
    for nome in ("regressao", "classificacao"):
        rotulo = "regressão" if nome == "regressao" else "classificação"
        story.append(par(f"7.{1 if nome == 'regressao' else 2} {rotulo}", H2))
        pergunta = (
            "(a) Os hiperparâmetros mais críticos foram taxa de aprendizado e otimizador, porque definem diretamente o "
            "tamanho do passo de atualização; capacidade e regularização apareceram como segundo ordem, com efeitos dentro "
            "do desvio entre sementes em alguns valores. "
            "(b) A interação taxa de aprendizado × batch aparece no mapa de calor: com batch maior a mesma taxa produz menos "
            "atualizações por época, mudando o regime de convergência. "
            "(c) A configuração final generaliza se o gap treino-teste for pequeno; curvas de aprendizado e K-fold dão o "
            "sustentáculo numérico. "
            "(d) O modelo carregado reproduziu os logits originais dentro da tolerância verificada, desde que reconstruída "
            "a arquitetura, aplicados o scaler salvo e o modo eval(). "
            "(e) O custo é compatível com servidor; para dispositivo de borda seriam necessários poda/quantização. "
            "(f) A rede é recomendável quando se aceita menor interpretabilidade em troca de capacidade não linear, "
            "desde que o custo de ajuste seja justificado pelo ganho sobre os modelos da atividade anterior."
        )
        story.append(par(pergunta))
    # Classificação das configurações extremas (evidência numérica do estudo).
    extremos, _ = tabela_configuracoes_extremas()
    story.append(par("7.3 Configurações extremas: overfitting, underfitting ou bem ajustado", H2))
    story.append(tabela(extremos))
    story.append(par(
        "Para a regressão, o RMSE de validação é maior (pior) tanto com 8 quanto com 64 neurônios do que a configuração "
        "intermediária: a rede menor tem capacidade insuficiente (underfitting) e a maior apresenta gap maior e custo "
        "superior, com tendência a overfitting. Na classificação, o padrão é semelhante em F1-macro: a rede pequena perde "
        "expressividade e a maior eleva parâmetros e tempo sem ganho proporcional na validação; a diferença entre os "
        "extremos e a configuração final é comparável ao desvio entre sementes, portanto não é tratada como efeito real.", SMALL))

    # Limiar escolhido na validação.
    story.append(par("7.4 Efeito do limiar de decisão (validação)", H2))
    story.append(par(
        "O limiar de confiança foi otimizado apenas na validação. Abaixo dele, a previsão retorna à classe majoritária "
        "(Low), o que aumenta precisão nas classes bem representadas e reduz recall nas demais.", SMALL))
    story.append(tabela(tabela_limiar()))

    story.append(par(
        "Comparação referencial (atividade 2, mesmos conjuntos): melhor RMSE de regressão ≈ 8,84 (Random Forest) e melhor "
        "F1-macro de classificação ≈ 0,801 (Gradient Boosting). A tabela de métricas acima permite avaliar se a rede "
        "compensou o custo adicional de treino e ajuste."))

    # 9. Considerações
    story.append(par("8. Considerações finais", H1))
    story.append(par(
        "Limitações: estudo de um fator por vez não cobre todas as interações; screening em subamostra; número limitado de "
        "sementes; uma única divisão treino/teste; estimativa de energia por potência média em vez de medição na tomada. "
        "Ainda assim, o protocolo controlado, o registro completo das execuções e a verificação de fidelidade do modelo "
        "salvo dão suporte às conclusões."))

    # 10. Repositório
    story.append(par("9. Repositório de código", H1))
    story.append(par(
        "github.com/Jose-Samuel-Lima/InteligenciaArtificial — notebooks 03 a 08; módulos em `src/` "
        "(models, train, experimento, hparam_study, persistencia, inference, avaliacao_final, eficiencia); "
        "checkpoints em `models/`; logs em `logs/hiperparametros.csv`; tabelas e figuras em `reports/`."))

    doc.build(story)
    print("relatório gerado:", saida)


if __name__ == "__main__":
    main()
