"""
gerar_graficos.py — Gráficos comparativos dos experimentos E4 e E5.

Lê os CSVs de resultados (reports/resultados/) e gera figuras em
reports/figures/comparativos/. NÃO treina modelos e NÃO altera resultados.

Uso, na raiz do repositório:
    python src/gerar_graficos.py
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # grava arquivos sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

# ─────────────────────────── CONFIGURAÇÃO ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = BASE_DIR / "reports" / "resultados"
FIGURAS_DIR = BASE_DIR / "reports" / "figures" / "comparativos"  # não toca nos gráficos SHAP

ARQUIVOS = {
    "E4": RESULTADOS_DIR / "equiparidade_20261009_112030_E4.csv",
    "E5": RESULTADOS_DIR / "equiparidade_20261009_102402_E5.csv",
}

# Valores que o enunciado da etapa define como esperados. Se os experimentos
# forem refeitos, atualize aqui; a validação acusa qualquer divergência.
ESPERADO = {
    "semente": 42,
    "hash_amostra": "ee971fa098f9",
    "n_treino": 50000,
    "n_teste": 229114,
    "n_atributos": {"E4": 51, "E5": 45},
}

MODELOS = ["Random Forest", "XGBoost", "SVM"]
EXPERIMENTOS = ["E4", "E5"]
NOMES_EXP = {
    "E4": "E4 · todos os atributos",
    "E5": "E5 · sem HOSPITAL, UTI e SUPORT_VEN",
}
METRICAS = {"acuracia": "Acurácia", "precisao": "Precisão",
            "recall": "Recall", "f1": "F1-score"}
ORDEM_B = ["f1", "acuracia", "precisao", "recall"]  # F1 e acurácia primeiro

# Paletas consistentes: cor = algoritmo (A, C, D, F); cor = experimento (B, E)
COR_MODELO = {"Random Forest": "#0072B2", "XGBoost": "#E69F00", "SVM": "#009E73"}
COR_EXP = {"E4": "#4D4D4D", "E5": "#CC79A7"}
MARCADOR_EXP = {"E4": "o", "E5": "s"}
SLUG_MODELO = {"Random Forest": "random_forest", "XGBoost": "xgboost", "SVM": "svm"}
ROTULOS_CLASSES = ["Cura (0)", "Óbito (1)"]  # ordem do sklearn: 0, 1

DPI = 300
MOSTRAR_RODAPE = False  # False remove o rodapé das figuras (útil para o artigo)

COLUNAS_OBRIGATORIAS = [
    "experimento", "modelo", "semente", "hash_amostra", "n_treino", "n_teste",
    "n_atributos", "tempo_treino_s", "tempo_predicao_s", "acuracia", "precisao",
    "recall", "f1", "matriz_confusao", "status",
]

GERADOS = []  # nomes dos arquivos gerados, para o resumo final


class ErroValidacao(Exception):
    """Problema nos arquivos de entrada (ausentes, incompletos ou inconsistentes)."""


# ───────────────────── LEITURA E VALIDAÇÃO DOS CSVs ─────────────────────
def carregar_dados() -> pd.DataFrame:
    """Lê os CSVs de E4 e E5 e junta numa única tabela."""
    partes = []
    for exp, caminho in ARQUIVOS.items():
        if not caminho.exists():
            raise ErroValidacao(
                f"Arquivo não encontrado: {caminho}\n"
                f"  Confira o nome em ARQUIVOS (topo do script) e a pasta reports/resultados/."
            )
        df = pd.read_csv(caminho, dtype={"hash_amostra": str})
        faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
        if faltando:
            raise ErroValidacao(f"{caminho.name}: colunas ausentes: {faltando}")
        if not (df["experimento"] == exp).all():
            raise ErroValidacao(
                f"{caminho.name}: esperado experimento '{exp}', "
                f"encontrado {sorted(df['experimento'].unique())}"
            )
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def validar(df: pd.DataFrame) -> None:
    """Confere as 6 execuções, o status e a consistência de amostra/semente/tamanhos."""
    problemas = []

    esperados = {(e, m) for e in EXPERIMENTOS for m in MODELOS}
    encontrados = set(zip(df["experimento"], df["modelo"]))
    if len(df) != len(esperados) or encontrados != esperados:
        problemas.append(
            f"Esperadas 6 execuções (3 modelos × 2 experimentos); encontradas {len(df)}. "
            f"Faltando: {sorted(esperados - encontrados)}; inesperadas: {sorted(encontrados - esperados)}."
        )
    if df.duplicated(["experimento", "modelo"]).any():
        problemas.append("Há execuções duplicadas para o mesmo experimento e modelo.")

    for _, l in df[df["status"] != "ok"].iterrows():
        problemas.append(f"{l['experimento']} / {l['modelo']}: status = '{l['status']}' (esperado 'ok').")

    for coluna in ("semente", "hash_amostra", "n_treino", "n_teste"):
        achados = set(df[coluna].astype(str))
        if achados != {str(ESPERADO[coluna])}:
            problemas.append(f"Coluna '{coluna}': esperado {ESPERADO[coluna]}, encontrado {sorted(achados)}.")

    for exp, esperado in ESPERADO["n_atributos"].items():
        achados = set(df.loc[df["experimento"] == exp, "n_atributos"].astype(str))
        if achados != {str(esperado)}:
            problemas.append(f"{exp}: n_atributos esperado {esperado}, encontrado {sorted(achados)}.")

    for coluna in list(METRICAS) + ["tempo_treino_s", "tempo_predicao_s"]:
        if not pd.api.types.is_numeric_dtype(df[coluna]) or df[coluna].isna().any():
            problemas.append(f"Coluna '{coluna}' tem valores ausentes ou não numéricos.")
    for coluna in METRICAS:
        if pd.api.types.is_numeric_dtype(df[coluna]) and not df[coluna].between(0, 1).all():
            problemas.append(f"Coluna '{coluna}' tem valores fora do intervalo [0, 1].")

    if problemas:
        raise ErroValidacao("\n".join(f"  - {p}" for p in problemas))


def extrair_matrizes(df: pd.DataFrame) -> dict:
    """Reconstrói as matrizes 2x2 a partir do JSON gravado em 'matriz_confusao'.
    Formato: [[TN, FP], [FN, TP]], com classes na ordem 0 = cura, 1 = óbito."""
    matrizes = {}
    for _, l in df.iterrows():
        try:
            m = np.array(json.loads(l["matriz_confusao"]), dtype=int)
        except (TypeError, ValueError) as erro:
            raise ErroValidacao(f"{l['experimento']} / {l['modelo']}: matriz_confusao ilegível ({erro}).")
        if m.shape != (2, 2):
            raise ErroValidacao(f"{l['experimento']} / {l['modelo']}: matriz com formato {m.shape}, esperado (2, 2).")
        matrizes[(l["experimento"], l["modelo"])] = m
    return matrizes


def validar_matrizes(df: pd.DataFrame, matrizes: dict) -> tuple:
    """Confere se as matrizes batem com as métricas do CSV e se o teste é o mesmo.
    Devolve (nº de curas, nº de óbitos) do conjunto de teste."""
    problemas, totais_reais = [], set()
    for (exp, modelo), m in matrizes.items():
        tn, fp, fn, tp = (int(v) for v in m.ravel())
        total = tn + fp + fn + tp
        if total != ESPERADO["n_teste"]:
            problemas.append(f"{exp} / {modelo}: matriz soma {total}, esperado {ESPERADO['n_teste']}.")
        totais_reais.add((tn + fp, fn + tp))

        linha = df[(df["experimento"] == exp) & (df["modelo"] == modelo)].iloc[0]
        recalculado = {
            "acuracia": (tn + tp) / total,
            "precisao": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        }
        for chave, valor in recalculado.items():
            if abs(valor - linha[chave]) > 1e-6:
                problemas.append(
                    f"{exp} / {modelo}: {chave} do CSV = {linha[chave]:.6f}, "
                    f"mas a matriz resulta em {valor:.6f}."
                )
    if len(totais_reais) != 1:
        problemas.append(f"As matrizes não têm os mesmos totais por classe real: {sorted(totais_reais)}.")
    if problemas:
        raise ErroValidacao("\n".join(f"  - {p}" for p in problemas))
    return next(iter(totais_reais))


# ───────────────────────── TABELAS DERIVADAS ─────────────────────────
def montar_tabela(df: pd.DataFrame, matrizes: dict) -> pd.DataFrame:
    """Tabela consolidada, em ordem fixa, usada por todos os gráficos."""
    colunas = ["experimento", "modelo", "n_treino", "n_teste", "n_atributos",
               "acuracia", "precisao", "recall", "f1", "tempo_treino_s", "tempo_predicao_s"]
    ordem = [(e, m) for e in EXPERIMENTOS for m in MODELOS]
    t = df[colunas].set_index(["experimento", "modelo"]).loc[ordem].reset_index()
    t["tempo_total_s"] = t["tempo_treino_s"] + t["tempo_predicao_s"]
    for nome, pos in (("tn", (0, 0)), ("fp", (0, 1)), ("fn", (1, 0)), ("tp", (1, 1))):
        t[nome] = [int(matrizes[(e, m)][pos]) for e, m in zip(t["experimento"], t["modelo"])]
    return t


def calcular_variacao(tabela: pd.DataFrame) -> pd.DataFrame:
    """Variação E4 − E5 em pontos percentuais (positivo = melhor no E4)."""
    idx = tabela.set_index(["modelo", "experimento"])
    linhas = []
    for modelo in MODELOS:
        for metrica in METRICAS:
            e4, e5 = idx.loc[(modelo, "E4"), metrica], idx.loc[(modelo, "E5"), metrica]
            linhas.append({"modelo": modelo, "metrica": metrica, "E4": e4, "E5": e5,
                           "variacao_pp": (e4 - e5) * 100})
    return pd.DataFrame(linhas)


# ───────────────────────── FUNÇÕES AUXILIARES ─────────────────────────
def configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "savefig.dpi": DPI,
        "figure.titlesize": 13, "figure.titleweight": "bold",
        "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9.5, "legend.title_fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.45,
    })


def _fmt(valor: float, casas: int = 3) -> str:
    """Número com vírgula decimal (padrão do texto em português)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _milhar(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def _fmt_tempo(v: float) -> str:
    """Tempo em segundos; valores abaixo da resolução do registro (0,1 s) viram '<0,1'."""
    if v < 0.1:
        return "<0,1"
    return _fmt(v, 1) if v < 100 else _fmt(v, 0)


def _eixo_virgula(ax, eixo: str = "y", casas: int = 1) -> None:
    formatador = FuncFormatter(lambda v, _: _fmt(v, casas))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(formatador)


def _eixo_metrica(ax, rotulo_y: str = "Valor da métrica") -> None:
    """Eixo y de 0 a 1 (sem truncar) para métricas."""
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    _eixo_virgula(ax, "y", 1)
    ax.set_ylabel(rotulo_y)
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)


def _barras_agrupadas(ax, dados, x, y, hue, ordem_x, ordem_hue, paleta,
                      formato=None, coluna_rotulo=None, tam_rotulo=8.5) -> None:
    """Barras agrupadas (Seaborn) com os valores escritos sobre cada barra.
    Os rótulos vêm da tabela de dados, não da altura desenhada."""
    sns.barplot(data=dados, x=x, y=y, hue=hue, order=ordem_x, hue_order=ordem_hue,
                palette=paleta, errorbar=None, ax=ax, legend=False,
                saturation=1, width=0.78)
    valores = dados.pivot(index=x, columns=hue, values=coluna_rotulo or y)
    formato = formato or _fmt
    for container, nivel in zip(ax.containers, ordem_hue):
        rotulos = [formato(valores.loc[cat, nivel]) for cat in ordem_x]
        ax.bar_label(container, labels=rotulos, padding=2, fontsize=tam_rotulo)


def _legenda(fig, cores: dict, titulo: str, nomes: dict = None) -> None:
    """Legenda única, abaixo da figura (requer constrained_layout)."""
    nomes = nomes or {k: k for k in cores}
    handles = [Patch(facecolor=c, label=nomes[k]) for k, c in cores.items()]
    fig.legend(handles=handles, title=titulo, loc="outside lower center",
               ncol=len(handles), frameon=False)


def _rodape(fig, texto: str) -> None:
    if not MOSTRAR_RODAPE or not texto:
        return
    largura = max(30, int(fig.get_figwidth() * 17))
    linhas = [textwrap.fill(p, largura) for p in texto.split("\n")]
    fig.text(0.0, -0.01, "\n".join(linhas), ha="left", va="top",
             fontsize=7.5, color="#555555")


def salvar(fig, nome: str, rodape: str = "") -> None:
    _rodape(fig, rodape)
    caminho = FIGURAS_DIR / nome
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    GERADOS.append(nome)
    print(f"  salvo: {caminho.relative_to(BASE_DIR)}")


# ───────────────────────────── GRÁFICOS ─────────────────────────────
def grafico_a(tabela, rodape) -> None:
    """A — Métricas dos três algoritmos, por experimento."""
    longo = tabela.melt(id_vars=["experimento", "modelo"], value_vars=list(METRICAS),
                        var_name="metrica", value_name="valor")
    longo["metrica"] = longo["metrica"].map(METRICAS)
    ordem_x = list(METRICAS.values())

    for exp in EXPERIMENTOS:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
        _barras_agrupadas(ax, longo[longo["experimento"] == exp],
                          "metrica", "valor", "modelo", ordem_x, MODELOS, COR_MODELO)
        _eixo_metrica(ax)
        fig.suptitle(f"Desempenho dos algoritmos — {NOMES_EXP[exp]}")
        _legenda(fig, COR_MODELO, "Algoritmo")
        salvar(fig, f"A_metricas_{exp}.png", rodape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, exp in zip(axes, EXPERIMENTOS):
        _barras_agrupadas(ax, longo[longo["experimento"] == exp],
                          "metrica", "valor", "modelo", ordem_x, MODELOS, COR_MODELO)
        _eixo_metrica(ax)
        ax.set_title(NOMES_EXP[exp])
    fig.suptitle("Desempenho dos algoritmos nos experimentos E4 e E5")
    _legenda(fig, COR_MODELO, "Algoritmo")
    salvar(fig, "A_metricas_E4_E5_lado_a_lado.png", rodape)


def grafico_b(tabela, rodape) -> None:
    """B — Comparação direta E4 × E5, por algoritmo."""
    for metrica in ORDEM_B:
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        dados = tabela[["experimento", "modelo", metrica]].rename(columns={metrica: "valor"})
        _barras_agrupadas(ax, dados, "modelo", "valor", "experimento",
                          MODELOS, EXPERIMENTOS, COR_EXP)
        _eixo_metrica(ax, METRICAS[metrica])
        fig.suptitle(f"{METRICAS[metrica]}: E4 × E5 por algoritmo")
        _legenda(fig, COR_EXP, "Experimento", NOMES_EXP)
        salvar(fig, f"B_comparacao_E4_E5_{metrica}.png", rodape)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    for ax, metrica in zip(axes.ravel(), ORDEM_B):
        dados = tabela[["experimento", "modelo", metrica]].rename(columns={metrica: "valor"})
        _barras_agrupadas(ax, dados, "modelo", "valor", "experimento",
                          MODELOS, EXPERIMENTOS, COR_EXP)
        _eixo_metrica(ax, "Valor da métrica")
        ax.set_title(METRICAS[metrica])
    fig.suptitle("Efeito da retirada de HOSPITAL, UTI e SUPORT_VEN (E4 × E5)")
    _legenda(fig, COR_EXP, "Experimento", NOMES_EXP)
    salvar(fig, "B_comparacao_E4_E5_painel.png", rodape)


def _anotacoes_matriz(m: np.ndarray) -> np.ndarray:
    """Texto de cada célula: nº de registros e % da classe real."""
    pct = m / m.sum(axis=1, keepdims=True) * 100
    return np.array([[f"{_milhar(m[i, j])}\n({_fmt(pct[i, j], 1)}%)" for j in range(2)]
                     for i in range(2)])


def _heatmap_matriz(ax, m: np.ndarray, titulo: str, cbar: bool = False) -> None:
    proporcao = m / m.sum(axis=1, keepdims=True)  # colorido pela proporção da classe real
    sns.heatmap(proporcao, annot=_anotacoes_matriz(m), fmt="", cmap="Blues",
                vmin=0, vmax=1, cbar=cbar,
                cbar_kws={"label": "Proporção da classe real"} if cbar else None,
                xticklabels=ROTULOS_CLASSES, yticklabels=ROTULOS_CLASSES,
                linewidths=1.5, linecolor="white", square=True,
                annot_kws={"fontsize": 10.5}, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="y", rotation=0)


def grafico_c(matrizes, rodape, rodape_curto) -> None:
    """C — Matrizes de confusão (6 individuais + 1 painel)."""
    for exp in EXPERIMENTOS:
        for modelo in MODELOS:
            fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
            _heatmap_matriz(ax, matrizes[(exp, modelo)], f"{modelo} — {exp}", cbar=True)
            salvar(fig, f"C_matriz_confusao_{SLUG_MODELO[modelo]}_{exp}.png", rodape_curto)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True)
    for i, exp in enumerate(EXPERIMENTOS):
        for j, modelo in enumerate(MODELOS):
            ax = axes[i, j]
            _heatmap_matriz(ax, matrizes[(exp, modelo)], f"{modelo} — {exp}")
            if j > 0:
                ax.set_ylabel("")
            if i == 0:
                ax.set_xlabel("")
    fig.colorbar(axes[0, 0].collections[0], ax=axes, shrink=0.6, label="Proporção da classe real")
    fig.suptitle("Matrizes de confusão: algoritmos × experimentos")
    salvar(fig, "C_matrizes_confusao_painel.png",
           rodape + "\nCélulas: nº de registros e % da classe real; cor = proporção da classe real.")


def grafico_d(variacao, rodape) -> None:
    """D — Variação de cada métrica entre E4 e E5, em pontos percentuais."""
    dados = variacao.copy()
    dados["metrica"] = dados["metrica"].map(METRICAS)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    _barras_agrupadas(ax, dados, "metrica", "variacao_pp", "modelo",
                      list(METRICAS.values()), MODELOS, COR_MODELO,
                      formato=lambda v: f"{v:+.1f}".replace(".", ","))
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xlabel("")
    ax.set_ylabel("Variação E4 − E5 (pontos percentuais)")
    _eixo_virgula(ax, "y", 0)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    ax.set_title("Valores positivos = melhor desempenho no E4 (com todos os atributos)",
                 fontsize=9.5, fontweight="normal", color="#444444")
    fig.suptitle("Variação das métricas após a retirada de HOSPITAL, UTI e SUPORT_VEN")
    _legenda(fig, COR_MODELO, "Algoritmo")
    salvar(fig, "D_variacao_metricas_E4_menos_E5.png", rodape)


def grafico_e(tabela, rodape) -> None:
    """E — Custo computacional: (1) escala log; (2) escala linear empilhada."""
    # (1) Escala logarítmica, um painel por etapa. Zeros (tempo < resolução do registro)
    # são desenhados no "piso" do eixo e rotulados como '<0,1'.
    piso = 0.05
    paineis = [("tempo_treino_s", "Treinamento"),
               ("tempo_predicao_s", "Predição (teste completo)"),
               ("tempo_total_s", "Total")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    for ax, (coluna, titulo) in zip(axes, paineis):
        dados = tabela[["experimento", "modelo", coluna]].rename(columns={coluna: "valor"}).copy()
        dados["plot"] = dados["valor"].clip(lower=piso)
        _barras_agrupadas(ax, dados, "modelo", "plot", "experimento", MODELOS, EXPERIMENTOS,
                          COR_EXP, formato=_fmt_tempo, coluna_rotulo="valor")
        ax.set_yscale("log")
        ax.set_ylim(piso / 2, dados["plot"].max() * 4)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.set_ylabel("Tempo (s, escala logarítmica)")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Custo computacional dos algoritmos")
    _legenda(fig, COR_EXP, "Experimento", NOMES_EXP)
    salvar(fig, "E_custo_computacional_log.png",
           rodape + "\nEscala logarítmica: o comprimento das barras não é proporcional ao tempo; "
                    "use os valores rotulados. Tempos registrados com resolução de 0,1 s.")

    # (2) Escala linear, treino + predição empilhados
    total_max = tabela["tempo_total_s"].max()
    fator, unidade = (60.0, "min") if total_max > 600 else (1.0, "s")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True, constrained_layout=True)
    x = np.arange(len(MODELOS))
    cores = [COR_MODELO[m] for m in MODELOS]
    for ax, exp in zip(axes, EXPERIMENTOS):
        sub = tabela[tabela["experimento"] == exp].set_index("modelo").loc[MODELOS]
        treino = sub["tempo_treino_s"].to_numpy() / fator
        predicao = sub["tempo_predicao_s"].to_numpy() / fator
        ax.bar(x, treino, color=cores, edgecolor="white", width=0.6)
        ax.bar(x, predicao, bottom=treino, color=cores, alpha=0.45, hatch="//",
               edgecolor=cores, linewidth=0.8, width=0.6)
        for xi, tr, pr in zip(x, treino, predicao):
            ax.text(xi, tr + pr, f"{_fmt(tr + pr, 1)} {unidade}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELOS)
        ax.set_ylim(0, (treino + predicao).max() * 1.15)
        ax.set_title(NOMES_EXP[exp])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"Tempo ({'minutos' if unidade == 'min' else 'segundos'})")
    _eixo_virgula(axes[0], "y", 0 if unidade == "min" else 1)
    fig.suptitle("Custo computacional total: treinamento + predição")
    handles = [Patch(facecolor="#555555", label="Treinamento"),
               Patch(facecolor="#BBBBBB", edgecolor="#555555", hatch="//", label="Predição")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    salvar(fig, "E_custo_total_empilhado.png",
           rodape + "\nEscala linear; predição sobre o conjunto de teste completo.")


def _rotular_curva_f1(ax, f: float, xlim, ylim) -> None:
    """Escreve 'F1 = f' onde a curva de F1 constante cruza a borda da área visível."""
    xmax = xlim[1]
    p = f * xmax / (2 * xmax - f)
    if ylim[0] <= p <= ylim[1]:
        ax.text(xmax, p, f"F1 = {_fmt(f, 1)}", fontsize=7.5, color="#888888", ha="right", va="bottom")
        return
    ymax = ylim[1]
    r = f * ymax / (2 * ymax - f)
    if xlim[0] <= r <= xlim[1]:
        ax.text(r, ymax, f"F1 = {_fmt(f, 1)}", fontsize=7.5, color="#888888", ha="center", va="top")


def grafico_f(tabela, rodape) -> None:
    """F — Precisão × recall (classe óbito), com curvas de F1 constante."""
    pad = 0.04
    xlim = (max(0, tabela["recall"].min() - pad), min(1, tabela["recall"].max() + pad))
    ylim = (max(0, tabela["precisao"].min() - pad), min(1, tabela["precisao"].max() + pad))

    fig, ax = plt.subplots(figsize=(7.4, 6.0), constrained_layout=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    r = np.linspace(0.01, 1, 600)
    for f in (0.5, 0.6, 0.7, 0.8):
        valido = 2 * r > f
        p = np.full_like(r, np.nan)
        p[valido] = f * r[valido] / (2 * r[valido] - f)
        ax.plot(r, p, ls="--", lw=0.8, color="#B5B5B5", zorder=1)
        _rotular_curva_f1(ax, f, xlim, ylim)

    idx = tabela.set_index(["modelo", "experimento"])
    for modelo in MODELOS:
        pontos = {e: (idx.loc[(modelo, e), "recall"], idx.loc[(modelo, e), "precisao"])
                  for e in EXPERIMENTOS}
        ax.annotate("", xy=pontos["E5"], xytext=pontos["E4"], zorder=2,
                    arrowprops=dict(arrowstyle="->", color=COR_MODELO[modelo], lw=1.2,
                                    shrinkA=7, shrinkB=7))
        for exp, (xr, yp) in pontos.items():
            ax.scatter(xr, yp, s=95, color=COR_MODELO[modelo], marker=MARCADOR_EXP[exp],
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.annotate(exp, (xr, yp), xytext=(7, 5), textcoords="offset points", fontsize=8.5)

    ax.set_xlabel("Recall (classe óbito)")
    ax.set_ylabel("Precisão (classe óbito)")
    _eixo_virgula(ax, "x", 2)
    _eixo_virgula(ax, "y", 2)
    fig.suptitle("Compromisso entre precisão e recall")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                      markersize=9, label=m) for m, c in COR_MODELO.items()]
    handles += [Line2D([0], [0], marker=MARCADOR_EXP[e], color="w", markerfacecolor="#999999",
                       markeredgecolor="black", markersize=9, label=NOMES_EXP[e]) for e in EXPERIMENTOS]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    salvar(fig, "F_precisao_vs_recall.png",
           rodape + "\nSetas: E4 → E5. Eixos recortados na região dos dados; "
                    "linhas tracejadas = F1 constante.")


# ───────────────────────────── PRINCIPAL ─────────────────────────────
def main() -> int:
    try:
        df = carregar_dados()
        validar(df)
        matrizes = extrair_matrizes(df)
        n_cura, n_obito = validar_matrizes(df, matrizes)
    except ErroValidacao as erro:
        print(f"\nERRO DE VALIDAÇÃO:\n{erro}\n\nNenhuma figura foi gerada.")
        return 1

    tabela = montar_tabela(df, matrizes)
    variacao = calcular_variacao(tabela)

    print("Validação concluída: 6 execuções com status 'ok'; amostra, semente e tamanhos consistentes;")
    print("matrizes de confusão conferem com as métricas dos CSVs.\n")
    print("Tabela consolidada (fonte de todos os gráficos):")
    print(tabela.round(4).to_string(index=False))
    print("\nVariação E4 − E5 (pontos percentuais):")
    print(variacao.round(3).to_string(index=False))

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(FIGURAS_DIR / "dados_consolidados.csv", index=False)
    variacao.to_csv(FIGURAS_DIR / "variacao_E4_menos_E5.csv", index=False)

    n_tr, n_te = int(df["n_treino"].iloc[0]), int(df["n_teste"].iloc[0])
    amostra, semente = df["hash_amostra"].iloc[0], int(df["semente"].iloc[0])
    rodape_curto = (f"Teste: {_milhar(n_te)} registros · amostra {amostra} · semente {semente}.")
    rodape = (f"Treino: {_milhar(n_tr)} registros estratificados (amostra {amostra}, semente {semente}) · "
              f"Teste: {_milhar(n_te)} registros ({_milhar(n_cura)} cura; {_milhar(n_obito)} óbito).\n"
              "Precisão, recall e F1-score referem-se à classe óbito. "
              "Execução única por modelo, sem teste de significância estatística.")

    configurar_estilo()
    print(f"\nGerando figuras em {FIGURAS_DIR.relative_to(BASE_DIR)}/ ...")
    grafico_a(tabela, rodape)
    grafico_b(tabela, rodape)
    grafico_c(matrizes, rodape, rodape_curto)
    grafico_d(variacao, rodape)
    grafico_e(tabela, rodape)
    grafico_f(tabela, rodape)
    print(f"\nConcluído: {len(GERADOS)} figuras PNG + 2 tabelas CSV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())