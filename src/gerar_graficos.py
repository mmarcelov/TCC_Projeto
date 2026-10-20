"""
gerar_graficos.py — Tabelas e gráficos dos experimentos E4–E9 (TCC SRAG/COVID-19, 2021).

Lê os CSVs registrados em reports/resultados/. NÃO treina modelos e NÃO altera resultados.

Três análises:
  tamanho   : E6, E4, E7, E8 — 51 atributos; varia o número de registros de treinamento
  atributos : E4 × E5        — 50 mil registros; 51 × 45 atributos
  e9       : E4, E5, E8, E9 — compara tamanho do treinamento e conjunto de atributos

Uso (na raiz do repositório):
    python src/gerar_graficos.py
    python src/gerar_graficos.py --analise tamanho
    python src/gerar_graficos.py --analise atributos
    python src/gerar_graficos.py --analise e9
    python src/gerar_graficos.py --analise todas
"""
import argparse
import itertools
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
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

# ─────────────────────────── CONFIGURAÇÃO ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = BASE_DIR / "reports" / "resultados"
FIG_AMOSTRAS = BASE_DIR / "reports" / "figures" / "amostras"
FIG_COMPARATIVOS = BASE_DIR / "reports" / "figures" / "comparativos"  # E4 × E5 (não toca nos SHAP)
FIG_E9 = BASE_DIR / "reports" / "figures" / "comparativos_E9"        # análise C (E4, E5, E8, E9)

ESPERADO = {
    "semente": 42, "n_teste": 229114, "n_cura": 152459, "n_obito": 76655,
    "hash_e4": "ee971fa098f9",   # amostra de 50 mil registros (E4 e E5)
    "hash_e8": "0eb33db8bae4",   # amostra de 100 mil registros (E8 e E9)
}
COLUNAS_REMOVIDAS = ["HOSPITAL", "UTI", "SUPORT_VEN_1", "SUPORT_VEN_2", "SUPORT_VEN_3", "SUPORT_VEN_9"]

CONFIGURACOES = {
    "completa": "Completa",
    "reduzida": "Reduzida (sem HOSPITAL, UTI e SUPORT_VEN)",
}

# Cada experimento possui uma entrada com seu CSV, tamanho de treino e configuração.
# A análise de tamanho utiliza os experimentos com configuração "completa".
# A análise E9 compara as configurações completa e reduzida com 50 mil e 100 mil registros.
EXPERIMENTOS = {
    "E4": {"arquivo": "equiparidade_20261009_112030_E4.csv", "n_treino": 50000, "n_atributos": 51, "config": "completa"},
    "E5": {"arquivo": "equiparidade_20261009_102402_E5.csv", "n_treino": 50000, "n_atributos": 45, "config": "reduzida"},
    "E6": {"arquivo": "amostras_E6_25000.csv", "n_treino": 25000, "n_atributos": 51, "config": "completa"},
    "E7": {"arquivo": "amostras_E7_75000.csv", "n_treino": 75000, "n_atributos": 51, "config": "completa"},
    "E8": {"arquivo": "amostras_E8_100000.csv", "n_treino": 100000, "n_atributos": 51, "config": "completa"},
    "E9": {"arquivo": "amostras_E9_100000_45atributos.csv", "n_treino": 100000, "n_atributos": 45, "config": "reduzida"},
}
PAR_ATRIBUTOS = ("E4", "E5")  # análise B
SERIE_TAMANHO = sorted(      # análise A: E6, E4, E7, E8
    (e for e, c in EXPERIMENTOS.items() if c["config"] == "completa"),
    key=lambda e: EXPERIMENTOS[e]["n_treino"],
)

# Análise C: grade 2 × 2 (tamanho × atributos)
EXPERIMENTOS_C = ["E4", "E5", "E8", "E9"]
PAR_CUSTO_C = ("E8", "E9")  # custo computacional com a maior amostra
# Contrastes (minuendo, subtraendo): diferença = minuendo − subtraendo
CONTRASTES = [("E4", "E5"), ("E8", "E9"), ("E8", "E4"), ("E9", "E5")]

MODELOS = ["Random Forest", "XGBoost", "SVM"]
METRICAS = {"acuracia": "Acurácia", "precisao": "Precisão", "recall": "Recall", "f1": "F1-score"}
ORDEM_B = ["f1", "acuracia", "precisao", "recall"]
NOMES_EXP = {"E4": "E4 · todos os atributos", "E5": "E5 · sem HOSPITAL, UTI e SUPORT_VEN"}

# Cor = algoritmo (todas as figuras exceto onde a cor indica experimento)
COR_MODELO = {"Random Forest": "#0072B2", "XGBoost": "#E69F00", "SVM": "#009E73"}
MARCADOR_MODELO = {"Random Forest": "o", "XGBoost": "s", "SVM": "^"}
COR_EXP = {"E4": "#4D4D4D", "E5": "#CC79A7"}
MARCADOR_EXP = {"E4": "o", "E5": "s"}
# Análise C: cor = conjunto de atributos (as mesmas de E4/E5); hachura = maior amostra (100 mil)
COR_C = {"E4": "#4D4D4D", "E5": "#CC79A7", "E8": "#4D4D4D", "E9": "#CC79A7"}
SLUG_MODELO = {"Random Forest": "random_forest", "XGBoost": "xgboost", "SVM": "svm"}
ROTULOS_CLASSES = ["Cura (0)", "Óbito (1)"]
CLASSES_CURTAS = ["Cura", "Óbito"]

DPI = 300
MOSTRAR_RODAPE = False  # False remove o rodapé das figuras (útil para o artigo)

COLUNAS_OBRIGATORIAS = [
    "experimento", "modelo", "data_hora", "semente", "hash_amostra", "n_treino", "n_teste",
    "n_atributos", "tempo_treino_s", "tempo_predicao_s", "acuracia", "precisao", "recall",
    "f1", "matriz_confusao", "status",
]
GERADOS = []


class ErroValidacao(Exception):
    """Problema nos CSVs de entrada (ausentes, incompletos ou inconsistentes)."""


# ───────────────── LEITURA, SELEÇÃO E VALIDAÇÃO DOS CSVs ─────────────────
def carregar_execucoes() -> pd.DataFrame:
    """Lê os CSVs de todos os experimentos configurados (todas as linhas, sem filtrar)."""
    partes = []
    for exp, cfg in EXPERIMENTOS.items():
        caminho = RESULTADOS_DIR / cfg["arquivo"]
        if not caminho.exists():
            raise ErroValidacao(
                f"Arquivo não encontrado: {caminho}\n"
                "  Confira o nome em EXPERIMENTOS (topo do script) e a pasta reports/resultados/."
            )
        df = pd.read_csv(caminho, dtype={"hash_amostra": str})
        faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
        if faltando:
            raise ErroValidacao(f"{caminho.name}: colunas ausentes: {faltando}")
        if not (df["experimento"] == exp).all():
            raise ErroValidacao(
                f"{caminho.name}: esperado experimento '{exp}', encontrado {sorted(df['experimento'].unique())}"
            )
        df["arquivo_origem"] = caminho.name
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def selecionar_execucoes(bruto: pd.DataFrame):
    """Critério explícito: para cada (experimento, algoritmo), usa a execução MAIS RECENTE com
    status 'ok'. Linhas com erro ou repetidas ficam de fora; nada é somado nem duplicado."""
    bruto = bruto.reset_index(drop=True)
    bruto["_pos"] = bruto.index
    bruto["_quando"] = pd.to_datetime(bruto["data_hora"], errors="coerce")
    avisos = []

    for _, l in bruto[bruto["status"] != "ok"].iterrows():
        avisos.append(f"{l['experimento']} / {l['modelo']}: linha com status '{str(l['status'])[:70]}' ignorada.")

    ok = bruto[bruto["status"] == "ok"]
    if ok["_quando"].isna().any():
        avisos.append("Há execuções 'ok' sem data_hora legível; nesses casos vale a ordem do arquivo.")

    repetidas = ok.groupby(["experimento", "modelo"]).size()
    for (exp, modelo), q in repetidas[repetidas > 1].items():
        linhas = ok[(ok["experimento"] == exp) & (ok["modelo"] == modelo)]
        iguais = linhas[list(METRICAS)].nunique().max() == 1
        avisos.append(
            f"{exp} / {modelo}: {q} execuções 'ok'; usada a mais recente "
            f"(métricas {'idênticas' if iguais else 'DIFERENTES entre elas — confira'})."
        )

    ordenado = ok.sort_values(["_quando", "_pos"], na_position="first", kind="stable")
    sel = ordenado.groupby(["experimento", "modelo"], sort=False).tail(1)
    sel = sel.drop(columns=["_pos", "_quando"]).set_index(["experimento", "modelo"])

    ordem = [(e, m) for e in EXPERIMENTOS for m in MODELOS]
    faltando = [par for par in ordem if par not in sel.index]
    if faltando:
        raise ErroValidacao(f"Sem execução com status 'ok' para: {faltando}")
    extras = sorted(set(sel.index) - set(ordem))
    if extras:
        avisos.append(f"Combinações fora do esperado foram ignoradas: {extras}")
    return sel.loc[ordem].reset_index(), avisos


def validar(sel: pd.DataFrame) -> None:
    """Confere as execuções selecionadas: quantidade, semente, tamanhos, atributos e amostras."""
    problemas = []
    esperado_n = len(EXPERIMENTOS) * len(MODELOS)
    if len(sel) != esperado_n:
        problemas.append(f"Esperadas {esperado_n} execuções; encontradas {len(sel)}.")
    if sel.duplicated(["experimento", "modelo"]).any():
        problemas.append("Há execuções duplicadas para o mesmo experimento e algoritmo.")

    for coluna in ("semente", "n_teste"):
        achados = set(sel[coluna].astype(str))
        if achados != {str(ESPERADO[coluna])}:
            problemas.append(f"Coluna '{coluna}': esperado {ESPERADO[coluna]}, encontrado {sorted(achados)}.")

    hash_exp = {}
    for exp, cfg in EXPERIMENTOS.items():
        s = sel[sel["experimento"] == exp]
        for coluna in ("n_treino", "n_atributos"):
            achados = set(s[coluna].astype(str))
            if achados != {str(cfg[coluna])}:
                problemas.append(f"{exp}: {coluna} esperado {cfg[coluna]}, encontrado {sorted(achados)}.")
        hashes = set(s["hash_amostra"].astype(str))
        if len(hashes) != 1 or "nan" in hashes:
            problemas.append(f"{exp}: os três algoritmos deveriam usar a mesma amostra; hash_amostra = {sorted(hashes)}.")
        else:
            hash_exp[exp] = next(iter(hashes))

    for exp in PAR_ATRIBUTOS:
        if hash_exp.get(exp) != ESPERADO["hash_e4"]:
            problemas.append(f"{exp}: hash_amostra {hash_exp.get(exp)} difere do registrado para a amostra de 50 mil ({ESPERADO['hash_e4']}).")
    for exp in ("E8", "E9"):
        if hash_exp.get(exp) != ESPERADO["hash_e8"]:
            problemas.append(f"{exp}: hash_amostra {hash_exp.get(exp)} difere do registrado para a amostra de 100 mil ({ESPERADO['hash_e8']}).")
    for e1, e2 in itertools.combinations(hash_exp, 2):
        mesmo_n = EXPERIMENTOS[e1]["n_treino"] == EXPERIMENTOS[e2]["n_treino"]
        mesmo_hash = hash_exp[e1] == hash_exp[e2]
        if mesmo_n != mesmo_hash:
            problemas.append(f"{e1} × {e2}: mesmo tamanho de treino ⇔ mesma amostra? tamanho igual = {mesmo_n}, hash igual = {mesmo_hash}.")

    for coluna in list(METRICAS) + ["tempo_treino_s", "tempo_predicao_s"]:
        if not pd.api.types.is_numeric_dtype(sel[coluna]) or sel[coluna].isna().any():
            problemas.append(f"Coluna '{coluna}' tem valores ausentes ou não numéricos.")
    for coluna in METRICAS:
        if pd.api.types.is_numeric_dtype(sel[coluna]) and not sel[coluna].between(0, 1).all():
            problemas.append(f"Coluna '{coluna}' tem valores fora do intervalo [0, 1].")
    if problemas:
        raise ErroValidacao("\n".join(f"  - {p}" for p in problemas))

def validar_atributos(sel: pd.DataFrame) -> list:
    """Atributos (nomes e ordem): iguais entre experimentos da mesma configuração; a reduzida
    deve ser a completa menos exatamente as 6 colunas previstas. Devolve avisos."""
    problemas, avisos, listas = [], [], {}
    for exp, cfg in EXPERIMENTOS.items():
        s = sel[sel["experimento"] == exp]
        if "atributos" not in s.columns or s["atributos"].isna().any():
            avisos.append(f"{exp}: lista de atributos ausente no CSV; conferência de nomes e ordem ignorada.")
            continue
        distintas = set(s["atributos"])
        if len(distintas) != 1:
            problemas.append(f"{exp}: os três algoritmos registraram listas de atributos diferentes.")
            continue
        lista = next(iter(distintas)).split("; ")
        if len(lista) != cfg["n_atributos"]:
            problemas.append(f"{exp}: lista com {len(lista)} atributos; esperado {cfg['n_atributos']}.")
        listas[exp] = lista

    por_config = {}
    for exp, lista in listas.items():
        por_config.setdefault(EXPERIMENTOS[exp]["config"], {})[exp] = lista
    for config, d in por_config.items():
        if len({tuple(l) for l in d.values()}) != 1:
            problemas.append(f"Configuração '{config}': {sorted(d)} não têm a mesma lista de atributos (nomes e ordem).")

    if "completa" in por_config and "reduzida" in por_config:
        completa = next(iter(por_config["completa"].values()))
        reduzida = next(iter(por_config["reduzida"].values()))
        removidas = [c for c in completa if c not in reduzida]
        if sorted(removidas) != sorted(COLUNAS_REMOVIDAS):
            problemas.append(f"Colunas removidas na reduzida: {removidas}; esperado {COLUNAS_REMOVIDAS}.")
        if reduzida != [c for c in completa if c not in COLUNAS_REMOVIDAS]:
            problemas.append("A configuração reduzida não é a completa sem as 6 colunas (nomes ou ordem).")
    if problemas:
        raise ErroValidacao("\n".join(f"  - {p}" for p in problemas))
    return avisos

def extrair_matrizes(sel: pd.DataFrame) -> dict:
    """Reconstrói as matrizes 2x2 [[TN, FP], [FN, TP]] (classes 0 = cura, 1 = óbito)."""
    matrizes = {}
    for _, l in sel.iterrows():
        try:
            m = np.array(json.loads(l["matriz_confusao"]), dtype=int)
        except (TypeError, ValueError) as erro:
            raise ErroValidacao(f"{l['experimento']} / {l['modelo']}: matriz_confusao ilegível ({erro}).")
        if m.shape != (2, 2):
            raise ErroValidacao(f"{l['experimento']} / {l['modelo']}: matriz com formato {m.shape}, esperado (2, 2).")
        matrizes[(l["experimento"], l["modelo"])] = m
    return matrizes


def validar_matrizes(sel: pd.DataFrame, matrizes: dict) -> tuple:
    """Confere matriz × métricas do CSV e se o teste é o mesmo. Devolve (nº cura, nº óbito)."""
    problemas, totais_reais = [], set()
    for (exp, modelo), m in matrizes.items():
        tn, fp, fn, tp = (int(v) for v in m.ravel())
        total = tn + fp + fn + tp
        if total != ESPERADO["n_teste"]:
            problemas.append(f"{exp} / {modelo}: matriz soma {total}, esperado {ESPERADO['n_teste']}.")
        totais_reais.add((tn + fp, fn + tp))
        linha = sel[(sel["experimento"] == exp) & (sel["modelo"] == modelo)].iloc[0]
        recalculado = {
            "acuracia": (tn + tp) / total,
            "precisao": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        }
        for chave, valor in recalculado.items():
            if abs(valor - linha[chave]) > 1e-6:
                problemas.append(f"{exp} / {modelo}: {chave} do CSV = {linha[chave]:.6f}, mas a matriz resulta em {valor:.6f}.")
    if len(totais_reais) != 1:
        problemas.append(f"As matrizes não têm os mesmos totais por classe real: {sorted(totais_reais)}.")
    elif next(iter(totais_reais)) != (ESPERADO["n_cura"], ESPERADO["n_obito"]):
        problemas.append(f"Totais por classe real {sorted(totais_reais)}; esperado "
                         f"({ESPERADO['n_cura']} cura, {ESPERADO['n_obito']} óbito).")
    if problemas:
        raise ErroValidacao("\n".join(f"  - {p}" for p in problemas))
    return next(iter(totais_reais))


# ───────────────────────────── TABELAS ─────────────────────────────
def montar_tabela(sel: pd.DataFrame, matrizes: dict) -> pd.DataFrame:
    """Tabela consolidada: exatamente uma linha por (experimento, algoritmo)."""
    linhas = []
    for exp, cfg in EXPERIMENTOS.items():
        for modelo in MODELOS:
            l = sel[(sel["experimento"] == exp) & (sel["modelo"] == modelo)].iloc[0]
            tn, fp, fn, tp = (int(v) for v in matrizes[(exp, modelo)].ravel())
            linhas.append({
                "experimento": exp, "modelo": modelo, "configuracao": CONFIGURACOES[cfg["config"]],
                "n_treino": int(l["n_treino"]), "n_teste": int(l["n_teste"]), "n_atributos": int(l["n_atributos"]),
                "acuracia": float(l["acuracia"]), "precisao": float(l["precisao"]),
                "recall": float(l["recall"]), "f1": float(l["f1"]),
                "tempo_treino_s": float(l["tempo_treino_s"]), "tempo_predicao_s": float(l["tempo_predicao_s"]),
                "tempo_total_s": round(float(l["tempo_treino_s"]) + float(l["tempo_predicao_s"]), 1),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "hash_amostra": l["hash_amostra"], "data_hora": l["data_hora"], "arquivo_origem": l["arquivo_origem"],
            })
    tabela = pd.DataFrame(linhas)
    if tabela.duplicated(["experimento", "modelo"]).any() or len(tabela) != len(EXPERIMENTOS) * len(MODELOS):
        raise ErroValidacao("A tabela consolidada tem duplicatas ou número de linhas inesperado.")
    return tabela


def tabela_tamanho(tabela: pd.DataFrame) -> pd.DataFrame:
    """Análise A: tamanhos de amostra (E6, E4, E7, E8) por algoritmo, com variações derivadas."""
    t = tabela[tabela["experimento"].isin(SERIE_TAMANHO)]
    linhas = []
    for modelo in MODELOS:
        s = t[t["modelo"] == modelo].sort_values("n_treino")
        base = s.iloc[0]
        for _, l in s.iterrows():
            linhas.append({
                "algoritmo": modelo, "experimento": l["experimento"], "n_treino": l["n_treino"],
                "n_teste": l["n_teste"], "n_atributos": l["n_atributos"],
                "acuracia": l["acuracia"], "precisao": l["precisao"], "recall": l["recall"], "f1": l["f1"],
                "tempo_treino_s": l["tempo_treino_s"], "tempo_predicao_s": l["tempo_predicao_s"],
                "tempo_total_s": l["tempo_total_s"],
                "variacao_f1_vs_menor_amostra_pp": (l["f1"] - base["f1"]) * 100,
                "razao_tempo_total_vs_menor_amostra": (l["tempo_total_s"] / base["tempo_total_s"]
                                                       if base["tempo_total_s"] > 0 else np.nan),
            })
    return pd.DataFrame(linhas)


def tabela_atributos(tabela: pd.DataFrame) -> pd.DataFrame:
    """Análise B: E4 × E5 por algoritmo (variação em pontos percentuais = E4 − E5)."""
    p4, p5 = PAR_ATRIBUTOS
    idx = tabela.set_index(["modelo", "experimento"])
    linhas = []
    for modelo in MODELOS:
        linha = {"algoritmo": modelo}
        for m in METRICAS:
            a, b = idx.loc[(modelo, p4), m], idx.loc[(modelo, p5), m]
            linha.update({f"{m}_{p4}": a, f"{m}_{p5}": b, f"{m}_variacao_pp_{p4}_menos_{p5}": (a - b) * 100})
        for c in ("tempo_treino_s", "tempo_predicao_s", "tempo_total_s"):
            linha.update({f"{c}_{p4}": idx.loc[(modelo, p4), c], f"{c}_{p5}": idx.loc[(modelo, p5), c]})
        linhas.append(linha)
    return pd.DataFrame(linhas)

def _descrever_contraste(a: str, b: str) -> str:
    """Texto do contraste a − b, derivado do cadastro de experimentos."""
    ca, cb = EXPERIMENTOS[a], EXPERIMENTOS[b]
    if ca["n_treino"] == cb["n_treino"]:
        return (f"Retirada de atributos com {_milhar(ca['n_treino'])} registros: "
                f"{a} ({ca['n_atributos']} atributos) − {b} ({cb['n_atributos']} atributos)")
    return (f"Mudança do tamanho de treinamento com {ca['n_atributos']} atributos: "
            f"{a} ({_milhar(ca['n_treino'])} registros) − {b} ({_milhar(cb['n_treino'])} registros)")


def tabela_par(tabela: pd.DataFrame, minuendo: str, subtraendo: str) -> pd.DataFrame:
    """Dois experimentos por algoritmo. Diferenças = minuendo − subtraendo, em pontos percentuais."""
    idx = tabela.set_index(["modelo", "experimento"])
    descricao = _descrever_contraste(minuendo, subtraendo)
    linhas = []
    for modelo in MODELOS:
        linha = {"algoritmo": modelo, "comparacao": f"{minuendo} − {subtraendo}", "descricao": descricao}
        for m in METRICAS:
            a, b = idx.loc[(modelo, minuendo), m], idx.loc[(modelo, subtraendo), m]
            linha.update({f"{m}_{minuendo}": a, f"{m}_{subtraendo}": b,
                          f"{m}_diferenca_pp_{minuendo}_menos_{subtraendo}": (a - b) * 100})
        for c in ("tempo_treino_s", "tempo_predicao_s", "tempo_total_s"):
            linha.update({f"{c}_{minuendo}": idx.loc[(modelo, minuendo), c],
                          f"{c}_{subtraendo}": idx.loc[(modelo, subtraendo), c]})
        ta, tb = idx.loc[(modelo, minuendo), "tempo_total_s"], idx.loc[(modelo, subtraendo), "tempo_total_s"]
        linha[f"razao_tempo_total_{minuendo}_sobre_{subtraendo}"] = ta / tb if tb > 0 else np.nan
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_comparativo(tabela: pd.DataFrame) -> pd.DataFrame:
    """Os quatro contrastes da grade 2 × 2 (formato longo: um contraste por algoritmo)."""
    idx = tabela.set_index(["modelo", "experimento"])
    linhas = []
    for minuendo, subtraendo in CONTRASTES:
        for modelo in MODELOS:
            ma, sb = idx.loc[(modelo, minuendo)], idx.loc[(modelo, subtraendo)]
            linha = {
                "algoritmo": modelo, "contraste": f"{minuendo} − {subtraendo}",
                "descricao": _descrever_contraste(minuendo, subtraendo),
                "n_treino_minuendo": int(ma["n_treino"]), "n_atributos_minuendo": int(ma["n_atributos"]),
                "n_treino_subtraendo": int(sb["n_treino"]), "n_atributos_subtraendo": int(sb["n_atributos"]),
            }
            for m in METRICAS:
                linha.update({f"{m}_minuendo": ma[m], f"{m}_subtraendo": sb[m],
                              f"{m}_diferenca_pp": (ma[m] - sb[m]) * 100})
            for c in ("tempo_treino_s", "tempo_predicao_s", "tempo_total_s"):
                linha.update({f"{c}_minuendo": ma[c], f"{c}_subtraendo": sb[c]})
            linha["razao_tempo_total_minuendo_sobre_subtraendo"] = (
                ma["tempo_total_s"] / sb["tempo_total_s"] if sb["tempo_total_s"] > 0 else np.nan)
            linhas.append(linha)
    return pd.DataFrame(linhas)

def exportar_tabela(df: pd.DataFrame, nome: str) -> None:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RESULTADOS_DIR / nome
    df.rename(columns={"modelo": "algoritmo"}).to_csv(caminho, index=False)
    print(f"  tabela salva: {caminho.relative_to(BASE_DIR)}")


# ───────────────────────── FUNÇÕES AUXILIARES ─────────────────────────
def configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "savefig.dpi": DPI,
        "figure.titlesize": 13, "figure.titleweight": "bold",
        "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9.5, "legend.title_fontsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": 0.45,
    })


def _fmt(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _milhar(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def _fmt_tempo(v: float) -> str:
    if v < 0.1:
        return "<0,1"
    return _fmt(v, 1) if v < 100 else _fmt(v, 0)


def _rotulo_tamanho(n: int) -> str:
    return f"{n // 1000} mil" if n % 1000 == 0 else _milhar(n)


def _tamanhos(tab: pd.DataFrame) -> list:
    return sorted(int(n) for n in tab["n_treino"].unique())


def _eixo_virgula(ax, eixo: str = "y", casas: int = 1) -> None:
    formatador = FuncFormatter(lambda v, _: _fmt(v, casas))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(formatador)


def _eixo_metrica(ax, rotulo_y: str = "Valor da métrica") -> None:
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    _eixo_virgula(ax, "y", 1)
    ax.set_ylabel(rotulo_y)
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)


def _barras_agrupadas(ax, dados, x, y, hue, ordem_x, ordem_hue, paleta,
                      formato=None, coluna_rotulo=None, tam_rotulo=8.5) -> None:
    """Barras agrupadas (Seaborn); os rótulos vêm da tabela, não da altura desenhada."""
    sns.barplot(data=dados, x=x, y=y, hue=hue, order=ordem_x, hue_order=ordem_hue,
                palette=paleta, errorbar=None, ax=ax, legend=False, saturation=1, width=0.78)
    valores = dados.pivot(index=x, columns=hue, values=coluna_rotulo or y)
    formato = formato or _fmt
    for container, nivel in zip(ax.containers, ordem_hue):
        rotulos = [formato(valores.loc[cat, nivel]) for cat in ordem_x]
        ax.bar_label(container, labels=rotulos, padding=2, fontsize=tam_rotulo)


def _legenda(fig, cores: dict, titulo: str, nomes: dict = None) -> None:
    """Legenda em área reservada abaixo da figura (requer constrained_layout)."""
    nomes = nomes or {k: k for k in cores}
    handles = [Patch(facecolor=c, label=nomes[k]) for k, c in cores.items()]
    fig.legend(handles=handles, title=titulo, loc="outside lower center", ncol=len(handles), frameon=False)


def _legenda_linhas(fig) -> None:
    handles = [Line2D([0], [0], color=COR_MODELO[m], marker=MARCADOR_MODELO[m], lw=1.8, ms=7, label=m)
               for m in MODELOS]
    fig.legend(handles=handles, title="Algoritmo", loc="outside lower center", ncol=len(handles), frameon=False)


def _rodape(fig, texto: str) -> None:
    """Rodapé FORA da área da figura (abaixo da legenda), com quebra de linha pela largura."""
    if not MOSTRAR_RODAPE or not texto:
        return
    largura = max(30, int(fig.get_figwidth() * 17))
    linhas = [textwrap.fill(p, largura) for p in texto.split("\n")]
    fig.text(0.0, -0.01, "\n".join(linhas), ha="left", va="top", fontsize=7.5, color="#555555")


def salvar(fig, pasta: Path, nome: str, rodape: str = "") -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    _rodape(fig, rodape)
    caminho = pasta / nome
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    GERADOS.append(caminho.relative_to(BASE_DIR))
    print(f"  salvo: {caminho.relative_to(BASE_DIR)}")


# ──────────── ANÁLISE A — TAMANHO DA AMOSTRA (E6, E4, E7, E8) ────────────
def _afastar_posicoes(posicoes, separacao, limite_inf, limite_sup):
    """Afasta rótulos próximos (no espaço do eixo), preservando a ordem, para não sobrepor."""
    pos = np.array(posicoes, dtype=float)
    ordem = np.argsort(pos)
    for a, b in zip(ordem[:-1], ordem[1:]):
        if pos[b] - pos[a] < separacao:
            pos[b] = pos[a] + separacao
    excesso = pos.max() - limite_sup
    if excesso > 0:
        pos -= excesso
    return np.maximum(pos, limite_inf)


def _rotular_finais(ax, pontos, log: bool) -> None:
    """Valor do último ponto de cada linha, à direita do gráfico e sem sobreposição."""
    y0, y1 = ax.get_ylim()
    t = np.log10 if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    lo, hi = t(y0), t(y1)
    ajustadas = _afastar_posicoes([t(p[2]) for p in pontos], 0.075 * (hi - lo),
                                  lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo))
    x0, x1 = ax.get_xlim()
    for (modelo, x, _, texto), pos in zip(pontos, ajustadas):
        ax.text(x + 0.025 * (x1 - x0), inv(pos), texto, color=COR_MODELO[modelo],
                fontsize=9, fontweight="bold", va="center", ha="left")


def _plot_linhas(ax, tab, coluna, log=False, formato=_fmt, piso=None, ylim=None) -> None:
    """Uma linha por algoritmo: `coluna` × nº de registros de treinamento (eixo x proporcional)."""
    tamanhos = _tamanhos(tab)
    pontos, todos = [], []
    for modelo in MODELOS:
        s = tab[tab["modelo"] == modelo].sort_values("n_treino")
        y_real = s[coluna].to_numpy(float)
        y_plot = np.clip(y_real, piso, None) if piso is not None else y_real
        ax.plot(s["n_treino"], y_plot, color=COR_MODELO[modelo], marker=MARCADOR_MODELO[modelo],
                lw=1.8, ms=6.5, zorder=3)
        pontos.append((modelo, s["n_treino"].iloc[-1], y_plot[-1], formato(y_real[-1])))
        todos.extend(y_plot)

    span = tamanhos[-1] - tamanhos[0]
    ax.set_xticks(tamanhos)
    ax.set_xticklabels([_rotulo_tamanho(n) for n in tamanhos])
    ax.set_xlim(tamanhos[0] - 0.06 * span, tamanhos[-1] + 0.28 * span)  # espaço para os rótulos finais
    ax.set_xlabel("Registros de treinamento")
    ax.grid(axis="x", visible=False)

    if log:
        ax.set_yscale("log")
        ax.set_ylim(min(todos) / 1.6, max(todos) * 1.6)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        if ylim is None:
            lo, hi = min(todos), max(todos)
            amp = max(hi - lo, 0.01)
            ylim = (lo - 0.25 * amp, hi + 0.25 * amp)  # eixo recortado: indicado no rodapé
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        _eixo_virgula(ax, "y", 3)
    _rotular_finais(ax, pontos, log)


def _nota_tempo(tab, colunas) -> str:
    nota = "Escala logarítmica no eixo y. Tempos registrados com resolução de 0,1 s."
    if any((tab[c] < 0.1).any() for c in colunas):
        nota += " Valores < 0,1 s foram desenhados em 0,05 s e rotulados como '<0,1'."
    return nota


def fig_tam_f1(tab, rodape) -> None:
    """A1 — F1-score × tamanho da amostra."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    _plot_linhas(ax, tab, "f1")
    ax.set_ylabel("F1-score (classe óbito)")
    fig.suptitle("F1-score × tamanho da amostra de treinamento")
    _legenda_linhas(fig)
    salvar(fig, FIG_AMOSTRAS, "A1_f1_vs_tamanho_treino.png",
           rodape + "\nEixo y recortado na faixa dos dados.")


def fig_tam_metricas(tab, rodape) -> None:
    """A2 — Acurácia, precisão e recall × tamanho da amostra."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), constrained_layout=True)
    for ax, metrica in zip(axes, ["acuracia", "precisao", "recall"]):
        _plot_linhas(ax, tab, metrica)
        ax.set_title(METRICAS[metrica])
        ax.set_ylabel("Valor da métrica")
    fig.suptitle("Acurácia, precisão e recall × tamanho da amostra de treinamento")
    _legenda_linhas(fig)
    salvar(fig, FIG_AMOSTRAS, "A2_metricas_vs_tamanho_treino_painel.png",
           rodape + "\nEixos y recortados na faixa dos dados de cada painel (escalas diferentes entre painéis).")


def fig_tam_tempos(tab, rodape) -> None:
    """A3 — Tempo de treinamento e de predição × tamanho da amostra (escala log)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    paineis = [("tempo_treino_s", "Tempo de treinamento"), ("tempo_predicao_s", "Tempo de predição (teste completo)")]
    for ax, (coluna, titulo) in zip(axes, paineis):
        _plot_linhas(ax, tab, coluna, log=True, formato=_fmt_tempo, piso=0.05)
        ax.set_title(titulo)
        ax.set_ylabel("Tempo (s, escala logarítmica)")
    fig.suptitle("Custo computacional por etapa × tamanho da amostra")
    _legenda_linhas(fig)
    salvar(fig, FIG_AMOSTRAS, "A3_tempos_vs_tamanho_treino.png",
           rodape + "\n" + _nota_tempo(tab, ["tempo_treino_s", "tempo_predicao_s"]))


def fig_tam_custo_total(tab, rodape) -> None:
    """A4 — Custo computacional total (treinamento + predição) × tamanho da amostra."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    _plot_linhas(ax, tab, "tempo_total_s", log=True, formato=_fmt_tempo, piso=0.05)
    ax.set_ylabel("Tempo total (s, escala logarítmica)")
    fig.suptitle("Custo computacional total × tamanho da amostra")
    _legenda_linhas(fig)
    salvar(fig, FIG_AMOSTRAS, "A4_custo_total_vs_tamanho_treino.png",
           rodape + "\n" + _nota_tempo(tab, ["tempo_total_s"]) + " Total = treinamento + predição.")


def fig_tam_matriz_metricas(tab, rodape) -> None:
    """A5 — Matriz comparativa: algoritmo × tamanho, uma matriz por métrica."""
    tamanhos = _tamanhos(tab)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.9), constrained_layout=True)
    for ax, metrica in zip(axes.ravel(), ORDEM_B):
        piv = tab.pivot(index="modelo", columns="n_treino", values=metrica).loc[MODELOS, tamanhos]
        v = piv.to_numpy(float)
        amp = v.max() - v.min()
        vmin = v.min() - 0.35 * amp if amp > 0 else v.min() - 1.0  # evita a célula mais clara ficar branca
        anot = np.array([[_fmt(x) for x in linha] for linha in v])
        sns.heatmap(piv, annot=anot, fmt="", cmap="Blues", vmin=vmin, vmax=v.max(), cbar=False,
                    linewidths=1.2, linecolor="white", annot_kws={"fontsize": 10.5},
                    xticklabels=[_rotulo_tamanho(n) for n in tamanhos], yticklabels=MODELOS, ax=ax)
        ax.set_title(METRICAS[metrica])
        ax.set_xlabel("Registros de treinamento")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Métricas por algoritmo e tamanho da amostra de treinamento")
    salvar(fig, FIG_AMOSTRAS, "A5_matriz_metricas_algoritmo_tamanho.png",
           rodape + "\nCor normalizada dentro de cada métrica (mais escuro = maior valor); "
                    "não compare cores entre painéis.")


def fig_tam_matrizes_confusao(tab, matrizes, rodape) -> None:
    """A6 — Matrizes de confusão (algoritmo × tamanho), só com % da classe real."""
    exps = SERIE_TAMANHO
    nl, nc = len(MODELOS), len(exps)
    fig, axes = plt.subplots(nl, nc, figsize=(3.0 * nc + 1.4, 2.8 * nl + 1.0), constrained_layout=True)
    for i, modelo in enumerate(MODELOS):
        for j, exp in enumerate(exps):
            ax = axes[i, j]
            m = matrizes[(exp, modelo)]
            prop = m / m.sum(axis=1, keepdims=True)
            anot = np.array([[f"{_fmt(prop[a, b] * 100, 1)}%" for b in range(2)] for a in range(2)])
            sns.heatmap(prop, annot=anot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, square=True,
                        linewidths=1.2, linecolor="white", annot_kws={"fontsize": 10},
                        xticklabels=CLASSES_CURTAS, yticklabels=CLASSES_CURTAS, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel(modelo if j == 0 else "", fontweight="bold")
            if i == 0:
                ax.set_title(_rotulo_tamanho(EXPERIMENTOS[exp]["n_treino"]) + " registros")
            ax.tick_params(axis="y", rotation=0, labelleft=(j == 0))
            ax.tick_params(axis="x", labelbottom=(i == nl - 1))
    fig.colorbar(axes[0, 0].collections[0], ax=axes.ravel().tolist(), shrink=0.55,
                 label="Proporção da classe real")
    fig.suptitle("Matrizes de confusão por algoritmo e tamanho da amostra")
    salvar(fig, FIG_AMOSTRAS, "A6_matrizes_confusao_algoritmo_tamanho.png",
           rodape + "\nLinhas = classe real; colunas = classe predita; valores = % da classe real "
                    "(diagonal: cura correta e recall do óbito). Contagens em consolidado_experimentos.csv.")


# ───────────── ANÁLISE B — ATRIBUTOS (E4 × E5), mantida do script anterior ─────────────
def calcular_variacao(tab) -> pd.DataFrame:
    """Variação E4 − E5 em pontos percentuais (positivo = melhor no E4)."""
    p4, p5 = PAR_ATRIBUTOS
    idx = tab.set_index(["modelo", "experimento"])
    linhas = []
    for modelo in MODELOS:
        for metrica in METRICAS:
            a, b = idx.loc[(modelo, p4), metrica], idx.loc[(modelo, p5), metrica]
            linhas.append({"modelo": modelo, "metrica": metrica, "E4": a, "E5": b, "variacao_pp": (a - b) * 100})
    return pd.DataFrame(linhas)


def fig_atr_metricas(tab, rodape) -> None:
    longo = tab.melt(id_vars=["experimento", "modelo"], value_vars=list(METRICAS),
                     var_name="metrica", value_name="valor")
    longo["metrica"] = longo["metrica"].map(METRICAS)
    ordem_x = list(METRICAS.values())
    for exp in PAR_ATRIBUTOS:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
        _barras_agrupadas(ax, longo[longo["experimento"] == exp], "metrica", "valor", "modelo",
                          ordem_x, MODELOS, COR_MODELO)
        _eixo_metrica(ax)
        fig.suptitle(f"Desempenho dos algoritmos — {NOMES_EXP[exp]}")
        _legenda(fig, COR_MODELO, "Algoritmo")
        salvar(fig, FIG_COMPARATIVOS, f"A_metricas_{exp}.png", rodape)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    for ax, exp in zip(axes, PAR_ATRIBUTOS):
        _barras_agrupadas(ax, longo[longo["experimento"] == exp], "metrica", "valor", "modelo",
                          ordem_x, MODELOS, COR_MODELO)
        _eixo_metrica(ax)
        ax.set_title(NOMES_EXP[exp])
    fig.suptitle("Desempenho dos algoritmos nos experimentos E4 e E5")
    _legenda(fig, COR_MODELO, "Algoritmo")
    salvar(fig, FIG_COMPARATIVOS, "A_metricas_E4_E5_lado_a_lado.png", rodape)


def fig_atr_comparacao(tab, rodape) -> None:
    par = list(PAR_ATRIBUTOS)
    for metrica in ORDEM_B:
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        dados = tab[["experimento", "modelo", metrica]].rename(columns={metrica: "valor"})
        _barras_agrupadas(ax, dados, "modelo", "valor", "experimento", MODELOS, par, COR_EXP)
        _eixo_metrica(ax, METRICAS[metrica])
        fig.suptitle(f"{METRICAS[metrica]}: E4 × E5 por algoritmo")
        _legenda(fig, COR_EXP, "Experimento", NOMES_EXP)
        salvar(fig, FIG_COMPARATIVOS, f"B_comparacao_E4_E5_{metrica}.png", rodape)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    for ax, metrica in zip(axes.ravel(), ORDEM_B):
        dados = tab[["experimento", "modelo", metrica]].rename(columns={metrica: "valor"})
        _barras_agrupadas(ax, dados, "modelo", "valor", "experimento", MODELOS, par, COR_EXP)
        _eixo_metrica(ax, "Valor da métrica")
        ax.set_title(METRICAS[metrica])
    fig.suptitle("Efeito da retirada de HOSPITAL, UTI e SUPORT_VEN (E4 × E5)")
    _legenda(fig, COR_EXP, "Experimento", NOMES_EXP)
    salvar(fig, FIG_COMPARATIVOS, "B_comparacao_E4_E5_painel.png", rodape)


def _anotacoes_matriz(m: np.ndarray) -> np.ndarray:
    pct = m / m.sum(axis=1, keepdims=True) * 100
    return np.array([[f"{_milhar(m[i, j])}\n({_fmt(pct[i, j], 1)}%)" for j in range(2)] for i in range(2)])


def _heatmap_matriz(ax, m: np.ndarray, titulo: str, cbar: bool = False) -> None:
    proporcao = m / m.sum(axis=1, keepdims=True)
    sns.heatmap(proporcao, annot=_anotacoes_matriz(m), fmt="", cmap="Blues", vmin=0, vmax=1, cbar=cbar,
                cbar_kws={"label": "Proporção da classe real"} if cbar else None,
                xticklabels=ROTULOS_CLASSES, yticklabels=ROTULOS_CLASSES,
                linewidths=1.5, linecolor="white", square=True, annot_kws={"fontsize": 10.5}, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="y", rotation=0)


def fig_atr_matrizes(matrizes, rodape, rodape_curto) -> None:
    for exp in PAR_ATRIBUTOS:
        for modelo in MODELOS:
            fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
            _heatmap_matriz(ax, matrizes[(exp, modelo)], f"{modelo} — {exp}", cbar=True)
            salvar(fig, FIG_COMPARATIVOS, f"C_matriz_confusao_{SLUG_MODELO[modelo]}_{exp}.png", rodape_curto)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True)
    for i, exp in enumerate(PAR_ATRIBUTOS):
        for j, modelo in enumerate(MODELOS):
            ax = axes[i, j]
            _heatmap_matriz(ax, matrizes[(exp, modelo)], f"{modelo} — {exp}")
            if j > 0:
                ax.set_ylabel("")
            if i == 0:
                ax.set_xlabel("")
    fig.colorbar(axes[0, 0].collections[0], ax=axes.ravel().tolist(), shrink=0.6, label="Proporção da classe real")
    fig.suptitle("Matrizes de confusão: algoritmos × experimentos")
    salvar(fig, FIG_COMPARATIVOS, "C_matrizes_confusao_painel.png",
           rodape + "\nCélulas: nº de registros e % da classe real; cor = proporção da classe real.")


def fig_atr_variacao(variacao, rodape) -> None:
    dados = variacao.copy()
    dados["metrica"] = dados["metrica"].map(METRICAS)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    _barras_agrupadas(ax, dados, "metrica", "variacao_pp", "modelo", list(METRICAS.values()), MODELOS,
                      COR_MODELO, formato=lambda v: f"{v:+.1f}".replace(".", ","))
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
    salvar(fig, FIG_COMPARATIVOS, "D_variacao_metricas_E4_menos_E5.png", rodape)


def fig_atr_custo(tab, rodape) -> None:
    piso = 0.05
    paineis = [("tempo_treino_s", "Treinamento"), ("tempo_predicao_s", "Predição (teste completo)"),
               ("tempo_total_s", "Total")]
    par = list(PAR_ATRIBUTOS)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    for ax, (coluna, titulo) in zip(axes, paineis):
        dados = tab[["experimento", "modelo", coluna]].rename(columns={coluna: "valor"}).copy()
        dados["plot"] = dados["valor"].clip(lower=piso)
        _barras_agrupadas(ax, dados, "modelo", "plot", "experimento", MODELOS, par, COR_EXP,
                          formato=_fmt_tempo, coluna_rotulo="valor")
        ax.set_yscale("log")
        ax.set_ylim(piso / 2, dados["plot"].max() * 4)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.set_ylabel("Tempo (s, escala logarítmica)")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Custo computacional dos algoritmos")
    _legenda(fig, COR_EXP, "Experimento", NOMES_EXP)
    salvar(fig, FIG_COMPARATIVOS, "E_custo_computacional_log.png",
           rodape + "\nEscala logarítmica: o comprimento das barras não é proporcional ao tempo; "
                    "use os valores rotulados. Tempos registrados com resolução de 0,1 s.")

    fator, unidade = (60.0, "min") if tab["tempo_total_s"].max() > 600 else (1.0, "s")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True, constrained_layout=True)
    x = np.arange(len(MODELOS))
    cores = [COR_MODELO[m] for m in MODELOS]
    for ax, exp in zip(axes, PAR_ATRIBUTOS):
        sub = tab[tab["experimento"] == exp].set_index("modelo").loc[MODELOS]
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
    salvar(fig, FIG_COMPARATIVOS, "E_custo_total_empilhado.png",
           rodape + "\nEscala linear; predição sobre o conjunto de teste completo.")


def _rotular_curva_f1(ax, f: float, xlim, ylim) -> None:
    xmax = xlim[1]
    p = f * xmax / (2 * xmax - f)
    if ylim[0] <= p <= ylim[1]:
        ax.text(xmax, p, f"F1 = {_fmt(f, 1)}", fontsize=7.5, color="#888888", ha="right", va="bottom")
        return
    ymax = ylim[1]
    r = f * ymax / (2 * ymax - f)
    if xlim[0] <= r <= xlim[1]:
        ax.text(r, ymax, f"F1 = {_fmt(f, 1)}", fontsize=7.5, color="#888888", ha="center", va="top")


def fig_atr_precisao_recall(tab, rodape) -> None:
    pad = 0.04
    xlim = (max(0, tab["recall"].min() - pad), min(1, tab["recall"].max() + pad))
    ylim = (max(0, tab["precisao"].min() - pad), min(1, tab["precisao"].max() + pad))
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
    idx = tab.set_index(["modelo", "experimento"])
    for modelo in MODELOS:
        pontos = {e: (idx.loc[(modelo, e), "recall"], idx.loc[(modelo, e), "precisao"]) for e in PAR_ATRIBUTOS}
        ax.annotate("", xy=pontos["E5"], xytext=pontos["E4"], zorder=2,
                    arrowprops=dict(arrowstyle="->", color=COR_MODELO[modelo], lw=1.2, shrinkA=7, shrinkB=7))
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
                       markeredgecolor="black", markersize=9, label=NOMES_EXP[e]) for e in PAR_ATRIBUTOS]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    salvar(fig, FIG_COMPARATIVOS, "F_precisao_vs_recall.png",
           rodape + "\nSetas: E4 → E5. Eixos recortados na região dos dados; linhas tracejadas = F1 constante.")

# ───────── ANÁLISE C — TAMANHO × ATRIBUTOS (E4, E5, E8, E9) ─────────
def grade_c() -> dict:
    """Grade 2 × 2 {(configuração, n_treino): experimento}; falha se estiver incompleta."""
    grade = {(EXPERIMENTOS[e]["config"], EXPERIMENTOS[e]["n_treino"]): e for e in EXPERIMENTOS_C}
    if len(grade) != 4 or len({c for c, _ in grade}) != 2 or len({n for _, n in grade}) != 2:
        raise ErroValidacao(f"A grade 2 × 2 (tamanho × atributos) da análise C está incompleta: {sorted(grade.items())}")
    return grade


def _nome_c(exp: str) -> str:
    cfg = EXPERIMENTOS[exp]
    return f"{exp} · {_rotulo_tamanho(cfg['n_treino'])} · {cfg['n_atributos']} atributos"


def _hachurar_maior(ax, experimentos) -> None:
    """Hachura (listras brancas) as barras dos experimentos com a maior amostra de treinamento."""
    n_max = max(EXPERIMENTOS[e]["n_treino"] for e in experimentos)
    for container, exp in zip(ax.containers, experimentos):
        if EXPERIMENTOS[exp]["n_treino"] == n_max:
            for patch in container.patches:
                patch.set_hatch("//")
                patch.set_edgecolor("white")
                patch.set_linewidth(0.6)


def _legenda_c(fig) -> None:
    n_max = max(EXPERIMENTOS[e]["n_treino"] for e in EXPERIMENTOS_C)
    handles = [Patch(facecolor=COR_C[e], edgecolor="white",
                     hatch="//" if EXPERIMENTOS[e]["n_treino"] == n_max else None, label=_nome_c(e))
               for e in EXPERIMENTOS_C]
    fig.legend(handles=handles, title="Experimento", loc="outside lower center", ncol=2, frameon=False)


def montar_rodape_c(tab, grade, n_cura, n_obito, semente) -> str:
    partes = []
    for n in sorted({n for _, n in grade}):
        h = tab.loc[tab["experimento"] == grade[("completa", n)], "hash_amostra"].iloc[0]
        partes.append(f"{_milhar(n)} (amostra {h})")
    atributos = " e ".join(str(a) for a in sorted({EXPERIMENTOS[e]["n_atributos"] for e in EXPERIMENTOS_C}, reverse=True))
    n_teste = int(tab["n_teste"].iloc[0])
    return (f"Treino: {' e '.join(partes)} registros; {atributos} atributos; semente {semente} · "
            f"Teste: {_milhar(n_teste)} registros ({_milhar(n_cura)} cura; {_milhar(n_obito)} óbito).\n"
            "Precisão, recall e F1-score referem-se à classe óbito. Execução única por experimento, sem teste "
            "de significância estatística; as diferenças são descritivas.")


def fig_c_f1(tab, rodape) -> None:
    """C1 — F1-score dos quatro experimentos, por algoritmo."""
    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    dados = tab[["experimento", "modelo", "f1"]].rename(columns={"f1": "valor"})
    _barras_agrupadas(ax, dados, "modelo", "valor", "experimento", MODELOS, EXPERIMENTOS_C, COR_C, tam_rotulo=8)
    _hachurar_maior(ax, EXPERIMENTOS_C)
    _eixo_metrica(ax, "F1-score (classe óbito)")
    fig.suptitle("F1-score por algoritmo: tamanho da amostra e conjunto de atributos")
    _legenda_c(fig)
    salvar(fig, FIG_E9, "C1_f1_E4_E5_E8_E9.png", rodape)


def fig_c_remocao(tab, grade, rodape) -> None:
    """C2 — Diferença de F1 (pontos percentuais) ao retirar os atributos, em cada tamanho."""
    idx = tab.set_index(["modelo", "experimento"])
    tamanhos = sorted({n for _, n in grade})
    linhas, contrastes = [], []
    for n in tamanhos:
        com, sem = grade[("completa", n)], grade[("reduzida", n)]
        rotulo = f"{com} − {sem} ({_rotulo_tamanho(n)} registros)"
        contrastes.append(rotulo)
        for modelo in MODELOS:
            dif = (idx.loc[(modelo, com), "f1"] - idx.loc[(modelo, sem), "f1"]) * 100
            linhas.append({"modelo": modelo, "contraste": rotulo, "variacao_pp": dif})
    n_com = EXPERIMENTOS[grade[("completa", tamanhos[0])]]["n_atributos"]
    n_sem = EXPERIMENTOS[grade[("reduzida", tamanhos[0])]]["n_atributos"]
    cor = "#7F7F7F"
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    _barras_agrupadas(ax, pd.DataFrame(linhas), "modelo", "variacao_pp", "contraste", MODELOS, contrastes,
                      {c: cor for c in contrastes}, formato=lambda v: f"{v:+.1f}".replace(".", ","))
    for patch in ax.containers[-1].patches:  # contraste da maior amostra: hachurado
        patch.set_hatch("//")
        patch.set_edgecolor("white")
        patch.set_linewidth(0.6)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xlabel("")
    ax.set_ylabel("Diferença de F1-score (pontos percentuais)")
    _eixo_virgula(ax, "y", 0)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    ax.set_title(f"Diferença = F1 com {n_com} atributos − F1 com {n_sem}; "
                 f"positivo = F1 maior com {n_com} atributos",
                 fontsize=9.5, fontweight="normal", color="#444444")
    fig.suptitle("Variação do F1-score após a retirada de HOSPITAL, UTI e SUPORT_VEN")
    handles = [Patch(facecolor=cor, edgecolor="white", hatch="//" if i == len(contrastes) - 1 else None, label=c)
               for i, c in enumerate(contrastes)]
    fig.legend(handles=handles, title="Contraste", loc="outside lower center", ncol=2, frameon=False)
    salvar(fig, FIG_E9, "C2_remocao_atributos_f1_pp.png", rodape)


def fig_c_tamanho(tab, grade, rodape) -> None:
    """C3 — F1-score com a amostra menor e a maior, em painéis separados por conjunto de atributos."""
    tamanhos = sorted({n for _, n in grade})
    valores = tab["f1"].to_numpy(float)
    amp = max(valores.max() - valores.min(), 0.01)
    ylim = (valores.min() - 0.25 * amp, valores.max() + 0.25 * amp)  # mesma escala nos dois painéis
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    for ax, config in zip(axes, ("completa", "reduzida")):
        exps = [grade[(config, n)] for n in tamanhos]
        _plot_linhas(ax, tab[tab["experimento"].isin(exps)], "f1", ylim=ylim)
        ax.set_title(f"{EXPERIMENTOS[exps[0]]['n_atributos']} atributos ({' → '.join(exps)})")
        ax.set_ylabel("F1-score (classe óbito)")
    fig.suptitle("Efeito do tamanho da amostra de treinamento ("
                 + " → ".join(_rotulo_tamanho(n) for n in tamanhos) + " registros)")
    _legenda_linhas(fig)
    salvar(fig, FIG_E9, "C3_efeito_tamanho_amostra_f1.png",
           rodape + "\nEixo y recortado na faixa dos dados, com a mesma escala nos dois painéis.")


def fig_c_custo(tab, rodape) -> None:
    """C4 — Custo computacional (treinamento, predição e total) de E8 e E9, em escala log."""
    par = list(PAR_CUSTO_C)
    piso = 0.05
    paineis = [("tempo_treino_s", "Treinamento"), ("tempo_predicao_s", "Predição (teste completo)"),
               ("tempo_total_s", "Total")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    for ax, (coluna, titulo) in zip(axes, paineis):
        dados = (tab[tab["experimento"].isin(par)][["experimento", "modelo", coluna]]
                 .rename(columns={coluna: "valor"}).copy())
        dados["plot"] = dados["valor"].clip(lower=piso)
        _barras_agrupadas(ax, dados, "modelo", "plot", "experimento", MODELOS, par, COR_C,
                          formato=_fmt_tempo, coluna_rotulo="valor")
        ax.set_yscale("log")
        ax.set_ylim(piso / 2, dados["plot"].max() * 4)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.set_ylabel("Tempo (s, escala logarítmica)")
        ax.grid(axis="x", visible=False)
    n = EXPERIMENTOS[par[0]]["n_treino"]
    fig.suptitle(f"Custo computacional com {_rotulo_tamanho(n)} registros de treinamento: {par[0]} × {par[1]}")
    _legenda(fig, {e: COR_C[e] for e in par}, "Experimento", {e: _nome_c(e) for e in par})
    salvar(fig, FIG_E9, "C4_custo_computacional_E8_E9.png",
           rodape + "\nEscala logarítmica: o comprimento das barras não é proporcional ao tempo; "
                    "use os valores rotulados. Tempos registrados com resolução de 0,1 s.")

# ───────────────────────────── PRINCIPAL ─────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Tabelas e gráficos dos experimentos E4–E9.")
    parser.add_argument("--analise", choices=["tamanho", "atributos", "e9", "todas"], default="tamanho",
                        help="tamanho (padrão): E6, E4, E7, E8 | atributos: E4 × E5 | "
                             "e9: E4, E5, E8, E9 | todas")
    args = parser.parse_args()

    try:
        bruto = carregar_execucoes()
        sel, avisos = selecionar_execucoes(bruto)
        validar(sel)
        avisos += validar_atributos(sel)
        grade = grade_c()
        matrizes = extrair_matrizes(sel)
        n_cura, n_obito = validar_matrizes(sel, matrizes)
    except ErroValidacao as erro:
        print(f"\nERRO DE VALIDAÇÃO:\n{erro}\n\nNenhuma tabela ou figura foi gerada.")
        return 1

    print(f"Validação concluída: {len(sel)} execuções principais com status 'ok' em {len(EXPERIMENTOS)} experimentos;")
    print("semente, tamanhos de treino/teste, atributos e amostras consistentes; matrizes conferem com as métricas.")
    print("Critério de seleção: por (experimento, algoritmo), execução 'ok' mais recente (data_hora).")
    for aviso in avisos:
        print(f"  aviso: {aviso}")

    tabela = montar_tabela(sel, matrizes)
    tab_tam = tabela_tamanho(tabela)
    tab_atr = tabela_atributos(tabela)
    print("\nTabela consolidada:")
    print(tabela.drop(columns=["hash_amostra", "data_hora", "arquivo_origem", "configuracao"]).round(4).to_string(index=False))
    print("\nTabelas:")
    exportar_tabela(tabela, "consolidado_experimentos.csv")
    exportar_tabela(tab_tam, "tabela_tamanho_amostra.csv")
    exportar_tabela(tab_atr, "tabela_E4_vs_E5.csv")
    exportar_tabela(tabela_par(tabela, "E8", "E9"), "tabela_E8_vs_E9.csv")      # diferença = E8 − E9
    exportar_tabela(tabela_par(tabela, "E9", "E5"), "tabela_E5_vs_E9.csv")      # diferença = E9 − E5
    exportar_tabela(tabela_comparativo(tabela), "tabela_comparativo_E4_E5_E8_E9.csv")

    configurar_estilo()
    semente = int(sel["semente"].iloc[0])
    n_teste = int(sel["n_teste"].iloc[0])
    cola_teste = f"Teste: {_milhar(n_teste)} registros ({_milhar(n_cura)} cura; {_milhar(n_obito)} óbito)."
    aviso_final = ("Precisão, recall e F1-score referem-se à classe óbito. "
                   "Execução única por experimento, sem teste de significância estatística.")

    if args.analise in ("tamanho", "todas"):
        tab_t = tabela[tabela["experimento"].isin(SERIE_TAMANHO)]
        n_atr = int(tab_t["n_atributos"].iloc[0])
        tamanhos = ", ".join(_milhar(n) for n in _tamanhos(tab_t))
        rodape = (f"Treino: {tamanhos} registros (amostras estratificadas e aninhadas; {n_atr} atributos; "
                  f"semente {semente}) · {cola_teste}\n{aviso_final}")
        print(f"\nAnálise A — tamanho da amostra ({' → '.join(SERIE_TAMANHO)}): figuras em "
              f"{FIG_AMOSTRAS.relative_to(BASE_DIR)}/")
        fig_tam_f1(tab_t, rodape)
        fig_tam_metricas(tab_t, rodape)
        fig_tam_tempos(tab_t, rodape)
        fig_tam_custo_total(tab_t, rodape)
        fig_tam_matriz_metricas(tab_t, rodape)
        fig_tam_matrizes_confusao(tab_t, matrizes, rodape)

    if args.analise in ("atributos", "todas"):
        tab_b = tabela[tabela["experimento"].isin(PAR_ATRIBUTOS)]
        n_tr = int(tab_b["n_treino"].iloc[0])
        amostra = tab_b["hash_amostra"].iloc[0]
        rodape = (f"Treino: {_milhar(n_tr)} registros estratificados (amostra {amostra}, semente {semente}) · "
                  f"{cola_teste}\n{aviso_final}")
        rodape_curto = f"Teste: {_milhar(n_teste)} registros · amostra {amostra} · semente {semente}."
        print(f"\nAnálise B — E4 × E5: figuras em {FIG_COMPARATIVOS.relative_to(BASE_DIR)}/")
        fig_atr_metricas(tab_b, rodape)
        fig_atr_comparacao(tab_b, rodape)
        fig_atr_matrizes(matrizes, rodape, rodape_curto)
        fig_atr_variacao(calcular_variacao(tab_b), rodape)
        fig_atr_custo(tab_b, rodape)
        fig_atr_precisao_recall(tab_b, rodape)

    if args.analise in ("e9", "todas"):
        tab_c = tabela[tabela["experimento"].isin(EXPERIMENTOS_C)]
        rodape = montar_rodape_c(tab_c, grade, n_cura, n_obito, semente)
        print(f"\nAnálise C — {', '.join(EXPERIMENTOS_C)}: figuras em {FIG_E9.relative_to(BASE_DIR)}/")
        fig_c_f1(tab_c, rodape)
        fig_c_remocao(tab_c, grade, rodape)
        fig_c_tamanho(tab_c, grade, rodape)
        fig_c_custo(tab_c[tab_c["experimento"].isin(PAR_CUSTO_C)], rodape)

    print(f"\nConcluído: 6 tabelas CSV + {len(GERADOS)} figuras PNG.")
    return 0


if __name__ == "__main__":
    sys.exit(main())