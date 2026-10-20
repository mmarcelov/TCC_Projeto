"""
experimento_E9.py — E9: 100.000 registros de treino, 45 atributos (configuração reduzida).

Reutiliza EXATAMENTE a amostra de treino do E8 (hash 0eb33db8bae4) e remove as mesmas seis
colunas do E5 com remover_variaveis_gravidade(). Não altera models.py nem
experimentos_amostras.py e não reexecuta E4–E8.

Uso (na raiz do repositório):
    python -u src/experimento_E9.py --verificar
    python -u src/experimento_E9.py --piloto [--piloto-comparativo]
    python -u src/experimento_E9.py --modelos rf xgb
    python -u src/experimento_E9.py --modelos svm
    python -u src/experimento_E9.py --conferir
    python -u src/experimento_E9.py --modelos svm --refazer     # só se quiser repetir
"""
import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from models import (
        RESULTADOS_DIR,
        calcular_hash_amostra,
        estimar_custo_svm,
        executar_modelo,
        remover_variaveis_gravidade,
    )
    from experimentos_amostras import (
        MODELOS,
        N_TESTE,
        SEMENTE,
        ErroAmostra,
        carregar_dados,
        construir_amostras,
        hash_teste,
        validar_amostras,
    )
except ImportError as erro:
    sys.exit(
        f"Não foi possível importar de src/models.py ou src/experimentos_amostras.py: {erro}\n"
        "Execute o script a partir da raiz do repositório: python -u src/experimento_E9.py ..."
    )

# ─────────────────────────── CONFIGURAÇÃO ───────────────────────────
EXPERIMENTO = "E9"
CONFIGURACAO = "sem_gravidade_100k"
N_TREINO = 100000
N_ATRIBUTOS = 45
HASH_E8 = "0eb33db8bae4"          # amostra de treino do E8 (e do E9, antes da remoção de colunas)
HASH_TESTE = "ea1d1b0329ce"       # conjunto de teste registrado anteriormente (51 atributos)
COLUNAS_REMOVIDAS = ["HOSPITAL", "UTI", "SUPORT_VEN_1", "SUPORT_VEN_2", "SUPORT_VEN_3", "SUPORT_VEN_9"]
TOLERANCIA = 1e-6

CSV_E9 = RESULTADOS_DIR / "amostras_E9_100000_45atributos.csv"
CSV_E8 = RESULTADOS_DIR / "amostras_E8_100000.csv"
CSV_E5 = RESULTADOS_DIR / "equiparidade_20261009_102402_E5.csv"

# Tempo da SVM no E8 informado para planejamento (usado só se o CSV do E8 não for encontrado)
REF_SVM_E8_INFORMADA = (998.8, 888.3)
# Na SVM estes valores dependem dos dados (gamma='scale'): diferenças são esperadas
CHAVES_SVM_DEPENDENTES_DOS_DADOS = {"gamma_efetivo", "n_vetores_suporte"}


# ───────────────────── PREPARAÇÃO E VERIFICAÇÕES ─────────────────────
def verificar_reducao(X_base, y_base, X_red, X_test, X_te_red) -> None:
    """Confere que a configuração reduzida é a do E5 e que registros e alvo não foram alterados."""
    problemas = []
    removidas = [c for c in X_base.columns if c not in X_red.columns]
    if sorted(removidas) != sorted(COLUNAS_REMOVIDAS):
        problemas.append(f"colunas removidas {removidas}; esperado {COLUNAS_REMOVIDAS}")
    if X_red.shape[1] != N_ATRIBUTOS:
        problemas.append(f"treino reduzido com {X_red.shape[1]} atributos; esperado {N_ATRIBUTOS}")
    if list(X_red.columns) != [c for c in X_base.columns if c not in removidas]:
        problemas.append("a ordem das colunas mantidas mudou")
    if list(X_red.columns) != list(X_te_red.columns):
        problemas.append("colunas de treino e teste reduzidos diferem (nomes ou ordem)")
    if len(X_te_red) != N_TESTE:
        problemas.append(f"teste reduzido com {len(X_te_red)} registros; esperado {N_TESTE}")
    if len(X_red) != N_TREINO:
        problemas.append(f"treino reduzido com {len(X_red)} registros; esperado {N_TREINO}")
    if not X_red.index.equals(X_base.index) or not X_red.index.equals(y_base.index):
        problemas.append("registros ou alinhamento com o alvo mudaram")
    if calcular_hash_amostra(X_red) != HASH_E8:
        problemas.append("hash da amostra reduzida difere do hash do E8")
    if not problemas:
        if not X_red.equals(X_base[list(X_red.columns)]):
            problemas.append("valores do treino mudaram em relação ao E8")
        if not X_te_red.equals(X_test[list(X_te_red.columns)]):
            problemas.append("valores do teste mudaram em relação ao teste original")
    if problemas:
        raise ErroAmostra("Configuração reduzida inválida: " + "; ".join(problemas))
    print(f"  Removidas exatamente as 6 colunas esperadas; restam {X_red.shape[1]} atributos.")
    print("  Colunas de treino e teste idênticas e na mesma ordem; registros e alvo inalterados.")
    print(f"  Treino reduzido {X_red.shape} | teste reduzido {X_te_red.shape}.")


def conferir_atributos_com_E5(X_red) -> None:
    """Compara nomes e ordem dos atributos com a lista registrada no CSV do E5."""
    if not CSV_E5.exists():
        print(f"  (aviso) {CSV_E5.name} não encontrado: lista de atributos não comparada com o E5.")
        return
    df = pd.read_csv(CSV_E5)
    df = df[df["status"] == "ok"]
    if df.empty:
        print("  (aviso) sem linha 'ok' no CSV do E5: lista de atributos não comparada.")
        return
    if set(df["atributos"]) != {"; ".join(X_red.columns)}:
        raise ErroAmostra("A lista de atributos do E9 difere da registrada no E5 (nomes ou ordem).")
    print("  Atributos idênticos (nomes e ordem) aos registrados no E5.")


def preparar_E9() -> dict:
    """Reconstrói a amostra do E8, confere hashes e aplica a configuração reduzida do E5."""
    X_train, X_test, y_train, y_test = carregar_dados()

    h_teste = hash_teste(X_test, y_test)
    if h_teste != HASH_TESTE:
        raise ErroAmostra(f"Hash do teste {h_teste} difere do registrado ({HASH_TESTE}).")
    print(f"Hash do conjunto de teste confere com o registrado: {h_teste}")

    amostras = construir_amostras(X_train, y_train)
    validar_amostras(amostras, X_train, y_train)   # as 4 amostras, ainda com 51 atributos

    X_e8, y_e8 = amostras["E8"]
    h8 = calcular_hash_amostra(X_e8)
    if h8 != HASH_E8:
        raise ErroAmostra(f"A amostra E8 reconstruída tem hash {h8}, mas o registrado é {HASH_E8}.")
    print(f"\nHash da amostra E8 reconstruída confere: {h8}")

    print("\nAplicando a configuração reduzida (mesma função usada no E5)...")
    X_red = remover_variaveis_gravidade(X_e8)
    X_te_red = remover_variaveis_gravidade(X_test)
    verificar_reducao(X_e8, y_e8, X_red, X_test, X_te_red)
    conferir_atributos_com_E5(X_red)
    print(f"  Proporção de óbitos na amostra: {y_e8.mean() * 100:.3f}% "
          f"(treino completo: {y_train.mean() * 100:.3f}%)")
    print(f"  Hash do conteúdo do treino reduzido: {hash_teste(X_red, y_e8)} "
          "(deve ser igual em todas as execuções)")
    return {"X_e8": X_e8, "y": y_e8, "X_red": X_red, "X_test": X_test,
            "X_te_red": X_te_red, "y_test": y_test}


# ───────────────────────── PILOTO DE CUSTO ─────────────────────────
def referencia_svm_e8():
    """Tempo real da SVM no E8 (51 atributos), só para planejamento."""
    if CSV_E8.exists():
        df = pd.read_csv(CSV_E8)
        df = df[(df["modelo"] == "SVM") & (df["status"] == "ok")]
        if not df.empty:
            l = df.iloc[-1]
            return float(l["tempo_treino_s"]), float(l["tempo_predicao_s"]), f"lido de {CSV_E8.name}"
    return (*REF_SVM_E8_INFORMADA, "valores informados no enunciado do E9")


def piloto_E9(ctx: dict, comparativo: bool) -> None:
    """Piloto de custo: não usa y_test, não calcula métricas e não grava resultados."""
    print("\n[Piloto — E9: 100.000 registros de treino planejados, 45 atributos]")
    estimar_custo_svm(ctx["X_red"], ctx["y"], ctx["X_te_red"])

    t_tr, t_pr, origem = referencia_svm_e8()
    print(f"\nReferência real da SVM no E8 ({origem}; 51 atributos, 100.000 registros):")
    print(f"  treino {t_tr:.1f} s ({t_tr / 60:.1f} min) | predição {t_pr:.1f} s ({t_pr / 60:.1f} min) "
          f"| total ~{(t_tr + t_pr) / 60:.1f} min")
    print("  Com 45 atributos o tempo pode ser menor, igual ou maior: use apenas para planejar.")

    if comparativo:
        print("\n[Piloto de comparação — E8 com 51 atributos, mesma subamostra e mesmo teste]")
        estimar_custo_svm(ctx["X_e8"], ctx["y"], ctx["X_test"])
        print("\nCompare os dois blocos de piloto (tempos e vetores de suporte), feitos na mesma máquina.")
    print("\nPiloto concluído: são estimativas de custo, NÃO resultados experimentais. Nada foi gravado.")


# ───────────────────────── EXECUÇÃO DOS MODELOS ─────────────────────────
def modelos_concluidos_E9(caminho, hash_amostra: str) -> set:
    """Modelos já concluídos ('ok') no CSV do E9. Interrompe se o arquivo for incoerente."""
    if not caminho.exists():
        return set()
    df = pd.read_csv(caminho, dtype={"hash_amostra": str})
    esperado = {"experimento": EXPERIMENTO, "hash_amostra": hash_amostra, "semente": SEMENTE,
                "n_treino": N_TREINO, "n_teste": N_TESTE, "n_atributos": N_ATRIBUTOS}
    for coluna, valor in esperado.items():
        if set(df[coluna].astype(str)) != {str(valor)}:
            raise ErroAmostra(
                f"{caminho.name}: coluna '{coluna}' diverge do esperado ({valor}). "
                "Misturar execuções diferentes no mesmo arquivo invalidaria a comparação."
            )
    return set(df.loc[df["status"] == "ok", "modelo"])


def executar_E9(chaves, refazer: bool, ctx: dict) -> None:
    X_tr, y_tr, X_te, y_te = ctx["X_red"], ctx["y"], ctx["X_te_red"], ctx["y_test"]
    hash_amostra = calcular_hash_amostra(X_tr)
    concluidos = modelos_concluidos_E9(CSV_E9, hash_amostra)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "#" * 60)
    print(f"{EXPERIMENTO} — {CONFIGURACAO} | treino {len(X_tr)} | teste {len(X_te)} "
          f"| atributos {X_tr.shape[1]}")
    print(f"hash da amostra {hash_amostra} | semente {SEMENTE} | run_id {run_id}")
    print(f"arquivo de resultados: {CSV_E9.name}")
    print("#" * 60)

    for chave in chaves:
        nome, construtor, prep = MODELOS[chave]
        if nome in concluidos and not refazer:
            print(f"\n>>> {EXPERIMENTO} | {nome}: já concluído em {CSV_E9.name} — pulando "
                  "(use --refazer para repetir).")
            continue
        executar_modelo(EXPERIMENTO, CONFIGURACAO, nome, construtor, prep,
                        X_tr, y_tr, X_te, y_te, run_id, hash_amostra, CSV_E9)

    if CSV_E9.exists():
        df = pd.read_csv(CSV_E9, dtype={"hash_amostra": str})
        ok = df[df["status"] == "ok"].drop_duplicates("modelo", keep="last")
        pend = df[~df["modelo"].isin(ok["modelo"])].drop_duplicates("modelo", keep="last")
        resumo = pd.concat([ok, pend])
        print(f"\nResumo {EXPERIMENTO} (última execução 'ok' de cada modelo):")
        print(resumo[["experimento", "modelo", "n_treino", "n_teste", "n_atributos",
                      "tempo_treino_s", "tempo_predicao_s", "acuracia", "precisao",
                      "recall", "f1", "status"]].to_string(index=False))


# ───────────────────────── CONFERÊNCIA DOS RESULTADOS ─────────────────────────
def _ler(caminho):
    return pd.read_csv(caminho, dtype={"hash_amostra": str})


def _ultimas_ok(df):
    return df[df["status"] == "ok"].drop_duplicates("modelo", keep="last").set_index("modelo")

def _valores_iguais(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
    return a == b


def conferir_resultados() -> int:
    """Confere o CSV do E9 (sem carregar os dados): campos fixos, matrizes × métricas,
    atributos e hiperparâmetros em relação ao E8. Somente leitura."""
    if not CSV_E9.exists():
        print(f"Arquivo não encontrado: {CSV_E9}\n  Nada a conferir ainda.")
        return 1
    e9 = _ler(CSV_E9)
    problemas, avisos = [], []

    fixos = {"experimento": EXPERIMENTO, "semente": SEMENTE, "hash_amostra": HASH_E8,
             "n_treino": N_TREINO, "n_teste": N_TESTE, "n_atributos": N_ATRIBUTOS}
    for coluna, valor in fixos.items():
        achados = set(e9[coluna].astype(str))
        if achados != {str(valor)}:
            problemas.append(f"coluna '{coluna}': esperado {valor}, encontrado {sorted(achados)}")

    ok9 = _ultimas_ok(e9)
    nomes = [n for n, _, _ in MODELOS.values()]
    pendentes = [n for n in nomes if n not in ok9.index]
    n_erro = int((e9["status"] != "ok").sum())
    print(f"Linhas no arquivo: {len(e9)} (ok: {len(e9) - n_erro}; com erro: {n_erro})")
    if pendentes:
        print(f"Modelos ainda sem execução 'ok': {pendentes}")

    ok8, attr_esperados = None, None
    if CSV_E8.exists():
        ok8 = _ultimas_ok(_ler(CSV_E8))
        if ok8.empty:
            avisos.append(f"{CSV_E8.name} sem execuções 'ok': comparação com o E8 ignorada.")
            ok8 = None
        else:
            if set(ok8["hash_amostra"]) != {HASH_E8}:
                problemas.append(f"hash_amostra do E8 difere de {HASH_E8}")
            base = ok8["atributos"].iloc[0].split("; ")
            attr_esperados = [c for c in base if c not in COLUNAS_REMOVIDAS]
    else:
        avisos.append(f"{CSV_E8.name} não encontrado: comparação com o E8 ignorada.")

    for modelo, l in ok9.iterrows():
        try:
            m = np.array(json.loads(l["matriz_confusao"]), dtype=int)
            if m.shape != (2, 2):
                raise ValueError(f"formato {m.shape}")
        except (TypeError, ValueError) as erro:
            problemas.append(f"{modelo}: matriz_confusao ilegível ({erro})")
            continue
        tn, fp, fn, tp = (int(v) for v in m.ravel())
        total = tn + fp + fn + tp
        if total != N_TESTE:
            problemas.append(f"{modelo}: matriz soma {total}; esperado {N_TESTE}")
        recalculado = {
            "acuracia": (tn + tp) / total,
            "precisao": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        }
        for k, v in recalculado.items():
            if abs(v - l[k]) > TOLERANCIA:
                problemas.append(f"{modelo}: {k} do CSV = {l[k]:.6f}, mas a matriz resulta em {v:.6f}")
        if attr_esperados is not None and l["atributos"].split("; ") != attr_esperados:
            problemas.append(f"{modelo}: lista de atributos difere da esperada (atributos do E8 sem as 6 colunas)")

        if ok8 is not None and modelo in ok8.index:
            m8 = np.array(json.loads(ok8.loc[modelo, "matriz_confusao"]), dtype=int)
            if (m8.sum(axis=1) != m.sum(axis=1)).any():
                problemas.append(f"{modelo}: totais por classe real diferem dos do E8 (teste diferente?)")
            p9, p8 = json.loads(l["parametros"]), json.loads(ok8.loc[modelo, "parametros"])
            ignorar = CHAVES_SVM_DEPENDENTES_DOS_DADOS if modelo == "SVM" else set()
            dif = sorted(k for k in set(p8) | set(p9) if k not in ignorar and not _valores_iguais(p8.get(k), p9.get(k)))
            if dif:
                problemas.append(f"{modelo}: hiperparâmetros diferem do E8 em {dif}")
            else:
                print(f"  {modelo}: hiperparâmetros idênticos aos do E8"
                      + (" (exceto valores que dependem dos dados)" if ignorar else ""))
            if modelo == "SVM":
                print(f"  SVM — gamma efetivo: E8 = {p8.get('gamma_efetivo')} | E9 = {p9.get('gamma_efetivo')}; "
                      f"vetores de suporte: E8 = {p8.get('n_vetores_suporte')} | E9 = {p9.get('n_vetores_suporte')}")

    if len(ok9):
        print("\nResultados do E9 (última execução 'ok' de cada modelo):")
        print(ok9.reset_index()[["modelo", "n_treino", "n_teste", "n_atributos", "n_atributos_modelo",
                                 "tempo_treino_s", "tempo_predicao_s", "acuracia", "precisao",
                                 "recall", "f1"]].round(4).to_string(index=False))
    for aviso in avisos:
        print(f"  aviso: {aviso}")
    if problemas:
        print("\nINCONSISTÊNCIAS ENCONTRADAS:")
        for p in problemas:
            print(f"  - {p}")
        return 1
    print("\nConferência concluída: sem inconsistências"
          + (" (execução incompleta: faltam modelos)." if pendentes else "."))
    return 0


# ───────────────────────────── PRINCIPAL ─────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="E9: 100.000 registros de treino, 45 atributos.")
    parser.add_argument("--verificar", action="store_true",
                        help="reconstrói e valida os dados e a amostra, sem treinar nem gravar")
    parser.add_argument("--piloto", action="store_true",
                        help="estima o custo da SVM (sem resultados experimentais)")
    parser.add_argument("--piloto-comparativo", action="store_true",
                        help="com --piloto: repete o piloto também com os 51 atributos do E8")
    parser.add_argument("--modelos", nargs="+", choices=list(MODELOS),
                        help="treina e registra os modelos indicados (rf xgb svm)")
    parser.add_argument("--refazer", action="store_true",
                        help="com --modelos: repete modelos já concluídos (acrescenta nova linha)")
    parser.add_argument("--conferir", action="store_true",
                        help="confere o CSV do E9 (somente leitura, sem carregar os dados)")
    args = parser.parse_args()

    acoes = [args.verificar, args.piloto, bool(args.modelos), args.conferir]
    if sum(acoes) != 1:
        parser.error("escolha exatamente uma ação: --verificar, --piloto, --modelos ... ou --conferir.")
    if args.refazer and not args.modelos:
        parser.error("--refazer só vale junto com --modelos.")
    if args.piloto_comparativo and not args.piloto:
        parser.error("--piloto-comparativo só vale junto com --piloto.")

    if args.conferir:
        return conferir_resultados()

    try:
        ctx = preparar_E9()
    except ErroAmostra as erro:
        print(f"\nERRO CRÍTICO:\n  {erro}\n\nNada foi treinado nem gravado.")
        return 1
    print("\nTodas as verificações críticas passaram.")

    if args.verificar:
        print("Verificação concluída. Nada foi treinado nem gravado.")
        return 0
    if args.piloto:
        piloto_E9(ctx, args.piloto_comparativo)
        return 0
    try:
        executar_E9(args.modelos, args.refazer, ctx)
    except ErroAmostra as erro:
        print(f"\nERRO:\n  {erro}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())