"""
experimentos_amostras.py — E6, E7 e E8: efeito do tamanho da amostra de treino.

Reaproveita as funções de src/models.py (sem alterá-las). Mantém os 51 atributos,
o mesmo conjunto de teste (229.114), os mesmos hiperparâmetros e a semente 42.
Amostras aninhadas:  E6 (25 mil) ⊂ E4 (50 mil) ⊂ E7 (75 mil) ⊂ E8 (100 mil).

Uso (na raiz do repositório):
    python -u src/experimentos_amostras.py --verificar
    python -u src/experimentos_amostras.py --piloto --exp E6
    python -u src/experimentos_amostras.py --exp E6 --modelos rf xgb
    python -u src/experimentos_amostras.py --exp E6 --modelos svm
"""
import argparse
import hashlib
import sys
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from models import (
        SEED,
        RESULTADOS_DIR,
        amostrar_treino_estratificado,
        calcular_hash_amostra,
        carregar_dados_processados,
        estimar_custo_svm,
        executar_modelo,
        treinar_random_forest,
        treinar_svm_pipeline,
        treinar_xgboost,
    )
except ImportError as erro:
    sys.exit(
        f"Não foi possível importar de src/models.py: {erro}\n"
        "Confira se models.py contém as funções da etapa de equiparidade "
        "(executar_modelo, treinar_svm_pipeline, estimar_custo_svm etc.)."
    )

# ─────────────────────────── CONFIGURAÇÃO ───────────────────────────
SEMENTE = 42
HASH_E4 = "ee971fa098f9"          # hash registrado da amostra de 50 mil do E4
N_TREINO_TOTAL = 916455           # treino completo (data/processed/X_train.parquet)
N_TESTE = 229114                  # teste original, igual para todos os experimentos
N_ATRIBUTOS = 51                  # configuração completa (HOSPITAL, UTI e SUPORT_VEN incluídos)
TOLERANCIA_PROPORCAO = 0.002      # diferença máxima na proporção de óbitos (0,2 p.p.)

EXPERIMENTOS = {"E6": 25000, "E7": 75000, "E8": 100000}   # nº de registros de treino

# Referência de custo da SVM no E4 (apenas leitura, para estimar o custo)
CSV_E4 = RESULTADOS_DIR / "equiparidade_20261009_112030_E4.csv"

PREP_ARVORES = "nenhum (árvores não dependem de escala nem de distância)"
PREP_SVM = "StandardScaler(IDADE_ANOS) + OneHotEncoder(colunas 0/1/2) + passthrough(dummies)"
MODELOS = {   # chave da linha de comando -> (nome registrado, construtor, pré-processamento)
    "rf": ("Random Forest", treinar_random_forest, PREP_ARVORES),
    "xgb": ("XGBoost", treinar_xgboost, PREP_ARVORES),
    "svm": ("SVM", treinar_svm_pipeline, PREP_SVM),
}


class ErroAmostra(Exception):
    """Falha na carga dos dados, na construção ou na verificação das amostras."""


# ───────────────────── DADOS E CONJUNTO DE TESTE ─────────────────────
def hash_teste(X_test, y_test) -> str:
    """Identificador do conteúdo do conjunto de teste (para comparar entre execuções)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X_test, index=False).to_numpy().tobytes())
    h.update(pd.util.hash_pandas_object(y_test, index=False).to_numpy().tobytes())
    return h.hexdigest()[:12]


def carregar_dados():
    """Carrega os dados já processados (sem reprocessar nada) e confere os tamanhos."""
    if SEED != SEMENTE:
        raise ErroAmostra(f"SEED em models.py = {SEED}, mas os experimentos exigem {SEMENTE}.")
    X_train, X_test, y_train, y_test = carregar_dados_processados()

    problemas = []
    if X_train.shape != (N_TREINO_TOTAL, N_ATRIBUTOS):
        problemas.append(f"X_train {X_train.shape}; esperado ({N_TREINO_TOTAL}, {N_ATRIBUTOS})")
    if X_test.shape != (N_TESTE, N_ATRIBUTOS):
        problemas.append(f"X_test {X_test.shape}; esperado ({N_TESTE}, {N_ATRIBUTOS})")
    if list(X_train.columns) != list(X_test.columns):
        problemas.append("colunas de treino e teste diferentes")
    if problemas:
        raise ErroAmostra("Dados processados inesperados: " + "; ".join(problemas))

    print(f"Treino completo: {X_train.shape} | Teste: {X_test.shape}")
    print(f"Hash do conjunto de teste: {hash_teste(X_test, y_test)} "
          "(deve ser o mesmo em todas as execuções)")
    return X_train, X_test, y_train, y_test


# ───────────────────── CONSTRUÇÃO DAS AMOSTRAS ─────────────────────
def _expandir_amostra(X_train, y_train, X_base, y_base, n_extra: int):
    """Mantém a amostra-base e acrescenta n_extra registros estratificados,
    sorteados apenas entre os registros do treino que NÃO estão na base."""
    resto = X_train.index.difference(X_base.index)
    X_resto, y_resto = X_train.loc[resto], y_train.loc[resto]
    X_extra, _, y_extra, _ = train_test_split(
        X_resto, y_resto, train_size=n_extra, stratify=y_resto, random_state=SEMENTE
    )
    return pd.concat([X_base, X_extra]), pd.concat([y_base, y_extra])


def construir_amostras(X_train, y_train) -> dict:
    """Devolve {'E6','E4','E7','E8': (X, y)} com E6 ⊂ E4 ⊂ E7 ⊂ E8."""
    print("\nReconstruindo a amostra do E4 (50.000) com a função original de models.py...")
    X50, y50 = amostrar_treino_estratificado(X_train, y_train, 50000)
    h50 = calcular_hash_amostra(X50)
    if h50 != HASH_E4:
        raise ErroAmostra(
            f"A amostra de 50 mil reconstruída tem hash {h50}, mas o E4 registrou {HASH_E4}.\n"
            "  Sem reproduzir a amostra do E4 não é possível garantir o aninhamento. "
            "Confira se X_train.parquet e a versão do scikit-learn são os mesmos do E4."
        )
    print(f"Hash da amostra de 50 mil confere com o E4: {h50}")

    X25, _, y25, _ = train_test_split(
        X50, y50, train_size=25000, stratify=y50, random_state=SEMENTE
    )
    X75, y75 = _expandir_amostra(X_train, y_train, X50, y50, 25000)
    X100, y100 = _expandir_amostra(X_train, y_train, X75, y75, 25000)
    return {"E6": (X25, y25), "E4": (X50, y50), "E7": (X75, y75), "E8": (X100, y100)}


def verificar_amostra(nome, n_esperado, X, y, X_train, y_train) -> None:
    """Confere tamanho, duplicatas, alinhamento, origem, atributos e proporção de classes."""
    problemas = []
    if len(X) != n_esperado or len(y) != n_esperado:
        problemas.append(f"tamanho X={len(X)}, y={len(y)}; esperado {n_esperado}")
    if not X.index.is_unique:
        problemas.append("há registros duplicados (índices repetidos)")
    if not X.index.equals(y.index):
        problemas.append("X e y não estão alinhados")
    if not X.index.isin(X_train.index).all():
        problemas.append("há registros que não pertencem ao treino original")
    if list(X.columns) != list(X_train.columns) or X.shape[1] != N_ATRIBUTOS:
        problemas.append(f"atributos diferentes dos {N_ATRIBUTOS} originais")

    if not problemas:  # só compara conteúdo se os índices estão íntegros
        if not (y_train.loc[X.index].to_numpy() == y.to_numpy()).all():
            problemas.append("rótulos diferentes dos rótulos originais")
        if not X.equals(X_train.loc[X.index]):
            problemas.append("atributos diferentes das linhas originais do treino")
        dif = abs(y.mean() - y_train.mean())
        if dif > TOLERANCIA_PROPORCAO:
            problemas.append(
                f"proporção de óbitos {y.mean():.4f} difere do treino ({y_train.mean():.4f}) "
                f"em {dif * 100:.3f} p.p. (tolerância {TOLERANCIA_PROPORCAO * 100:.1f} p.p.)"
            )
    if problemas:
        raise ErroAmostra(f"Amostra {nome}: " + "; ".join(problemas))


def verificar_aninhamento(amostras: dict) -> None:
    """Confere E6 ⊂ E4 ⊂ E7 ⊂ E8."""
    ordem = ["E6", "E4", "E7", "E8"]
    for menor, maior in zip(ordem[:-1], ordem[1:]):
        if not amostras[menor][0].index.isin(amostras[maior][0].index).all():
            raise ErroAmostra(f"{menor} não está contido em {maior}.")
        print(f"  {menor} ⊂ {maior}: ok")


def validar_amostras(amostras: dict, X_train, y_train) -> None:
    """Executa todas as verificações e imprime o resumo das amostras."""
    tamanhos = {"E4": 50000, **EXPERIMENTOS}
    print("\nVerificando as amostras...")
    for nome, (X, y) in amostras.items():
        verificar_amostra(nome, tamanhos[nome], X, y, X_train, y_train)
        print(f"  {nome}: {len(X)} registros, sem duplicatas, alinhados, 51 atributos, proporção ok")
    print("Aninhamento:")
    verificar_aninhamento(amostras)

    linhas = []
    for nome, (X, y) in amostras.items():
        linhas.append({
            "amostra": nome, "n": len(X), "hash": calcular_hash_amostra(X),
            "n_cura": int((y == 0).sum()), "n_obito": int((y == 1).sum()),
            "pct_obito": round(y.mean() * 100, 3),
            "dif_vs_treino_pp": round((y.mean() - y_train.mean()) * 100, 4),
        })
    print("\nResumo das amostras (E6 ⊂ E4 ⊂ E7 ⊂ E8):")
    print(pd.DataFrame(linhas).sort_values("n").to_string(index=False))
    print(f"Proporção de óbitos no treino completo: {y_train.mean() * 100:.3f}%")


# ───────────────────────── PILOTO DE CUSTO ─────────────────────────
def estimar_custo_ancorado(n_alvo: int) -> None:
    """Estimativa a partir do tempo REAL da SVM no E4 (lido do CSV, só leitura).
    Treino: entre n² e n³ do fator de aumento; predição: ~linear no nº de treino."""
    if not CSV_E4.exists():
        print(f"(Referência do E4 indisponível: {CSV_E4.name} não encontrado.)")
        return
    ref = pd.read_csv(CSV_E4)
    ref = ref[(ref["modelo"] == "SVM") & (ref["status"] == "ok")]
    if ref.empty:
        print("(Sem linha 'ok' da SVM no CSV do E4; estimativa ancorada ignorada.)")
        return
    linha = ref.iloc[-1]
    t_tr, t_pr, n_ref = float(linha["tempo_treino_s"]), float(linha["tempo_predicao_s"]), int(linha["n_treino"])
    f = n_alvo / n_ref
    print(f"\nEstimativa ancorada no E4 (valores lidos de {CSV_E4.name}):")
    print(f"  SVM no E4 ({n_ref} registros): treino {t_tr / 60:.1f} min | predição {t_pr / 60:.1f} min")
    print(f"  Para {n_alvo} registros (fator {f:.2f}): "
          f"treino {t_tr * f**2 / 60:.1f} a {t_tr * f**3 / 60:.1f} min | "
          f"predição ~{t_pr * f / 60:.1f} min")
    print("  Regra empírica: use apenas como ordem de grandeza.")


def piloto(exp: str, X_tr, y_tr, X_te) -> None:
    """Piloto de custo: não usa y_test, não calcula métricas e não grava resultados."""
    print(f"\n[Piloto — {exp}: {len(X_tr)} registros de treino planejados]")
    estimar_custo_svm(X_tr, y_tr, X_te)
    estimar_custo_ancorado(len(X_tr))


# ───────────────────────── EXECUÇÃO DOS MODELOS ─────────────────────────
def modelos_concluidos(caminho, hash_amostra: str) -> set:
    """Modelos já concluídos ('ok') no CSV do experimento. Interrompe se a amostra mudou."""
    if not caminho.exists():
        return set()
    df = pd.read_csv(caminho, dtype={"hash_amostra": str})
    if (df["hash_amostra"] != hash_amostra).any():
        raise ErroAmostra(
            f"{caminho.name} tem hash_amostra diferente do atual ({hash_amostra}). "
            "Misturar amostras diferentes no mesmo arquivo invalidaria a comparação."
        )
    return set(df.loc[df["status"] == "ok", "modelo"])


def executar_experimento(exp, chaves_modelos, refazer, X_tr, y_tr, X_te, y_te) -> None:
    n = len(X_tr)
    hash_amostra = calcular_hash_amostra(X_tr)
    caminho = RESULTADOS_DIR / f"amostras_{exp}_{n}.csv"   # um arquivo por experimento
    concluidos = modelos_concluidos(caminho, hash_amostra)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    configuracao = f"completa_{n // 1000}k"

    print("\n" + "#" * 60)
    print(f"{exp} — {configuracao} | treino {n} | teste {len(X_te)} | atributos {X_tr.shape[1]}")
    print(f"hash da amostra {hash_amostra} | semente {SEMENTE} | run_id {run_id}")
    print(f"arquivo de resultados: {caminho.name}")
    print("#" * 60)

    for chave in chaves_modelos:
        nome, construtor, prep = MODELOS[chave]
        if nome in concluidos and not refazer:
            print(f"\n>>> {exp} | {nome}: já concluído em {caminho.name} — pulando "
                  "(use --refazer para repetir).")
            continue
        executar_modelo(exp, configuracao, nome, construtor, prep,
                        X_tr, y_tr, X_te, y_te, run_id, hash_amostra, caminho)

    if caminho.exists():
        res = pd.read_csv(caminho, dtype={"hash_amostra": str}).drop_duplicates("modelo", keep="last")
        print(f"\nResumo {exp} (última execução de cada modelo):")
        print(res[["experimento", "modelo", "n_treino", "n_teste", "n_atributos",
                   "tempo_treino_s", "tempo_predicao_s", "acuracia", "precisao",
                   "recall", "f1", "status"]].to_string(index=False))


# ───────────────────────────── PRINCIPAL ─────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="E6, E7 e E8: efeito do tamanho da amostra de treino.")
    parser.add_argument("--exp", choices=list(EXPERIMENTOS), help="experimento a executar")
    parser.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS),
                        help="modelos a executar (padrão: rf xgb svm)")
    parser.add_argument("--piloto", action="store_true",
                        help="apenas estima o custo da SVM (sem resultados experimentais)")
    parser.add_argument("--verificar", action="store_true",
                        help="apenas constrói e valida as amostras (sem treinar)")
    parser.add_argument("--refazer", action="store_true",
                        help="repete modelos já concluídos (acrescenta nova linha ao CSV)")
    args = parser.parse_args()
    if not (args.exp or args.piloto or args.verificar):
        parser.error("informe --exp E6|E7|E8, --piloto ou --verificar.")

    try:
        X_train, X_test, y_train, y_test = carregar_dados()
        amostras = construir_amostras(X_train, y_train)
        validar_amostras(amostras, X_train, y_train)
    except ErroAmostra as erro:
        print(f"\nERRO:\n  {erro}\n\nNada foi treinado nem gravado.")
        return 1

    if args.verificar:
        print("\nVerificação concluída. Nada foi treinado nem gravado.")
        return 0

    alvos = [args.exp] if args.exp else list(EXPERIMENTOS)   # --piloto sem --exp: os três
    if args.piloto:
        for exp in alvos:
            piloto(exp, *amostras[exp], X_test)
        print("\nPiloto concluído: são estimativas de custo, NÃO resultados experimentais.")
        return 0

    try:
        for exp in alvos:
            X_tr, y_tr = amostras[exp]
            executar_experimento(exp, args.modelos, args.refazer, X_tr, y_tr, X_test, y_test)
    except ErroAmostra as erro:
        print(f"\nERRO:\n  {erro}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())