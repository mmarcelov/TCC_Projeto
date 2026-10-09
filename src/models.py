import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import shap
import matplotlib.pyplot as plt
import json
import time
import hashlib
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = BASE_DIR / "reports" / "resultados"
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_AMOSTRA = 50000

# Colunas codificadas como 0=Não, 1=Sim, 2=Ignorado
# (espelha colunas_sim_nao de data_loading.tratar_ausentes)
COLUNAS_TRINARIAS = [
    "CARDIOPATI", "HEMATOLOGI", "SIND_DOWN", "HEPATICA", "ASMA", "DIABETES",
    "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL", "OBESIDADE", "PUERPERA",
    "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
    "DIARREIA", "VOMITO", "DOR_ABD", "FADIGA", "PERD_OLFT", "PERD_PALA",
    "HOSPITAL", "UTI",
]

CAMPOS_REGISTRO = [
    "run_id", "data_hora", "experimento", "configuracao", "modelo", "semente",
    "hash_amostra", "n_treino", "n_teste", "n_atributos", "n_atributos_modelo",
    "preprocessamento", "parametros", "tempo_treino_s", "tempo_predicao_s",
    "acuracia", "precisao", "recall", "f1", "matriz_confusao", "status", "atributos",
]

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def carregar_dados_processados():
    """Carrega os arquivos de treino/teste já preparados na etapa anterior."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["TARGET"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["TARGET"]
    return X_train, X_test, y_train, y_test


def avaliar_modelo(nome: str, modelo, X_test, y_test, y_pred=None):
    """Calcula e imprime as métricas padrão de classificação para um modelo treinado."""
    # Se a predição já foi feita (ex.: em lotes, com tempo medido), reaproveita
    if y_pred is None:
        y_pred = modelo.predict(X_test)

    print(f"\n{'='*50}")
    print(f"Resultados: {nome}")
    print(f"{'='*50}")
    print(f"Acurácia:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão:  {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório completo:")
    print(classification_report(y_test, y_pred))

    return {
        "modelo": nome,
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def treinar_random_forest(X_train, y_train):
    """Treina Random Forest com pesos balanceados (compensa o desbalanceamento sem SMOTE)."""
    # class_weight='balanced' ajusta o peso de cada classe automaticamente
    # com base na proporção real (66,5%/33,5%) — substitui a necessidade de reamostragem
    modelo = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    return modelo


def treinar_xgboost(X_train, y_train):
    """Treina XGBoost com peso de classe ajustado."""
    # scale_pos_weight é o equivalente do XGBoost ao class_weight='balanced'
    peso = (y_train == 0).sum() / (y_train == 1).sum()
    modelo = XGBClassifier(
        n_estimators=200, max_depth=6, scale_pos_weight=peso,
        random_state=42, eval_metric="logloss"
    )
    modelo.fit(X_train, y_train)
    return modelo


def treinar_svm(X_train, y_train, amostra: int = 50000):
    """Treina SVM com pesos balanceados. Usa uma amostra do treino, pois SVM
    não escala bem para mais de algumas dezenas de milhares de registros."""
    # Com quase 1 milhão de linhas, treinar SVM no dataset completo seria
    # computacionalmente inviável (SVM tem custo quadrático/cúbico com o volume)
    X_amostra = X_train.sample(n=amostra, random_state=42)
    y_amostra = y_train.loc[X_amostra.index]

    modelo = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    modelo.fit(X_amostra, y_amostra)
    return modelo


def analisar_importancia(modelo, X_train, nome: str, usar_shap: bool = True):
    """Extrai a importância das variáveis: nativa do modelo + SHAP (mais robusto)."""
    importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
    importancias = importancias.sort_values(ascending=False)

    print(f"\nTop 15 variáveis mais importantes ({nome} - importância nativa):")
    print(importancias.head(15))

    if usar_shap:
        # Reduzido de 5000 para 1000 — RF com 200 árvores profundas é lento pro SHAP
        amostra = X_train.sample(n=1000, random_state=42)
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(amostra)

        plt.figure()
        shap.summary_plot(shap_values, amostra, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(f"reports/figures/shap_{nome.lower().replace(' ', '_')}.png", dpi=150)
        print(f"\nGráfico SHAP salvo em reports/figures/shap_{nome.lower().replace(' ', '_')}.png")


def remover_variaveis_gravidade(X: pd.DataFrame) -> pd.DataFrame:
    """Remove indicadores de gravidade/atendimento coletados DURANTE a internação
    (HOSPITAL, UTI, SUPORT_VEN), que representam vazamento conceitual de informação
    para um modelo de risco calculado no momento da admissão."""
    colunas_remover = ["HOSPITAL", "UTI"] + [c for c in X.columns if c.startswith("SUPORT_VEN_")]
    print(f"Removendo {len(colunas_remover)} colunas de gravidade/atendimento: {colunas_remover}")
    return X.drop(columns=colunas_remover)


def comparar_resultados(resultados: list):
    """Monta uma tabela comparativa entre versão completa e versão sem leakage."""
    df_comp = pd.DataFrame(resultados)
    print("\n" + "=" * 60)
    print("COMPARATIVO: modelo completo vs. sem variáveis de gravidade")
    print("=" * 60)
    print(df_comp.to_string(index=False))
    return df_comp

def amostrar_treino_estratificado(X_train, y_train, n: int = N_AMOSTRA):
    """Sorteia UMA amostra estratificada do treino, usada pelos três algoritmos."""
    X_amostra, _, y_amostra, _ = train_test_split(
        X_train, y_train, train_size=n, stratify=y_train, random_state=SEED
    )
    print(f"Amostra de treino: {X_amostra.shape[0]} registros de {X_train.shape[0]}")
    print("Proporção do alvo na amostra:")
    print(y_amostra.value_counts(normalize=True))
    return X_amostra, y_amostra


def calcular_hash_amostra(X_amostra) -> str:
    """Identificador da amostra (posições sorteadas no X_train.parquet)."""
    ids = ",".join(map(str, sorted(X_amostra.index)))
    return hashlib.sha256(ids.encode()).hexdigest()[:12]


def colunas_trinarias_presentes(X):
    """Colunas 0/1/2 existentes em X (a configuração sem gravidade não tem HOSPITAL/UTI)."""
    cols = [c for c in COLUNAS_TRINARIAS if c in X.columns]
    valores = set(np.unique(X[cols].to_numpy()))
    assert valores <= {0, 1, 2}, f"Valores inesperados nas colunas trinárias: {valores}"
    return cols


def treinar_svm_pipeline(X_treino, y_treino):
    """SVM RBF com pré-processamento próprio, ajustado somente com o treino:
    StandardScaler em IDADE_ANOS, OneHotEncoder nas colunas 0/1/2 e as demais
    colunas (já one-hot) sem alteração. Hiperparâmetros da SVM inalterados."""
    preprocessador = ColumnTransformer(
        transformers=[
            ("idade", StandardScaler(), ["IDADE_ANOS"]),
            ("trinarias",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             colunas_trinarias_presentes(X_treino)),
        ],
        remainder="passthrough",
    )
    modelo = Pipeline([
        ("preprocessamento", preprocessador),
        ("svm", SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
    ])
    modelo.fit(X_treino, y_treino)
    return modelo


def predizer_em_lotes(modelo, X, tamanho: int = 20000):
    """Predição em lotes, com progresso. O resultado é idêntico ao de predict() direto."""
    partes = []
    for i in range(0, len(X), tamanho):
        partes.append(modelo.predict(X.iloc[i:i + tamanho]))
        print(f"  predição: {min(i + tamanho, len(X))}/{len(X)}", end="\r")
    print()
    return np.concatenate(partes)


def estimar_custo_svm(X_tr, y_tr, X_te, n_treino: int = 10000, n_teste: int = 5000):
    """PILOTO DE CUSTO: mede tempos com subamostras e extrapola.
    Não usa y_test, não calcula métricas e não grava resultados."""
    X_p, _, y_p, _ = train_test_split(
        X_tr, y_tr, train_size=n_treino, stratify=y_tr, random_state=SEED
    )
    X_te_p = X_te.sample(n=n_teste, random_state=SEED)

    t0 = time.perf_counter()
    modelo = treinar_svm_pipeline(X_p, y_p)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    modelo.predict(X_te_p)
    t_pred = time.perf_counter() - t0

    f_tr = len(X_tr) / n_treino
    f_te = len(X_te) / n_teste
    print("\n" + "=" * 60)
    print("PILOTO DE CUSTO DA SVM — ESTIMATIVA, NÃO É RESULTADO EXPERIMENTAL")
    print("Nenhuma métrica é calculada; y_test não é utilizado.")
    print("=" * 60)
    print(f"Piloto: treino {n_treino} | predição em {n_teste} linhas do teste")
    print(f"Vetores de suporte no piloto: {int(modelo.named_steps['svm'].n_support_.sum())}")
    print(f"Tempo piloto -> treino: {t_fit:.1f} s | predição: {t_pred:.1f} s")
    print(f"Estimativa de treino com {len(X_tr)} registros: "
          f"{t_fit * f_tr**2 / 60:.1f} a {t_fit * f_tr**3 / 60:.1f} min (regra n² a n³)")
    print(f"Estimativa de predição em {len(X_te)} registros: "
          f"~{t_pred * f_te * f_tr / 60:.1f} min (regra ~ n_treino x n_teste)")


def parametros_do_modelo(modelo) -> str:
    """Parâmetros do modelo em JSON (para o registro do experimento)."""
    if isinstance(modelo, Pipeline):
        svm = modelo.named_steps["svm"]
        params = svm.get_params()
        # gamma='scale' é recalculado a partir dos dados: registra o valor efetivo
        params["gamma_efetivo"] = getattr(svm, "_gamma", None)
        params["n_vetores_suporte"] = int(svm.n_support_.sum())
    else:
        params = modelo.get_params()
    return json.dumps(params, default=str, sort_keys=True)


def n_atributos_apos_preprocessamento(modelo, X) -> int:
    """Nº de colunas que o estimador final de fato recebe."""
    if isinstance(modelo, Pipeline):
        return len(modelo[:-1].get_feature_names_out())
    return X.shape[1]


def salvar_registro_csv(registro: dict, caminho: Path):
    """Acrescenta UMA linha ao CSV (cria o cabeçalho se o arquivo ainda não existe).
    Gravar modelo a modelo preserva os resultados já obtidos se algo falhar depois."""
    pd.DataFrame([registro])[CAMPOS_REGISTRO].to_csv(
        caminho, mode="a", index=False, header=not caminho.exists()
    )


def executar_modelo(experimento, configuracao, nome, construtor, preprocessamento,
                    X_tr, y_tr, X_te, y_te, run_id, hash_amostra, caminho_csv):
    """Treina, prediz, avalia e registra UM modelo, de forma independente dos demais."""
    registro = dict.fromkeys(CAMPOS_REGISTRO)
    registro.update({
        "run_id": run_id,
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "experimento": experimento, "configuracao": configuracao, "modelo": nome,
        "semente": SEED, "hash_amostra": hash_amostra,
        "n_treino": X_tr.shape[0], "n_teste": X_te.shape[0],
        "n_atributos": X_tr.shape[1], "preprocessamento": preprocessamento,
        "atributos": "; ".join(X_tr.columns),
    })
    print(f"\n>>> {experimento} | {nome}: treinando com {X_tr.shape[0]} registros...")
    try:
        t0 = time.perf_counter()
        modelo = construtor(X_tr, y_tr)
        registro["tempo_treino_s"] = round(time.perf_counter() - t0, 1)
        print(f"    treino concluído em {registro['tempo_treino_s']} s")

        t0 = time.perf_counter()
        y_pred = predizer_em_lotes(modelo, X_te)
        registro["tempo_predicao_s"] = round(time.perf_counter() - t0, 1)

        metricas = avaliar_modelo(f"{nome} [{experimento}]", modelo, X_te, y_te, y_pred=y_pred)
        for k in ("acuracia", "precisao", "recall", "f1"):
            registro[k] = metricas[k]
        registro["matriz_confusao"] = json.dumps(confusion_matrix(y_te, y_pred).tolist())
        registro["parametros"] = parametros_do_modelo(modelo)
        registro["n_atributos_modelo"] = n_atributos_apos_preprocessamento(modelo, X_tr)
        registro["status"] = "ok"
    except Exception as erro:
        registro["status"] = f"erro: {type(erro).__name__}: {erro}"
        print(f"    FALHOU: {registro['status']}")

    salvar_registro_csv(registro, caminho_csv)   # grava já, antes do próximo modelo
    return registro


if __name__ == "__main__":
    RODAR_RF = False
    RODAR_XGB = False
    RODAR_SVM = False
    RODAR_IMPORTANCIA = False
    RODAR_ABLACAO = False

    RODAR_PILOTO_SVM = False
    RODAR_EQUIPARIDADE = True

    CONFIG_SEM_GRAVIDADE = True
    CONFIG_COMPLETA = False
    # ─────────────────────────────────────────────────────────

    X_train, X_test, y_train, y_test = carregar_dados_processados()
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

    resultados = []
    modelos_treinados = {}

    if RODAR_RF:
        rf = treinar_random_forest(X_train, y_train)
        resultados.append(avaliar_modelo("Random Forest (completo)", rf, X_test, y_test))
        modelos_treinados["Random Forest"] = rf

    if RODAR_XGB:
        xgb = treinar_xgboost(X_train, y_train)
        resultados.append(avaliar_modelo("XGBoost (completo)", xgb, X_test, y_test))
        modelos_treinados["XGBoost"] = xgb

    if RODAR_SVM:
        svm = treinar_svm(X_train, y_train)
        resultados.append(avaliar_modelo("SVM (completo)", svm, X_test, y_test))

    if RODAR_IMPORTANCIA:
        for nome, modelo in modelos_treinados.items():
            analisar_importancia(modelo, X_train, nome)

    if RODAR_ABLACAO:
        print("\n\n" + "#" * 60)
        print("ESTUDO DE ABLAÇÃO — removendo HOSPITAL, UTI, SUPORT_VEN")
        print("#" * 60)

        X_train_reduzido = remover_variaveis_gravidade(X_train)
        X_test_reduzido = remover_variaveis_gravidade(X_test)

        rf_reduzido = treinar_random_forest(X_train_reduzido, y_train)
        resultados.append(avaliar_modelo("Random Forest (sem gravidade)", rf_reduzido, X_test_reduzido, y_test))

        xgb_reduzido = treinar_xgboost(X_train_reduzido, y_train)
        resultados.append(avaliar_modelo("XGBoost (sem gravidade)", xgb_reduzido, X_test_reduzido, y_test))

        # Importância das variáveis na versão sem leakage — aqui sim os sintomas
        # e comorbidades devem aparecer com peso mais relevante
        analisar_importancia(rf_reduzido, X_train_reduzido, "Random Forest (sem gravidade)", usar_shap=False)
        analisar_importancia(xgb_reduzido, X_train_reduzido, "XGBoost (sem gravidade)", usar_shap=False)

        # SVM só na versão sem leakage, que é o resultado que efetivamente importa
        svm_reduzido = treinar_svm(X_train_reduzido, y_train)
        resultados.append(avaliar_modelo("SVM (sem gravidade)", svm_reduzido, X_test_reduzido, y_test))


    if RODAR_PILOTO_SVM or RODAR_EQUIPARIDADE:
        # Sorteio único: todos os modelos e configurações partem desta amostra
        X_amostra, y_amostra = amostrar_treino_estratificado(X_train, y_train)
        hash_amostra = calcular_hash_amostra(X_amostra)
        print(f"Hash da amostra: {hash_amostra} | semente: {SEED}")

        configs = {}
        if CONFIG_SEM_GRAVIDADE:
            configs["E5"] = ("sem_gravidade",
                             remover_variaveis_gravidade(X_amostra),
                             remover_variaveis_gravidade(X_test))
        if CONFIG_COMPLETA:
            configs["E4"] = ("completa", X_amostra, X_test)

        if RODAR_PILOTO_SVM:
            for exp, (cfg, X_tr, X_te) in configs.items():
                print(f"\n[Piloto — configuração {exp}: {cfg}]")
                estimar_custo_svm(X_tr, y_amostra, X_te)

        if RODAR_EQUIPARIDADE:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            modelos = [
                ("Random Forest", treinar_random_forest,
                 "nenhum (árvores não dependem de escala nem de distância)"),
                ("XGBoost", treinar_xgboost,
                 "nenhum (árvores não dependem de escala nem de distância)"),
                ("SVM", treinar_svm_pipeline,
                 "StandardScaler(IDADE_ANOS) + OneHotEncoder(colunas 0/1/2) + passthrough(dummies)"),
            ]
            for exp, (cfg, X_tr, X_te) in configs.items():
                assert X_tr.index.equals(X_amostra.index)
                caminho = RESULTADOS_DIR / f"equiparidade_{run_id}_{exp}.csv"
                print("\n" + "#" * 60)
                print(f"{exp} — {cfg} | treino {X_tr.shape[0]} | teste {X_te.shape[0]} "
                      f"| atributos {X_tr.shape[1]} | run_id {run_id}")
                print("#" * 60)
                for nome, construtor, prep in modelos:
                    executar_modelo(exp, cfg, nome, construtor, prep,
                                    X_tr, y_amostra, X_te, y_test,
                                    run_id, hash_amostra, caminho)

                print(f"\nResumo {exp} (arquivo: {caminho.name}):")
                print(pd.read_csv(caminho)[[
                    "experimento", "modelo", "n_treino", "n_teste", "tempo_treino_s",
                    "tempo_predicao_s", "acuracia", "precisao", "recall", "f1", "status",
                ]].to_string(index=False))

    if resultados:
        comparar_resultados(resultados)