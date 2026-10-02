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

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def carregar_dados_processados():
    """Carrega os arquivos de treino/teste já preparados na etapa anterior."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["TARGET"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["TARGET"]
    return X_train, X_test, y_train, y_test


def avaliar_modelo(nome: str, modelo, X_test, y_test):
    """Calcula e imprime as métricas padrão de classificação para um modelo treinado."""
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

if __name__ == "__main__":
    RODAR_RF = False              # já temos o resultado completo documentado
    RODAR_XGB = False             # idem
    RODAR_SVM = False             # não usado (SVM roda só dentro da ablação)
    RODAR_IMPORTANCIA = False     # já rodado
    RODAR_ABLACAO = True          # mantém True — é aqui que o SVM está
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

    comparar_resultados(resultados)