import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

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


if __name__ == "__main__":
    RODAR_RF = True
    RODAR_XGB = True
    RODAR_SVM = False  # deixar False por enquanto — é o mais lento, roda por último
    # ─────────────────────────────────────────────────────────

    X_train, X_test, y_train, y_test = carregar_dados_processados()
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

    resultados = []

    if RODAR_RF:
        rf = treinar_random_forest(X_train, y_train)
        resultados.append(avaliar_modelo("Random Forest", rf, X_test, y_test))

    if RODAR_XGB:
        xgb = treinar_xgboost(X_train, y_train)
        resultados.append(avaliar_modelo("XGBoost", xgb, X_test, y_test))

    if RODAR_SVM:
        svm = treinar_svm(X_train, y_train)
        resultados.append(avaliar_modelo("SVM", svm, X_test, y_test))

    print("\n\nResumo comparativo:")
    print(pd.DataFrame(resultados))