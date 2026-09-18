import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Caminho para a pasta onde estão os arquivos .parquet brutos do SRAG
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def carregar_srag(ano: str) -> pd.DataFrame:
    """Carrega o parquet do SRAG de um ano específico (ex: '20', '21', '22')."""
    # Monta o caminho completo do arquivo a partir do ano informado
    arquivo = RAW_DIR / f"INFLUD{ano}-23-03-2026.parquet"
    # pd.read_parquet já lê o formato binário e devolve um DataFrame pronto
    df = pd.read_parquet(arquivo)
    return df


def inspecionar(df: pd.DataFrame):
    """Mostra uma visão geral do dataframe: tamanho, colunas e valores ausentes."""
    print(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas\n")

    print("Colunas disponíveis:")
    print(list(df.columns))

    # isna() marca True onde tem valor ausente; .mean() por coluna vira o percentual
    print("\n% de valores ausentes por coluna (top 20):")
    ausentes = (df.isna().mean() * 100).sort_values(ascending=False)
    print(ausentes.head(20))


def checar_alvo_e_classificacao(df: pd.DataFrame):
    """Verifica a distribuição da variável-alvo (EVOLUCAO) e do filtro de doença (CLASSI_FIN)."""
    print("Distribuição de EVOLUCAO (desfecho clínico):")
    # value_counts(dropna=False) mostra também quantos registros estão vazios (NaN)
    print(df["EVOLUCAO"].value_counts(dropna=False))
    print(f"\n% ausente em EVOLUCAO: {df['EVOLUCAO'].isna().mean()*100:.2f}%\n")

    print("Distribuição de CLASSI_FIN (classificação final do caso):")
    print(df["CLASSI_FIN"].value_counts(dropna=False))


def filtrar_covid(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só os casos classificados como COVID-19 (CLASSI_FIN == 5)."""
    print(f"dtype de CLASSI_FIN: {df['CLASSI_FIN'].dtype}")
    print(f"Valores únicos: {df['CLASSI_FIN'].unique()}\n")

    # astype(str) converte pra texto antes de comparar, evitando erro de tipo
    # (a coluna veio como Decimal, então comparar direto com o número 5 poderia falhar)
    df_covid = df[df["CLASSI_FIN"].astype(str) == "5"].copy()
    print(f"Casos de COVID-19 filtrados: {df_covid.shape[0]} de {df.shape[0]} "
          f"({df_covid.shape[0]/df.shape[0]*100:.1f}%)\n")

    print("Distribuição de EVOLUCAO só entre casos de COVID-19:")
    print(df_covid["EVOLUCAO"].value_counts(dropna=False))

    return df_covid


def preparar_dataset_final(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra COVID, define o alvo binário (cura/óbito) e seleciona as colunas do modelo."""
    df_covid = filtrar_covid(df)

    # Alvo binário: mantém só cura (1) e óbito por COVID (2), descarta óbito por
    # outra causa (3) e ignorado/ausente (9/NaN), que não servem pro classificador
    df_final = df_covid[df_covid["EVOLUCAO"].astype(str).isin(["1", "2"])].copy()
    # TARGET = 1 quando é óbito, 0 quando é cura
    df_final["TARGET"] = (df_final["EVOLUCAO"].astype(str) == "2").astype(int)

    print(f"Registros para modelagem: {df_final.shape[0]}")
    print(df_final["TARGET"].value_counts(normalize=True) * 100)

    # Lista das colunas que de fato entram no modelo — organizadas por grupo
    # pra facilitar leitura e futura inclusão/remoção de variáveis
    colunas_selecionadas = [
        # Demografia
        "CS_SEXO", "NU_IDADE_N", "TP_IDADE", "CS_RACA", "CS_ESCOL_N", "CS_ZONA",
        # Comorbidades
        "CARDIOPATI", "HEMATOLOGI", "SIND_DOWN", "HEPATICA", "ASMA", "DIABETES",
        "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL", "OBESIDADE", "PUERPERA",
        # Sintomas
        "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
        "DIARREIA", "VOMITO", "DOR_ABD", "FADIGA", "PERD_OLFT", "PERD_PALA",
        # Indicadores de gravidade/atendimento
        "HOSPITAL", "UTI", "SUPORT_VEN",
        # Alvo
        "TARGET",
    ]
    df_final = df_final[colunas_selecionadas]

    print("\n% de ausentes nas colunas selecionadas:")
    print((df_final.isna().mean() * 100).sort_values(ascending=False))

    return df_final


def tratar_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Recodifica colunas Sim/Não/Ignorado (1/2/9) e categóricas multi-valor,
    tratando NaN como 'Ignorado' em ambos os casos."""
    df = df.copy()

    # Adicionei SATURACAO aqui — segue o mesmo padrão 1/2/9 dos sintomas
    colunas_sim_nao = [
        "CARDIOPATI", "HEMATOLOGI", "SIND_DOWN", "HEPATICA", "ASMA", "DIABETES",
        "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL", "OBESIDADE", "PUERPERA",
        "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
        "DIARREIA", "VOMITO", "DOR_ABD", "FADIGA", "PERD_OLFT", "PERD_PALA",
        "HOSPITAL", "UTI",
    ]

    mapeamento = {"1": 1, "1.0": 1, "2": 0, "2.0": 0, "9": 2, "9.0": 2}

    for col in colunas_sim_nao:
        serie_mapeada = df[col].astype(str).map(mapeamento)
        df[col] = serie_mapeada.fillna(2).astype(int)

    # Colunas categóricas com mais de 2 categorias: mantém os códigos originais,
    # só une o NaN (nulo de verdade, qualquer que seja sua representação interna)
    # à categoria "Ignorado" (9), que já existe no dicionário
    colunas_categoricas_multi = ["CS_ESCOL_N", "SUPORT_VEN", "CS_ZONA"]
    for col in colunas_categoricas_multi:
        # fillna age sobre o nulo real, então funciona independente do tipo interno da coluna
        df[col] = df[col].fillna("9").astype(str)
        # Rede de segurança: caso alguma representação textual de nulo tenha sobrado
        df[col] = df[col].replace({"nan": "9", "None": "9", "<NA>": "9"})

    print("Após tratamento, contagem de categorias (exemplo CARDIOPATI):")
    print(df["CARDIOPATI"].value_counts())

    print("\nExemplo CS_ESCOL_N após tratamento:")
    print(df["CS_ESCOL_N"].value_counts())

    print("\n% de ausentes restantes no dataset (deve ser tudo 0 agora):")
    print((df.isna().mean() * 100).sort_values(ascending=False).head(10))

    return df


def padronizar_idade(df: pd.DataFrame) -> pd.DataFrame:
    """Converte NU_IDADE_N + TP_IDADE numa única coluna de idade em anos."""
    df = df.copy()

    idade = pd.to_numeric(df["NU_IDADE_N"].astype(str), errors="coerce")
    tipo = df["TP_IDADE"].astype(str)

    # .astype(float) aqui é essencial: sem isso, idade_anos fica como int64
    # (porque idade não tem valores ausentes) e não aceita os resultados
    # decimais da divisão por 365 ou 12 — dava o erro LossySetitemError
    idade_anos = idade.astype(float)
    idade_anos[tipo == "1"] = idade[tipo == "1"] / 365   # dias -> anos
    idade_anos[tipo == "2"] = idade[tipo == "2"] / 12    # meses -> anos
    # tipo == "3" já está em anos, não precisa alterar

    df["IDADE_ANOS"] = idade_anos.round(1)
    df = df.drop(columns=["NU_IDADE_N", "TP_IDADE"])

    print("Estatísticas de IDADE_ANOS após padronização:")
    print(df["IDADE_ANOS"].describe())

    return df


def remover_idades_invalidas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove registros com idade fisicamente impossível (erro de digitação na base)."""
    antes = df.shape[0]

    # Faixa plausível: 0 a 120 anos. Fora disso é erro de notificação, não dado real
    df = df[(df["IDADE_ANOS"] >= 0) & (df["IDADE_ANOS"] <= 120)].copy()

    depois = df.shape[0]
    print(f"Registros removidos por idade inválida: {antes - depois} ({(antes-depois)/antes*100:.3f}%)")
    print(f"Registros restantes: {depois}")

    print("\nEstatísticas de IDADE_ANOS após limpeza:")
    print(df["IDADE_ANOS"].describe())

    return df


def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding nas variáveis categóricas de múltiplas classes."""
    df = df.copy()

    # Essas colunas têm mais de 2 categorias sem ordem natural entre elas
    # (não é como "gravidade" que vai de leve a grave — aqui não há hierarquia),
    # então one-hot encoding é mais indicado que atribuir um número direto
    colunas_categoricas = ["CS_SEXO", "CS_RACA", "CS_ESCOL_N", "CS_ZONA", "SUPORT_VEN"]
    for col in colunas_categoricas:
        # get_dummies precisa de texto, não de número, pra gerar os nomes das colunas certo
        df[col] = df[col].astype(str)

    # Cada categoria vira uma coluna própria (0 ou 1) — ex: CS_SEXO_M, CS_SEXO_F
    df = pd.get_dummies(df, columns=colunas_categoricas, prefix=colunas_categoricas)

    print(f"\nShape após encoding: {df.shape}")
    print("Colunas geradas (parcial):")
    print(list(df.columns)[:15], "...")

    return df


def dividir_treino_teste(df: pd.DataFrame):
    """Separa treino (80%) e teste (20%), mantendo a proporção de TARGET (stratify)."""
    # X = todas as variáveis explicativas; y = só o alvo que queremos prever
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # stratify=y garante que treino e teste mantenham a mesma proporção de
    # cura/óbito do dataset original — sem isso, o split poderia por acaso
    # concentrar mais óbitos de um lado só, distorcendo a avaliação
    # random_state=42 fixa a aleatoriedade, pra o resultado ser reproduzível
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTreino: {X_train.shape[0]} registros | Teste: {X_test.shape[0]} registros")
    print("Proporção do alvo no treino:")
    print(y_train.value_counts(normalize=True))
    print("Proporção do alvo no teste:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    RODAR_INSPECAO = False
    RODAR_CHECAGEM_ALVO = False
    RODAR_FILTRO_COVID = False
    RODAR_DATASET_FINAL = False
    RODAR_TRATAR_AUSENTES = False
    RODAR_PIPELINE_COMPLETO = True
    # ─────────────────────────────────────────────────────────

    df_2021 = carregar_srag("21")

    if RODAR_INSPECAO:
        inspecionar(df_2021)
    if RODAR_CHECAGEM_ALVO:
        checar_alvo_e_classificacao(df_2021)
    if RODAR_FILTRO_COVID:
        filtrar_covid(df_2021)
    if RODAR_DATASET_FINAL:
        df_modelagem = preparar_dataset_final(df_2021)
        df_modelagem.to_parquet("data/processed/srag_covid_2021.parquet", index=False)
    if RODAR_TRATAR_AUSENTES:
        df_modelagem = preparar_dataset_final(df_2021)
        df_tratado = tratar_ausentes(df_modelagem)
        df_tratado.to_parquet("data/processed/srag_covid_2021_tratado.parquet", index=False)

    if RODAR_PIPELINE_COMPLETO:
        df_modelagem = preparar_dataset_final(df_2021)
        df_tratado = tratar_ausentes(df_modelagem)
        df_idade = padronizar_idade(df_tratado)
        df_idade = remover_idades_invalidas(df_idade)
        df_encoded = codificar_categoricas(df_idade)
        X_train, X_test, y_train, y_test = dividir_treino_teste(df_encoded)

        X_train.to_parquet("data/processed/X_train.parquet", index=False)
        X_test.to_parquet("data/processed/X_test.parquet", index=False)
        y_train.to_frame().to_parquet("data/processed/y_train.parquet", index=False)
        y_test.to_frame().to_parquet("data/processed/y_test.parquet", index=False)
        print("\nArquivos de treino/teste salvos em data/processed/")