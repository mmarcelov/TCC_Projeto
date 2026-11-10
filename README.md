# TCC_Projeto

Projeto de Trabalho de Conclusão de Curso voltado à análise de dados de SRAG (Síndrome Respiratória Aguda Grave), utilizando algoritmos de aprendizado de máquina para realização de experimentos de classificação.

O foco deste repositório é o experimento final **E9**, com os experimentos E4 a E8 mantidos como referência para comparação.

## Requisitos

- Python 3.14.7
- Git
- Base de dados SRAG 2021 em formato Parquet

## Instalação

Clone o repositório:

```bash
git clone https://github.com/mmarcelov/TCC_Projeto.git
cd TCC_Projeto
```

Crie e ative um ambiente virtual:

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Preparação dos dados

A base de dados não está incluída neste repositório devido ao seu tamanho. É necessário baixá-la separadamente pelo portal de dados abertos do Ministério da Saúde:

[Base de dados SRAG – DATASUS](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026)

Utilize a base de **2021 em formato Parquet**, correspondente ao arquivo `INFLUD21-23-03-2026.parquet`.

Após o download, coloque o arquivo no seguinte caminho:

```text
data/raw/INFLUD21-23-03-2026.parquet
```

Com a base disponível, execute o processamento dos dados:

```bash
python src/data_loading.py
```

Esse procedimento realiza a filtragem e preparação dos dados, gerando os arquivos necessários para os experimentos em `data/processed/`.

## Execução dos experimentos

O experimento final E9 pode ser executado por meio do arquivo:

```bash
python src/experimento_E9.py
```

O script permite selecionar os modelos individualmente ou em conjunto:

```bash
python -u src/experimento_E9.py --modelos rf xgb
```

```bash
python -u src/experimento_E9.py --modelos svm
```

Os modelos disponíveis são:

- `rf`: Random Forest
- `xgb`: XGBoost
- `svm`: Support Vector Machine

Também existem opções para verificação da amostra, execução de piloto e conferência dos resultados:

```bash
python -u src/experimento_E9.py --verificar
python -u src/experimento_E9.py --piloto
python -u src/experimento_E9.py --conferir
```

A SVM pode demandar maior tempo de processamento, dependendo do ambiente utilizado.

## Resultados

Os resultados e gráficos gerados pelos experimentos são armazenados separadamente:

- `reports/resultados/`: resultados dos experimentos.
- `reports/figures/`: gráficos gerados pelo projeto.

## Observações

- Os comandos devem ser executados a partir da pasta raiz do repositório.
- A base de dados deve ser baixada e posicionada no caminho indicado antes da execução do processamento.
- Atualizações na base disponibilizada pelo DATASUS podem gerar diferenças nos resultados obtidos.
- O arquivo `requirements.txt` reúne as dependências utilizadas no projeto.
