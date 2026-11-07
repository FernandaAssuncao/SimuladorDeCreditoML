
# Simulador de Crédito com Machine Learning

Aplicação desktop desenvolvida em Python para simular análises de solicitações de empréstimo utilizando Machine Learning.

O projeto utiliza um modelo de classificação baseado em Random Forest para estimar o risco de inadimplência de uma solicitação, considerando características financeiras e o histórico de crédito do solicitante.

## Objetivo

Desenvolver uma aplicação que utilize Machine Learning para auxiliar na análise de risco de crédito, demonstrando etapas de preparação dos dados, treinamento, avaliação e utilização de um modelo preditivo.

## Tecnologias

- Python
- Scikit-learn
- Pandas
- NumPy
- CustomTkinter
- Joblib

## Funcionalidades

- Cadastro de propostas de empréstimo.
- Análise de solicitações utilizando Machine Learning.
- Probabilidade estimada de inadimplência.
- Histórico das solicitações analisadas.
- Persistência e carregamento do modelo treinado.

## Dataset

Os dados utilizados foram obtidos no Kaggle, por meio de um conjunto de dados de risco de crédito.

O modelo utiliza características como:

- Idade do solicitante (`person_age`).
- Renda anual (`person_income`).
- Valor do empréstimo (`loan_amnt`).
- Proporção entre empréstimo e renda (`loan_percent_income`).
- Histórico de inadimplência (`cb_person_default_on_file`).

**Variável-alvo:** `loan_status`.

## Desenvolvimento do modelo

O projeto utiliza o algoritmo Random Forest Classifier, implementado com Scikit-learn.

O desenvolvimento envolve:

- Preparação e processamento dos dados.
- Separação entre variáveis preditoras e variável-alvo.
- Treinamento do modelo.
- Avaliação utilizando métricas de classificação.
- Otimização de hiperparâmetros com GridSearchCV.
- Análise de diferentes thresholds de classificação.

## Avaliação do modelo

**Modelo original:**

| Métrica | Resultado |
|---|---:|
| Accuracy | 83,69% |
| Precision | 77,00% |
| Recall | 49,00% |
| F1-score | 60,00% |

A avaliação considera diferentes métricas para analisar o desempenho do classificador, especialmente sua capacidade de identificar solicitações com risco de inadimplência.

## Ajuste de Threshold

Foram realizados experimentos com diferentes limiares de classificação para analisar o impacto nas métricas de Precision e Recall.

Essa etapa permite investigar como diferentes critérios de decisão afetam a identificação de possíveis inadimplentes.

## Persistência do modelo

O modelo treinado é salvo utilizando Joblib, permitindo seu carregamento pela aplicação desktop sem a necessidade de realizar um novo treinamento a cada execução.

## Demonstração da interface

| Empréstimo Reprovado | Empréstimo Aprovado |
|---|---|
| ![Empréstimo Reprovado](caminho-da-imagem-reprovado) | ![Empréstimo Aprovado](caminho-da-imagem-aprovado) |

## Como executar

1. Clone o repositório.
2. Instale as dependências necessárias.
3. Execute o arquivo principal da aplicação.

## Dataset

Os dados utilizados no projeto foram obtidos no Kaggle:

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

É necessario baixar e criar uma pasta com o nome data na raiz do projeto colocando o dataset dentro dela. 
## Observação

Projeto desenvolvido para fins de estudo e demonstração de conceitos de Machine Learning. Os dados utilizados são provenientes do Kaggle.
 |