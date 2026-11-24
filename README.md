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

Os dados utilizados foram obtidos no Kaggle, por meio do [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

O modelo utiliza as seguintes características:

- Idade do solicitante (`person_age`).
- Renda anual (`person_income`).
- Valor do empréstimo (`loan_amnt`).
- Proporção entre empréstimo e renda (`loan_percent_income`).
- Histórico de inadimplência (`cb_person_default_on_file`).

**Variável-alvo:** `loan_status`.

A classe `1` representa solicitações com risco de inadimplência, enquanto a classe `0` representa solicitações sem esse risco.

Para executar o projeto, é necessário baixar o dataset, criar uma pasta chamada `data` na raiz do projeto e colocar o arquivo CSV dentro dela.

## Desenvolvimento do modelo

O projeto utiliza o algoritmo Random Forest Classifier, implementado com Scikit-learn.

O desenvolvimento envolve:

- Preparação e limpeza dos dados.
- Separação entre variáveis preditoras e variável-alvo.
- Codificação de variáveis categóricas.
- Treinamento e avaliação do modelo.
- Análise de métricas de classificação.
- Otimização experimental de hiperparâmetros com GridSearchCV.
- Análise de diferentes thresholds de classificação.

## Modelo integrado à aplicação

Atualmente, a aplicação utiliza um Random Forest Classifier com os seguintes parâmetros:

- `n_estimators=100`
- `max_depth=10`
- `random_state=42`

O modelo é treinado utilizando os dados preparados e salvo com Joblib. Quando a aplicação é executada novamente, o arquivo salvo é carregado, evitando a necessidade de um novo treinamento.

### Avaliação do modelo original

Resultados registrados durante o desenvolvimento:

| Métrica | Resultado |
|---|---:|
| Accuracy | 83,69% |
| Precision | 77,00% |
| Recall | 49,00% |
| F1-score | 60,00% |

## Experimentos de otimização

Além do modelo integrado, foram realizados experimentos para investigar melhorias na identificação de solicitações com risco de inadimplência.

### 1. Pipeline e pré-processamento

Foi desenvolvido um Pipeline utilizando `ColumnTransformer` e `OrdinalEncoder` para automatizar o pré-processamento das variáveis categóricas e o treinamento do modelo.

A variável `cb_person_default_on_file` foi codificada numericamente, enquanto as demais variáveis foram preservadas.

### 2. Otimização de hiperparâmetros

Foi utilizado `GridSearchCV` para testar diferentes combinações de hiperparâmetros do Random Forest.

Os parâmetros avaliados foram:

- `n_estimators`: 100 e 200.
- `max_depth`: 10, 20 e None.
- `min_samples_leaf`: 1 e 2.

A busca utilizou validação cruzada com 3 divisões (`cv=3`) e Recall como métrica de seleção.

### 3. Balanceamento de classes

Foi realizado um experimento utilizando `class_weight='balanced'` para atribuir pesos maiores à classe minoritária durante o treinamento.

O objetivo foi melhorar a identificação de clientes com risco de inadimplência sem remover exemplos da classe majoritária.

### 4. Ajuste de threshold

Foram avaliados diferentes thresholds utilizando as probabilidades produzidas pelo modelo no conjunto de validação.

O critério adotado foi buscar Precision mínima de 65%, maximizando o Recall entre os thresholds que atendessem a essa condição.

No experimento com o Pipeline balanceado, o threshold selecionado foi 0,60.

## Resultados dos experimentos

A comparação entre o Pipeline sem balanceamento e o Pipeline balanceado apresentou os seguintes resultados no conjunto de teste:

| Métrica | Pipeline sem balanceamento | Pipeline balanceado |
|---|---:|---:|
| Threshold | 0,45 | 0,60 |
| Precision | 63% | 65% |
| Recall | 53% | 55% |
| F1-score | 58% | 60% |
| Accuracy | 83% | 84% |

O Pipeline balanceado apresentou resultados ligeiramente superiores nas métricas avaliadas, identificando mais clientes com risco de inadimplência e reduzindo a quantidade de falsos positivos.

**Melhor resultado experimental:** Pipeline com `class_weight='balanced'` e threshold de 0,60.

Esses resultados representam experimentos realizados durante o desenvolvimento e não correspondem ao modelo atualmente integrado à aplicação.

**Observação metodológica:** o conjunto de teste já havia sido consultado em avaliações exploratórias anteriores. Portanto, os resultados devem ser interpretados como uma comparação experimental, e não como uma estimativa final totalmente independente de desempenho.

## Persistência do modelo

O modelo atualmente integrado à aplicação é salvo utilizando Joblib no caminho `models/modelo_sistema_financeiro.joblib`.

Isso permite seu carregamento pela aplicação desktop sem a necessidade de realizar um novo treinamento a cada execução.

## Demonstração da interface

| Empréstimo Reprovado | Empréstimo Aprovado |
|---|---|
| ![Empréstimo Reprovado](caminho-da-imagem-reprovado) | ![Empréstimo Aprovado](caminho-da-imagem-aprovado) |

## Como executar

1. Clone o repositório.
2. Instale as dependências necessárias.
3. Baixe o dataset e coloque o arquivo CSV na pasta `data`.
4. Execute o arquivo principal da aplicação.

## Observação

Projeto desenvolvido para fins de estudo e demonstração de conceitos de Machine Learning. Os dados utilizados são provenientes do Kaggle.