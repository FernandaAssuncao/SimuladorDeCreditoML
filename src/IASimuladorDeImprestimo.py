import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.inspection import permutation_importance
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np


class IAFinanceira:
    nome_arquivo = 'models/modelo_sistema_financeiro.joblib'

    colunas_x = ['person_age', 'person_income', 'loan_amnt', 'loan_percent_income', 'cb_person_default_on_file']

    def __init__(self):
        self.__modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.__treinada = False
        self.verificar()
        self.testar_modelo()

    def __treinar_ia(self):
        df = pd.read_csv('data/credit_risk_dataset.csv')

        colunas_selecionadas = [
            'person_age',
            'person_income',
            'loan_amnt',
            'loan_percent_income',
            'cb_person_default_on_file',
            'loan_status'
        ]
        df_filtrado = df[colunas_selecionadas].copy()

        # Removendo valores nulos
        df_filtrado = df_filtrado.dropna()

        # 'Y' (Yes) vira 1 (Histórico de inadimplência/Nome sujo)
        # 'N' (No) vira 0 (Nome limpo no histórico)
        df_filtrado['cb_person_default_on_file'] = df_filtrado['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

        x = df_filtrado[
            ['person_age', 'person_income', 'loan_amnt', 'loan_percent_income', 'cb_person_default_on_file']]
        y = df_filtrado['loan_status']

        x_treino, x_test, y_treino, y_test = train_test_split(x, y,
                                                              test_size=0.2,
                                                              random_state=42,
                                                              stratify=y)

        self.__modelo.fit(x_treino, y_treino)

        previsoes = self.__modelo.predict(x_test)
        accuracy = accuracy_score(y_test, previsoes)
        matriz = confusion_matrix(y_test, previsoes)
        precision = precision_score(y_test, previsoes)
        recall = recall_score(y_test, previsoes)
        classification = classification_report(y_test, previsoes)
        f1 = f1_score(y_test, previsoes)
        print(f'Accuracy do modelo: {accuracy:.4f}%')
        print(f'Matriz confusion modelo: {matriz}%')
        print(f'Precision: {precision:.4f}%')
        print(f'Recall: {recall:.4f}%')
        print(f'F1 score: {f1:.4f}%')
        print(f'Classification report: {classification}')

        self.__salvar_modelo()

    def prever(self, idade:int, salario:float, valor:float, porcentual_renda:float, nome_limpo:int):
        dados_clientes = pd.DataFrame([[idade, salario, valor, porcentual_renda, nome_limpo]], columns=self.colunas_x)
        resultado = self.__modelo.predict(dados_clientes)
        return resultado[0] == 0 #Se for 0 retorna true (aprovado).

    def __salvar_modelo(self):
        joblib.dump(self.__modelo, self.nome_arquivo)

    def verificar(self):
        if os.path.exists(self.nome_arquivo):
            self.__modelo = joblib.load(self.nome_arquivo)
            self.__treinada = True
            print('Modelo carregado com sucesso!')
        else:
            self.__treinar_ia()
            self.__salvar_modelo()
            print('Modelo treinado e salvo com sucesso!')

    def calcular_probabilidade_aprovado(self, idade:int, salario:float, valor:float, porcentual_renda:float,nome_limpo:int):
        dados_cliente = pd.DataFrame([[idade, salario, valor, porcentual_renda, nome_limpo]], columns=self.colunas_x)
        resultado = self.__modelo.predict_proba(dados_cliente)
        return resultado[0][0]

    def calcular_probabilidade_reprovado(self, idade:int, salario:float, valor:float, porcentual_renda:float, nome_limpo:int):
        dados_cliente = pd.DataFrame([[idade, salario, valor, porcentual_renda, nome_limpo]], columns=self.colunas_x)
        resultado = self.__modelo.predict_proba(dados_cliente)
        return resultado[0][1]

    def testar_modelo(self):
        df = pd.read_csv('data/credit_risk_dataset.csv')
        colunas_selecionadas = [
            'person_age',
            'person_income',
            'loan_amnt',
            'loan_percent_income',
            'cb_person_default_on_file',
            'loan_status'
        ]
        df_filtrado = df[colunas_selecionadas].copy()
        df_filtrado = df_filtrado.dropna()
        df_filtrado['cb_person_default_on_file'] = df_filtrado['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
        x = df_filtrado[['person_age', 'person_income', 'loan_amnt', 'loan_percent_income', 'cb_person_default_on_file']]
        y = df_filtrado['loan_status']
        #self.__testando_grid(x, y)
        self.__testando_grid_threshold(x, y)
        x_treino, x_test, y_treino, y_test = train_test_split(x, y,
                                                              test_size=0.2,
                                                              random_state=42,
                                                              stratify=y)
        modelo = RandomForestClassifier(n_estimators=100, max_depth=10,
                                        random_state=42, class_weight='balanced')
        modelos = {'Modelo': self.__modelo, 'Modelo classe': modelo}
        resultados = []
        for nome, m in modelos.items():
            if nome == 'Modelo classe':
                m.fit(x_treino, y_treino)
            print(f'{nome}')
            previsoes = m.predict(x_test)
            probabilidades = m.predict_proba(x_test)
            if nome == 'Modelo':
                probabilidade_classe_1 = probabilidades[:, 1]
                for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
                    previsoes_thansbord = (probabilidade_classe_1 >= threshold).astype(int)
                    resultados.append({
                        'Threshold': threshold,
                        'precision': precision_score(y_test, previsoes_thansbord),
                        'recall': recall_score(y_test, previsoes_thansbord),
                        'f1': f1_score(y_test, previsoes_thansbord),
                    })
                precision, recall, thresholds = precision_recall_curve(y_test,probabilidade_classe_1)
                plt.plot(recall, precision)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision Recall Curve')
                plt.show()
                print(f'precision: {precision}')
                print(f'recall: {recall}')
                print(f'thresholds: {thresholds}')
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
                        precision[:-1] + recall[:-1]
                )
                melhor_f1_scores = np.argmax(f1_scores)
                melhor_threshold = thresholds[melhor_f1_scores]
                melhor_f1 = f1_scores[melhor_f1_scores]
                print(f'Melhor Threshold: {melhor_threshold}')
                print(f'Melhor F1 Score: {melhor_f1}')
                previsoes_classe_threshold = (probabilidade_classe_1 >= melhor_threshold).astype(int)
                print(f'Matriz confusion: {confusion_matrix(y_test, previsoes_classe_threshold)}')
                print(f'Classification report: {classification_report(y_test, previsoes_classe_threshold)}')
                fpr, tpr, thresholds_roc = roc_curve(y_test, probabilidade_classe_1)
                auc = roc_auc_score(y_test, probabilidade_classe_1)
                print(f'FPR: {fpr[:10]}')
                print(f'TPR: {tpr[:10]}')
                print(f'Thresholds: {thresholds_roc[:10]}')
                print(f'AUC: {auc}')
                score = cross_val_score(self.__modelo, x, y, cv=5, scoring='accuracy')
                print(f'Score: {score}')
                print(f'Média: {score.mean()}')
                resultadoss = cross_validate(self.__modelo,
                                             x,
                                             y,
                                             cv=5,
                                             scoring=[
                                                 'accuracy',
                                                 'precision',
                                                 'recall',
                                                 'f1',
                                                 'roc_auc'
                                             ])
                print('Accuracy média:', resultadoss['test_accuracy'].mean())
                print(f'Medía precision: {resultadoss["test_precision"].mean()}')
                print(f'Medía recall: {resultadoss["test_recall"].mean()}')
                print(f'Medía F1 score: {resultadoss["test_f1"].mean()}')
                print(f'Medía roc_auc: {resultadoss["test_roc_auc"].mean()}')
            #print(f'Accuracy: {accuracy_score(y_test, previsoes):.4f}%')
            #print(f'Matrix de confusão: {confusion_matrix(y_test, previsoes)}')
            #print(f'Precision score: {precision_score(y_test, previsoes)}')
            #print(f'Recall score: {recall_score(y_test, previsoes)}')
            #print(f'F1 score: {f1_score(y_test, previsoes)}')
            #print(f'Classification report: {classification_report(y_test, previsoes)}')
            print('=' * 30)

    def __testando_grid(self, x, y):
        parametros = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
        x_treino, x_test, y_treino, y_test = train_test_split(x,
                                                              y,
                                                              test_size=0.2,
                                                              random_state=42,
                                                              stratify=y)

        grid = GridSearchCV(
            estimator=self.__modelo,
            param_grid=parametros,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )

        grid.fit(x_treino, y_treino)

        print(f'Melhores parametros: {grid.best_params_}')
        print(f'Melhor f1: {grid.best_score_}')

        #resultados_grid = pd.DataFrame(grid.cv_results_)
        #print(resultados_grid[
        #['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score',ascending=False))

        melhor_modelo = grid.best_estimator_
        previsoes = melhor_modelo.predict(x_test)
        probabilidades = melhor_modelo.predict_proba(x_test)[:,1]
        print(f'Accuracy {accuracy_score(y_test, previsoes)}')
        print(f'Precision {precision_score(y_test, previsoes)}')
        print(f'Recall {recall_score(y_test, previsoes)}')
        print(f'F1 {f1_score(y_test, previsoes)}')
        print(f'ROC-AUC {roc_auc_score(y_test, probabilidades)}')

        print('Matriz confusion')
        print(confusion_matrix(y_test, previsoes))

        print('Classification report')
        print(classification_report(y_test, previsoes))

        for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
            previsoes_threshold = (probabilidades >= threshold).astype(int)
            precision = precision_score(y_test, previsoes_threshold)
            recall = recall_score(y_test, previsoes_threshold)
            f1 = f1_score(y_test, previsoes_threshold)
            print(f'Threshold: {threshold}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1: {f1:.4f}')

    def __testando_grid_threshold(self, x, y):
        parametros = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }

        x_desenvolvimento, x_test, y_desenvolvimento, y_test = train_test_split(x,y,
                                                                                test_size=0.2,
                                                                                random_state=42,
                                                                                stratify=y)

        x_treino, x_validacao, y_treino, y_validacao = train_test_split(x_desenvolvimento,
                                                                        y_desenvolvimento,
                                                                        test_size=0.2,
                                                                        random_state=42,
                                                                        stratify=y_desenvolvimento)

        grid = GridSearchCV(
            estimator=self.__modelo,
            param_grid=parametros,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid.fit(x_treino, y_treino)

        print(f'Melhores parametros: {grid.best_params_}')
        print(f'Melhor f1: {grid.best_score_}')

        melhor_modelo = grid.best_estimator_

        resultadoss = permutation_importance(
            melhor_modelo,
            x_test,
            y_test,
            scoring='f1',
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        for colunaa, importanciaa in zip(x.columns, resultadoss.importances_mean):
            print(f'{colunaa} : {importanciaa:.4f}')

        importance = melhor_modelo.feature_importances_
        for coluna, importancia in zip(x.columns, importance):
            print(f'{coluna} : {importancia:.4f}')

        probabilidades_validacao = melhor_modelo.predict_proba(x_validacao)[:,1]

        melhor_threshold = 0
        melhor_f1 = 0
        for threshold in [0.50, 0.48, 0.46, 0.44, 0.42, 0.40,
                  0.38, 0.36, 0.34, 0.32, 0.30]:
            previsoes_validacao = (probabilidades_validacao >= threshold).astype(int)
            precision_validacao = precision_score(y_validacao, previsoes_validacao)
            recall_validacao = recall_score(y_validacao, previsoes_validacao)
            f1_validacao = f1_score(y_validacao, previsoes_validacao)
            matriz = confusion_matrix(
                y_validacao,
                previsoes_validacao
            )
            print(f'Threshold: {threshold}')
            print(f'Precision: {precision_validacao:.4f}')
            print(f'Recall: {recall_validacao:.4f}')
            print(f'F1: {f1_validacao:.4f}')
            print(f'Matriz:')
            print(matriz)
            tn, fp, fn, tp = matriz.ravel()

            if fp <= 280 and f1_validacao > melhor_f1:
                melhor_f1 = f1_validacao
                melhor_threshold = threshold

        print('=' * 30)
        print(f'Melhor Threshold: {melhor_threshold}')
        print(f'Melhor F1: {melhor_f1}')

        probabilidades_teste = melhor_modelo.predict_proba(x_test)[:,1]

        previsoes_teste = (probabilidades_teste >= melhor_threshold).astype(int)

        accurracy = accuracy_score(y_test, previsoes_teste)

        precision = precision_score(y_test, previsoes_teste)

        recall = recall_score(y_test, previsoes_teste)

        f1 = f1_score(y_test, previsoes_teste)

        roc = roc_auc_score(y_test, probabilidades_teste)

        print('\n==============================')
        print('AVALIAÇÃO FINAL')
        print('==============================')

        print(f'Accuracy: {accurracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1: {f1:.4f}')
        print(f'ROC-AUC: {roc:.4f}')

        print('\nMatriz de confusão:')
        print(confusion_matrix(y_test, previsoes_teste))

        print('\nClassification Report:')
        print(classification_report(y_test, previsoes_teste))

        resultado_teste = x_test.copy()
        resultado_teste['real'] = y_test
        resultado_teste['probabilidade_reprovado'] = probabilidades_teste
        resultado_teste['previsto'] = previsoes_teste

        falsos_positivos = resultado_teste[
            (resultado_teste['real'] == 0) &  (resultado_teste['previsto'] == 1)]

        falsos_negativos = resultado_teste[
            (resultado_teste['real'] == 1) & (resultado_teste['previsto'] == 0)]
        print(f'FP: {len(falsos_positivos)}')
        print(f'FN: {len(falsos_negativos)}')

        resultado_teste['tipo'] = np.select(
            [
                (resultado_teste['real'] == 0) & (resultado_teste['previsto'] == 0),
                (resultado_teste['real'] == 0) & (resultado_teste['previsto'] == 1),
                (resultado_teste['real'] == 1) & (resultado_teste['previsto'] == 0),
                (resultado_teste['real'] == 1) & (resultado_teste['previsto'] == 1)
            ],
            [
                'Verdadeiro Negativo',
                'Falso Positivo',
                'Falso Negativo',
                'Verdadeiro Positivo'
            ],
            default = 'Outro'
        )

        print(resultado_teste['tipo'].value_counts())

        print(
            resultado_teste.groupby('tipo')[
                [
                    'person_age',
                    'person_income',
                    'loan_amnt',
                    'loan_percent_income'
                ]
            ].agg(['mean', 'median']))

        print(
            resultado_teste.groupby('tipo')['probabilidade_reprovado'].agg(
                ['mean', 'median', 'min', 'max']
            )
        )

        pd.set_option('display.max_columns', None)

        print(
            resultado_teste[
                resultado_teste['tipo'].isin([
                    'Falso Positivo',
                    'Verdadeiro Positivo'
                ])
            ].groupby('tipo')[
                [
                    'person_age',
                    'person_income',
                    'loan_amnt',
                    'loan_percent_income',
                    'cb_person_default_on_file'
                ]
            ].agg(['mean', 'median'])
        )

        print(
            resultado_teste[
                resultado_teste['tipo'].isin([
                    'Falso Negativo',
                    'Verdadeiro Negativo'
                ])
            ].groupby('tipo')[
                [
                    'person_age',
                    'person_income',
                    'loan_amnt',
                    'loan_percent_income',
                    'cb_person_default_on_file'
                ]
            ].agg(['mean', 'median'])
        )

        print('=' * 30)
        grupos = [
            'Verdadeiro Negativo',
            'Falso Positivo',
            'Falso Negativo',
            'Verdadeiro Positivo'
        ]

        for grupo in grupos:
            dados = resultado_teste[
                resultado_teste['tipo'] == grupo
                ]['loan_percent_income']

            plt.hist(
                dados,
                bins=30,
                alpha=0.5,
                density=True,
                label=grupo
            )

        plt.xlabel('Loan Percent Income')
        plt.ylabel('Quantidade de clientes')
        plt.title('Distribuição de Loan Percent Income por tipo de previsão')
        plt.legend()
        plt.show()

        grupos = [
            'Verdadeiro Negativo',
            'Falso Positivo',
            'Falso Negativo',
            'Verdadeiro Positivo'
        ]

        for grupo in grupos:
            dados = resultado_teste[
                resultado_teste['tipo'] == grupo
                ]['person_income']

            plt.hist(
                dados,
                bins=30,
                alpha=0.5,
                density=True,
                label=grupo
            )

        plt.xlabel('Person Income')
        plt.ylabel('Densidade')
        plt.title('Distribuição de Renda por tipo de previsão')
        plt.legend()
        plt.show()

        print(
            resultado_teste.groupby('tipo')['person_income'].agg(
                ['mean', 'median', 'min', 'max']
            )
        )

        print(
            resultado_teste.groupby('tipo')['loan_amnt'].agg(
                ['mean', 'median', 'min', 'max']
            )
        )

        print(
            pd.crosstab(
                resultado_teste['tipo'],
                resultado_teste['cb_person_default_on_file'],
                normalize='index'
            )
        )

        resultado_teste['faixa_loan_percent_income'] = pd.qcut(
            resultado_teste['loan_percent_income'],
            q=5,
            duplicates='drop'
        )

        print(
            pd.crosstab(
                resultado_teste['faixa_loan_percent_income'],
                resultado_teste['tipo']
            ).reindex(
                columns=[
                    'Verdadeiro Negativo',
                    'Falso Positivo',
                    'Falso Negativo',
                    'Verdadeiro Positivo'
                ],
                fill_value=0
            ))

        print('=' * 50)

    def __atualizar_treinamento_da_ia(self):
        self.__treinar_ia()
