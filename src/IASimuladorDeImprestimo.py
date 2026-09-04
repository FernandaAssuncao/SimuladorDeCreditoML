import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, precision_recall_curve
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
            #print(f'Accuracy: {accuracy_score(y_test, previsoes):.4f}%')
            #print(f'Matrix de confusão: {confusion_matrix(y_test, previsoes)}')
            #print(f'Precision score: {precision_score(y_test, previsoes)}')
            #print(f'Recall score: {recall_score(y_test, previsoes)}')
            #print(f'F1 score: {f1_score(y_test, previsoes)}')
            #print(f'Classification report: {classification_report(y_test, previsoes)}')
            print('=' * 30)

    def __atualizar_treinamento_da_ia(self):
        self.__treinar_ia()
