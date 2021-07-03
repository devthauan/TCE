# -*- coding: utf-8 -*-
import sys
import pickle
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from tratamentos import pickles
from tratarDados import tratarDados
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from conexaoDados import range_dados
from datetime import date, timedelta
from conexaoDados import todos_dados
from preparacaoDados import tratamentoDados
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tratarDados import refinamento_hiperparametros

data = pd.read_excel("dados_analisados.xlsx")
print(data.shape)
label = data["ANÁLISE"]
data.drop("ANÁLISE", axis = 1, inplace = True)
data_treino, data_teste, label_treino, label_teste = train_test_split(data, label, test_size = 0.1,stratify = label, random_state = 10)
data_treino.reset_index(drop = True, inplace = True)
data_teste.reset_index(drop = True, inplace = True)
label_treino.reset_index(drop = True, inplace = True)
label_teste.reset_index(drop = True, inplace = True)
del data, label
# Tratando os dados de treino
tratamentoDados(data_treino.copy(),'tfidf')
tratamentoDados(data_treino.copy(),'Modelo 2')
data_treino = pickles.carregarPickle("data")
label_treino = pickles.carregarPickle("label")
tfidf_treino = pickles.carregarPickle("tfidf")
data_treino = sparse.hstack((csr_matrix(data_treino),csr_matrix(tfidf_treino) ))
del tfidf_treino
print("Treino: ",data_treino.shape)
# Aplicando o tratamento nos dados de teste
data_teste, label_teste = tratarDados(data_teste, "Modelo 2")
print("Teste: ",data_teste.shape)
# Modelo
valor_c = 1.3
modelo = SVC(kernel="linear", random_state=0, C = valor_c)
modelo.fit(data_treino, label_treino.values.ravel())
y_predito = modelo.predict(data_teste)
micro = f1_score(label_teste,y_predito,average='micro')
macro = f1_score(label_teste,y_predito,average='macro')
print("O f1Score micro do SVC com parametro C = ",valor_c,"é: ",micro)
print("O f1Score macro do SVC com parametro C = ",valor_c,"é: ",macro)

modelo = RandomForestClassifier(n_jobs = -1, random_state = 10, n_estimators = 400 )
modelo.fit(data_treino, label_treino.values.ravel())
y_predito = modelo.predict(data_teste)
micro = f1_score(label_teste,y_predito,average='micro')
macro = f1_score(label_teste,y_predito,average='macro')
print("O f1Score micro do RF é: ",micro)
print("O f1Score micro do RF é: ",macro)

