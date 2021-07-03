# -*- coding: utf-8 -*-
import xlrd
import numpy as np
import pandas as pd
from scipy import sparse
from tratamentos import pickles
from tratarDados import tratarDados
from scipy.sparse import csr_matrix
from tratamentos import tratar_label
from preparacaoDados import tratamentoDados
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("arquivos/dadosTCE.csv", low_memory = False)
label = data["Natureza Despesa (Cod)(EOF)"]
data['Período (Dia/Mes/Ano)(EOF)'] = [xlrd.xldate_as_datetime(data['Período (Dia/Mes/Ano)(EOF)'].iloc[i],0).date().isoformat() for i in range(data.shape[0])]
# Retirando naturezas com numero de empenhos menor ou igual a x
label, index_label_x_empenhos = tratar_label.label_elemento(label, 9)
data.drop(index_label_x_empenhos,inplace = True, axis = 0)
data.reset_index(drop = True, inplace = True)
# Separando treino e teste
data, data_teste, label, label_teste = train_test_split(data, label, test_size = 0.3,stratify = label, random_state = 10)
print("Tamanho dos dados de treino", data.shape)
print("Tamanho dos dados de teste", data_teste.shape)
# Tratando os documentos de treino e salvando os modelos para serem aplicados no teste
data.reset_index(drop = True, inplace = True)
data_teste.reset_index(drop = True, inplace = True)
label.reset_index(drop = True, inplace = True)
label_teste.reset_index(drop = True, inplace = True)
tratamentoDados(data.copy(), "tfidf")
tratamentoDados(data.copy(), "OHE")
# Carregar os dados tratados
data = pickles.carregarPickle("data")
label = pickles.carregarPickle("label")
label.reset_index(drop = True, inplace = True)
tfidf = pickles.carregarPickle("tfidf")
data = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
#data =  pd.DataFrame.sparse.from_spmatrix(data)
# Tratando os dados de teste
data_teste , label_teste = tratarDados(data_teste, "visao dupla")
# =============================================================================
# RF
# =============================================================================
modelo = RandomForestClassifier(n_jobs=-1, random_state= 10,max_samples=int(data.shape[0]*0.3))
scores = cross_validate(modelo, data, label.values.ravel(), cv=3,scoring=['f1_micro','f1_macro'])
print(scores)
print("Media dos micros", np.mean(list(scores["test_f1_micro"])))
print("Media dos macros", np.mean(list(scores["test_f1_macro"])))
