# -*- coding: utf-8 -*-
import time
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from tratamentos import pickles
from tratarDados import tratarDados
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from tratamentos import tratar_label
from preparacaoDados import tratamentoDados
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tratarDados import refinamento_hiperparametros
from sklearn.model_selection import train_test_split
from tratarDados import refinamento_hiperparametros

def load_dados(pouca_natureza, porcentagem_split):
    # Importando os dados do tribunal
    data = pd.read_csv("dadosTCE.csv", low_memory = False)[:500]
    data.drop("Empenho (Sequencial Empenho)(EOF).1", axis = 1, inplace = True)
    label = data["Natureza Despesa (Cod)(EOF)"]
    # Retirando naturezas com numero de empenhos menor ou igual a x
    label, index_label_x_empenhos = tratar_label.label_elemento(label, pouca_natureza)
    data.drop(index_label_x_empenhos,inplace = True, axis = 0)
    data.reset_index(drop = True, inplace = True)
    # Separando X% dos dados para selecao de hiperparametros
    data_treino, data_teste, label_treino, label_teste = train_test_split(data, label, test_size = porcentagem_split,stratify = label, random_state = 10)
    del data, label, label_teste, label_treino
    # Resetando os indexes dos dados
    data_treino.reset_index(drop = True, inplace = True)
    data_teste.reset_index(drop = True, inplace = True)
    return data_treino, data_teste

def tratar_dados(data_treino, data_teste, teste):
    tratamentoDados(data_treino.copy(), "OHE")
    tratamentoDados(data_treino.copy(), "tfidf")
    # Carregar os dados tratados
    data_treino = pickles.carregarPickle("data")
    label_treino = pickles.carregarPickle("label")
    tfidf_treino = pickles.carregarPickle("tfidf")
    # Retirando naturezas com numero de empenhos menor que X depois da limpesa
    label_treino, index_label_x_empenhos = tratar_label.label_elemento(label_treino["natureza_despesa_cod"], 2)
    label_treino = pd.DataFrame(label_treino)["natureza_despesa_cod"]
    data_treino.drop(index_label_x_empenhos,inplace = True, axis = 0)
    data_treino.reset_index(drop = True, inplace = True)
    tfidf_treino.drop(index_label_x_empenhos,inplace = True, axis = 0)
    tfidf_treino.reset_index(drop = True, inplace = True)
    del index_label_x_empenhos
    # Tamanhos dos dados de treino tratados
    print("OHE_treino",data_treino.shape)
    print("TF-IDF_treino",tfidf_treino.shape)
    visao_dupla_treino = csr_matrix( sparse.hstack((csr_matrix(data_treino),csr_matrix(tfidf_treino) )) )
    print("Visao dupla, dados combinados do treino",visao_dupla_treino.shape)
    if(teste):
        # Aplicar o tratamento no teste
        tfidf_teste, label_teste = tratarDados(data_teste.copy(),'tfidf')
        data_teste, label_teste = tratarDados(data_teste.copy(),'OHE')
        # Retirando naturezas com numero de empenhos menor que X depois da limpesa
        label_teste, index_label_x_empenhos = tratar_label.label_elemento(label_teste["natureza_despesa_cod"], 2)
        label_teste = pd.DataFrame(label_teste)["natureza_despesa_cod"]
        data_teste.drop(index_label_x_empenhos,inplace = True, axis = 0)
        data_teste.reset_index(drop = True, inplace = True)
        tfidf_teste.drop(index_label_x_empenhos,inplace = True, axis = 0)
        tfidf_teste.reset_index(drop = True, inplace = True)
        # Tamanhos dos dados de treino tratados
        print("OHE_teste",data_teste.shape)
        print("TF-IDF_teste",tfidf_teste.shape)
        visao_dupla_teste = csr_matrix( sparse.hstack((csr_matrix(data_teste),csr_matrix(tfidf_teste) )) )
        print("Visao dupla, dados combinados do testes",visao_dupla_teste.shape)
        return data_treino, label_treino, data_teste, label_teste, tfidf_treino, tfidf_teste, visao_dupla_treino, visao_dupla_teste
    else:
        return data_treino, label_treino, tfidf_treino, visao_dupla_treino
    
data_treino, data_teste = load_dados(6, 0.6)
teste = 1
data_treino, label_treino, data_teste, label_teste, tfidf_treino, tfidf_teste, visao_dupla_treino, visao_dupla_teste = tratar_dados(data_treino, data_teste, teste)
del tfidf_teste, tfidf_treino
data_treino = visao_dupla_treino
data_teste = visao_dupla_teste

hiperparametros = {'n_estimators':[100,300,500,600,700,800,980] }
for i in range(len(hiperparametros["n_estimators"])):
    modelo = RandomForestClassifier(n_estimators = hiperparametros["n_estimators"][i], n_jobs = -1, random_state = 10, max_samples = int(label_treino.shape[0]*0.3) )
    modelo.fit(data_treino, label_treino.values.ravel())
    y_predito = modelo.predict(data_teste)
    micro = f1_score(label_teste,y_predito,average='micro')
    macro = f1_score(label_teste,y_predito,average='macro')
    print(hiperparametros["n_estimators"][i])
    print("micro: ", micro)
    print("macro: ", macro)
    print(modelo.get_params)
