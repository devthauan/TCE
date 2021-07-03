# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from tratamentos import pickles
from scipy.sparse import csr_matrix
from tratarDados import tratarDados
from sklearn.metrics import f1_score
from preparacaoDados import tratamentoDados
from sklearn.ensemble import RandomForestClassifier


dados_tce = pickles.carregarPickle("dados_tce_limpos")[:1000] # 265246
dados_juliete = pd.read_excel("dados_analisados.xlsx")
dados_juliete.insert (25, "Empenho (Sequencial Empenho)(EOF).1", dados_juliete["Empenho (Sequencial Empenho)(EOF)"])

#deixando apenas os dados corretos e incorretos
dados_juliete = dados_juliete[dados_juliete["ANÁLISE"] != "INCONCLUSIVO"]
dados_juliete.reset_index(drop = True, inplace = True)
#dados_juliete['Natureza Despesa (Cod)(EOF)'].value_counts() #379 naturezas
tratamentoDados(dados_tce.copy(),"tfidf")
tratamentoDados(dados_tce.copy(),"OHE")
dados_tce = pickles.carregarPickle("data")
label = pickles.carregarPickle("label")
tfidf = pickles.carregarPickle("tfidf")
dados_tce = sparse.hstack((csr_matrix(dados_tce),csr_matrix(tfidf) ))
del tfidf
print(dados_tce.shape)
# retirando as classes que nao estao no treino
naturezas_juliete = pd.DataFrame(dados_juliete['Natureza Despesa (Cod)(EOF)'])
natureza_tce = list(label['natureza_despesa_cod'].value_counts().index)
fora_do_modelo = []
# remove naturezas que nao estao presentes nos dados de treino
for i in range(len(naturezas_juliete)):
    if(naturezas_juliete.iloc[i][0] not in natureza_tce):
        fora_do_modelo.append(i)
dados_juliete.drop(fora_do_modelo,inplace = True)
dados_juliete.reset_index(inplace = True,drop=True)
del naturezas_juliete,fora_do_modelo, natureza_tce
# Trata os dados
analise = dados_juliete["ANÁLISE"]
dados_juliete.drop("ANÁLISE",axis = 1, inplace = True)
dados_teste, label_teste = tratarDados(dados_juliete)
del dados_juliete
# =============================================================================
# RF
# =============================================================================
modelo = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state = 10)
modelo.fit(dados_tce, label.values.ravel())
y_predito = pd.DataFrame(modelo.predict(dados_teste))
y_predito.columns = label_teste.columns
label_string = [0]*label_teste.shape[0]
for i in range(label_teste.shape[0]):
    if(label_teste['natureza_despesa_cod'].iloc[i] == y_predito['natureza_despesa_cod'].iloc[i]):
        label_string[i] = "OK"
    else:
        label_string[i] = "INCORRETO"
    
#
micro = f1_score(analise,label_string,average='micro')
macro = f1_score(analise,label_string,average='macro')
print("O f1Score micro do RandomForest com ",200,"arvores é: ",micro)
print("O f1Score macro do RandomForest com ",200,"arvores é: ",macro)
# =============================================================================
# SVC
# =============================================================================
modelo = SVC(kernel="linear",C= 10,random_state=0)
modelo.fit(csr_matrix(dados_tce), label.values.ravel())
y_predito = pd.DataFrame(modelo.predict(dados_teste))
y_predito.columns = label_teste.columns
label_string = [0]*label_teste.shape[0]
for i in range(label_teste.shape[0]):
    if(label_teste['natureza_despesa_cod'].iloc[i] == y_predito['natureza_despesa_cod'].iloc[i]):
        label_string[i] = "OK"
    else:
        label_string[i] = "INCORRETO"
    
#
micro = f1_score(analise,label_string,average='micro')
macro = f1_score(analise,label_string,average='macro')
string = ""
valor_c = 10
print("O f1Score micro do SVC ",string," com parametro C = ",valor_c,"é: ",micro)
print("O f1Score macro do SVC ",string," com parametro C = ",valor_c,"é: ",macro)
