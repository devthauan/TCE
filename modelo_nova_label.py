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

#def modelo_validacao(dados_tce):
# Carregando os dados validados
dados_tce = pickles.carregarPickle("dados_tce_limpos")[:1000] # 265246
dados_juliete = pd.read_excel("dados_analisados.xlsx")
dados_juliete.insert (25, "Empenho (Sequencial Empenho)(EOF).1", dados_juliete["Empenho (Sequencial Empenho)(EOF)"])

# Definindo a label como a avaliacao da validacao ( correto, incorreto, inconclusivo )
label_juliete = pd.DataFrame(tratamentoDados(dados_juliete.copy(),"dropar"))
dados_juliete.drop("ANÁLISE",axis = 1, inplace = True)
#
tratamentoDados(dados_juliete.copy(),"tfidf")
tratamentoDados(dados_juliete.copy(),"OHE")
dados_juliete = pickles.carregarPickle("data")
tfidf_juliete = pickles.carregarPickle("tfidf")
dados_juliete = sparse.hstack((csr_matrix(dados_juliete),csr_matrix(tfidf_juliete) ))
del tfidf_juliete
print(dados_juliete.shape)
# tratando os dados de teste
dados_teste, label_teste = tratarDados(dados_tce)
# =============================================================================
# MODELOS
# =============================================================================
identificador_empenho = pickles.carregarPickle("modelos_tratamentos/identificador_empenho")
modelo = SVC(kernel="linear",C= 10,random_state=0)
modelo.fit(dados_juliete, label_juliete.values.ravel())
y_predito = pd.DataFrame(modelo.predict(dados_teste))
# =============================================================================
# PORCENTAGENS DE CORRETUDE
# =============================================================================
corretude  = pd.concat([label_teste,y_predito], axis =1)
#
corretos = pd.DataFrame(corretude["natureza_despesa_cod"][corretude[0] == "OK"])
corretos = corretos["natureza_despesa_cod"].value_counts().to_dict()
incorretos = pd.DataFrame(corretude["natureza_despesa_cod"][corretude[0] == "INCORRETO"])
incorretos = incorretos["natureza_despesa_cod"].value_counts().to_dict()
inconclusivos = pd.DataFrame(corretude["natureza_despesa_cod"][corretude[0] == "INCONCLUSIVO"])
inconclusivos = inconclusivos["natureza_despesa_cod"].value_counts().to_dict()
#
resultado = corretude["natureza_despesa_cod"].value_counts().to_dict()
for i in range(len(resultado.keys())):
    resultado[list(resultado.keys())[i]] = [ resultado[list(resultado.keys())[i]] ]
    try:
        resultado[list(resultado.keys())[i]].append(corretos[list(resultado.keys())[i]])
    except:
        resultado[list(resultado.keys())[i]].append(0)
    try:
        resultado[list(resultado.keys())[i]].append(incorretos[list(resultado.keys())[i]])
    except:
        resultado[list(resultado.keys())[i]].append(0)
    try:
        resultado[list(resultado.keys())[i]].append(inconclusivos[list(resultado.keys())[i]])
    except:
        resultado[list(resultado.keys())[i]].append(0)
resultado = pd.DataFrame(data = pd.concat([pd.DataFrame(resultado.keys()), pd.DataFrame(resultado.values())],axis =1))
resultado.columns = ["Label","Quantidade total","Quantidade corretos", "Quantidade incorretos", "Quantidade inconclusivos"]
resultado["Porcentagem correto"] =  [resultado["Quantidade corretos"].iloc[i]/resultado["Quantidade total"].iloc[i] for i in range(len(resultado))]
resultado["Porcentagem incorreto"] =  [resultado["Quantidade incorretos"].iloc[i]/resultado["Quantidade total"].iloc[i] for i in range(len(resultado))]
resultado["Porcentagem inconclusivo"] =  [resultado["Quantidade inconclusivos"].iloc[i]/resultado["Quantidade total"].iloc[i] for i in range(len(resultado))]
resultado.to_excel("resultado.xlsx",index = False)
## =============================================================================
## corretos
## =============================================================================   
#incorretos.value_counts()
#corretos = identificador_empenho.iloc[list(corretos.index)]
#corretos.reset_index(drop = True, inplace = True)
#dados_corretos = [0]*len(corretos)
#for i in range(len(corretos)):
#    dados_corretos[i] = dados_tce[dados_tce["empenho_sequencial_empenho"] == corretos["empenho_sequencial_empenho"].iloc[i]].index[0]
#dados_corretos = dados_tce.iloc[dados_corretos]
#dados_corretos.to_excel("corretos.xlsx", index = False)
#del dados_corretos
## =============================================================================
## incorretos
## =============================================================================
#incorretos = y_predito[y_predito[0] == "INCORRETO"]
#incorretos = identificador_empenho.iloc[list(incorretos.index)]
#incorretos.reset_index(drop = True, inplace = True)
#dados_incorretos = [0]*len(incorretos)
#for i in range(len(incorretos)):
#    dados_incorretos[i] = dados_tce[dados_tce["empenho_sequencial_empenho"] == incorretos["empenho_sequencial_empenho"].iloc[i]].index[0]
#dados_incorretos = dados_tce.iloc[dados_incorretos]
#dados_incorretos.to_excel("incorretos.xlsx", index = False)
#del dados_incorretos
## =============================================================================
## inconclusivos
## =============================================================================
#inconclusivos = y_predito[y_predito[0] == "INCONCLUSIVO"]
#inconclusivos = identificador_empenho.iloc[list(inconclusivos.index)]
#inconclusivos.reset_index(drop = True, inplace = True)
#dados_inconclusivos = [0]*len(inconclusivos)
#for i in range(len(inconclusivos)):
#    dados_inconclusivos[i] = dados_tce[dados_tce["empenho_sequencial_empenho"] == inconclusivos["empenho_sequencial_empenho"].iloc[i]].index[0]
#dados_inconclusivos = dados_tce.iloc[dados_inconclusivos]
#dados_inconclusivos.to_excel("inconclusivos.xlsx", index = False)
## =============================================================================