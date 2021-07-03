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

dados_juliete = pd.read_excel("dados_analisados.xlsx")

#dados_juliete['Natureza Despesa (Cod)(EOF)'].value_counts() # 392
corretos = dados_juliete["Natureza Despesa (Cod)(EOF)"][dados_juliete["ANÁLISE"] == "OK"].value_counts().to_dict()
incorretos = dados_juliete["Natureza Despesa (Cod)(EOF)"][dados_juliete["ANÁLISE"] == "INCORRETO"].value_counts().to_dict()
inconclusivos = dados_juliete["Natureza Despesa (Cod)(EOF)"][dados_juliete["ANÁLISE"] == "INCONCLUSIVO"].value_counts().to_dict()
quantidade_total = dados_juliete["Natureza Despesa (Cod)(EOF)"].value_counts().to_dict()


labels = pd.DataFrame(dados_juliete["Natureza Despesa (Cod)(EOF)"].value_counts().index,columns =["natureza"])
labels["porcentagem_correto"] = 0
labels["porcentagem_incorreto"] = 0
labels["porcentagem_inconclusivo"] = 0
labels["total"] = [quantidade_total[labels["natureza"].iloc[i]] for i in range(labels.shape[0])]
for i in range(labels.shape[0]):
    try:
        labels["porcentagem_correto"].iloc[i] = corretos[labels["natureza"].iloc[i]] / quantidade_total[labels["natureza"].iloc[i]]
    except:
        pass
    try:
        labels["porcentagem_incorreto"].iloc[i] = incorretos[labels["natureza"].iloc[i]] / quantidade_total[labels["natureza"].iloc[i]]
    except:
        pass
    try:
        labels["porcentagem_inconclusivo"].iloc[i] = inconclusivos[labels["natureza"].iloc[i]] / quantidade_total[labels["natureza"].iloc[i]]
    except:
        pass
labels.to_excel("validacao_dados.xlsx")
