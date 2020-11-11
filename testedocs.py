from sklearn.neighbors import NearestCentroid
from preparacaoDados import tratamentoDados
from scipy.spatial import distance
import pandas as pd
import random

data, label = tratamentoDados("OHE")
dados_tce = dados_completos
del dados_completos, dados_outliers, data_analisar, outliers, rows,docs
random_state = 0
nums_aleatorios = random.sample(range(0, 324705), 15000)
dados_aleatorios = data[data.columns].iloc[nums_aleatorios]
del nums_aleatorios

clf = clf = NearestCentroid()
clf.fit(data, label[0])
centroids = pd.DataFrame(clf.centroids_)

media_dados = [0]* dados_tce.shape[0]
# percorre todos os documentos calculando a distancia dele com toda a base
for i in range(dados_tce.shape[0]):
    # percorre todos os centroids calculando a distancia com o doc i
    for j in range(centroids.shape[0]):
        media_dados[i] += distance.cosine(dados_tce.iloc[i],centroids.iloc[j])
    # divide as distancias do doc i para toda a base j pela quantidade de elementos da base j
    media_dados[i] =  media_dados[i] / centroids.shape[0]

media_dados_val = sum(media_dados )/ len(media_dados)

media_dados_test = [0]* dados_aleatorios.shape[0]
# percorre todos os documentos calculando a distancia dele com toda a base
for i in range(dados_aleatorios.shape[0]):
    # percorre todos os docs da base calculando a distancia com o doc i
    for j in range(centroids.shape[0]):
        media_dados_test[i] += distance.cosine(dados_aleatorios.iloc[i],centroids.iloc[j])
    # divide as distancias do doc i para toda a base j pela quantidade de elementos da base j
    media_dados_test[i] =  media_dados_test[i] / centroids.shape[0]

media_dados_test_val = sum(media_dados_test) / len(media_dados_test)