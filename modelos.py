# Packages python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split

# Meus packages
from modelos import single_value_decomposition
from preparacaoDados import tratamentoDados
from modelos import feature_importance
from modelos import gaussian_mixtures
from modelos import analise_cluster
from modelos import rocchio
from modelos import kmeans
from modelos import dbscan


# Retorna os dados tratados
data, label = tratamentoDados("sem OHE")

# Seleção de atributos usando Feature Importance
data_feature_importance, colunasMaisImportantes = feature_importance.featureImportance(data,label,30)
# Acha o melhor valor de K
#best_k = kmeans.bestK(data_feature_importance)
# Rótulos atribuidos pelo algoritmo kmeans
labels_kmeans = pd.DataFrame((kmeans.kmeans(250,data_feature_importance)).ravel())
# Metrica para medir a qualidade do cluster
shilhouette_result = silhouette_score(data_feature_importance, labels_kmeans)
print("Silhouette score do Kmeans com Feature Importance é: ",shilhouette_result," e foi dividido em ",pd.DataFrame(labels_kmeans)[0].value_counts().count()," clusters")
porcentagem_labels = analise_cluster.porcentagem_correta_cluster(pd.DataFrame(label),pd.DataFrame(labels_kmeans))
print("Esse método gerou ",porcentagem_labels," de ",pd.DataFrame(label)[0].value_counts().count()," labels acima de 80% " ,"resultando em ",'%.2f' %((porcentagem_labels/pd.DataFrame(label)[0].value_counts().count())*100),"% das naturezas bem classificadas" )

## Rótulos atribuidos pelo algoritmo dbscan
#labels_dbscan = dbscan.dbscan(data_feature_importance, 0.1,10)
#labels_dbscan[0]=1
#pd.DataFrame(labels_dbscan)[0].value_counts()
## Metrica para medir a qualidade do cluster
#shilhouette_result = silhouette_score(data_feature_importance, labels_dbscan)
#print("Silhouette score do DBscan com Feature Importance é: ",shilhouette_result)
#print("Silhouette score do DBscan com Feature Importance é: ",shilhouette_result," e foi dividido em ",pd.DataFrame(labels_dbscan)[0].value_counts().count()," clusters")

###############################################################################
del data_feature_importance

# Seleção de atributos usando Principal Component Analysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA().fit(data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid()
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

numero_componentes = 40
n_batches = 100
inc_pca = IncrementalPCA(n_components=numero_componentes)
for X_batch in np.array_split(data, n_batches):
    inc_pca.partial_fit(X_batch)
data_PCA = inc_pca.transform(data)


#best_k = kmeans.bestK(data_PCA)
labels_kmeans = kmeans.kmeans(250,data_PCA)
shilhouette_result = silhouette_score(data_PCA, labels_kmeans)
print("Silhouette score do Kmeans com Principal Component Analysis é: ",shilhouette_result," e foi dividido em ",pd.DataFrame(labels_kmeans)[0].value_counts().count()," clusters")
porcentagem_labels = analise_cluster.porcentagem_correta_cluster(pd.DataFrame(label),pd.DataFrame(labels_kmeans))
print("Esse método gerou ",porcentagem_labels," de ",pd.DataFrame(label)[0].value_counts().count()," labels acima de 80% " ,"resultando em ",'%.2f' %((porcentagem_labels/pd.DataFrame(label)[0].value_counts().count())*100),"% das naturezas bem classificadas" )

#labels_dbscan = dbscan.dbscan(data_SVD, 0.1,10)
#labels_dbscan[0]=1
#shilhouette_result = silhouette_score(data_SVD, labels_dbscan)
#pd.DataFrame(labels_dbscan)[0].value_counts()
#print("Silhouette score do DBscan com Single Value Decomposition é: ",shilhouette_result," e foi dividido em ",pd.DataFrame(labels_dbscan)[0].value_counts().count()," clusters")

###############################################################################
del data_PCA
#best_k = kmeans.bestK(data)
labels_kmeans = kmeans.kmeans(250,data)
shilhouette_result = silhouette_score(data, labels_kmeans)
print("Silhouette score do Kmeans sem seleção de atributos é: ",shilhouette_result," e foi dividido em ",pd.DataFrame(labels_kmeans)[0].value_counts().count()," clusters")
porcentagem_labels = analise_cluster.porcentagem_correta_cluster(pd.DataFrame(label),pd.DataFrame(labels_kmeans))
print("Esse método gerou ",porcentagem_labels," de ",pd.DataFrame(label)[0].value_counts().count()," labels acima de 80% " ,"resultando em ",'%.2f' %((porcentagem_labels/pd.DataFrame(label)[0].value_counts().count())*100),"% das naturezas bem classificadas" )


############################ MODELOS ##########################################
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.2,stratify = label)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
print("O score da Regressão Linear sem Feature selection é: ",reg.score(X_test,y_test.values.ravel()))


from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=1)
rnd_clf.fit(X_train, y_train.values.ravel())
print("O score do random Forest sem Feature selection é: ",rnd_clf.score(X_test,y_test.values.ravel()))









######################Quantidade de subelementos por elemento##################

label_elemento = label.copy()
label_subelemento = label.copy()
# Separando os codigos do elemento e subelemento
for i in range(label.shape[0]):    
    if(len(str(label_elemento[0].iloc[i]))==3):
        label_elemento[0].iloc[i] = str(label[0].iloc[i])[:1]
        label_subelemento[0].iloc[i] = str(label[0].iloc[i])[1:]
    else:
        label_elemento[0].iloc[i] = str(label[0].iloc[i])[:2]
        label_subelemento[0].iloc[i] = str(label[0].iloc[i])[2:]

f = open('subelemento_elemento_60.xlsx','a+')
for i in range(label_elemento[0].value_counts().count()):
    posicao = label_elemento.where(label_elemento == label_elemento[0].value_counts().index[i]).dropna().index
    subelemento = label_subelemento.iloc[posicao][0].value_counts()
    print("elemento: ",label_elemento[0].value_counts().index[i])
    f.write("elemento: "+str(label_elemento[0].value_counts().index[i])+'\n')
#    print("subelemento quantidade\n")
    f.write("subelemento ; quantidade ; total:"+str(len(subelemento))+"\n")
    for i in range(len(subelemento)):
       f.write(str(subelemento.index[i])+";")
       f.write(str(subelemento.values[i])+"\n")
    f.write("\n")
f.flush()
f.close()

f = open('subelemento_elemento2_60.xlsx','a+')
for i in range(label_elemento[0].value_counts().count()):
    posicao = label_elemento.where(label_elemento == label_elemento[0].value_counts().index[i]).dropna().index
    subelemento = label_subelemento.iloc[posicao][0].value_counts()
    f.write("elemento:"+str(label_elemento[0].value_counts().index[i])+" ; quantidade diferentes:"+str(len(subelemento))+" ; total:"+str(sum(subelemento.values))+"\n")
f.flush()
f.close()
label_elemento[0].value_counts().count()

# =============================================================================
# ANALISE DE ERROS
# =============================================================================
# Abaixo de 50%
classes = [119,305,314,323,411,807,808,812,1206,1706,3012,3020,3030,3031,3036,3039,
3045,3048,3049,3054,3055,3056,3058,3061,3209,3210,3212,3301,3505,3507,3508,
3514,3601,3603,3606,3607,3608,3612,3615,3620,3630,3634,3637,3646,3647,3650,
3706,3710,3907,3952,3954,3958,3961,3967,3970,3976,3979,4016,4115,4125,4305,
4515,4701,5115,5201,5204,5206,5208,5217,5218,5219,5220,5223,5241,5242,9203,
9206,9208,9213,9222,9225,9227,9232,9233,9234,9239,9240,9241,9242,9245,9246,
9247,9248,9251,9255,9256,9257,9259,9262,9263,9264,9266,9267,9270,9271,9272,
9275,9277,9278,9279,9280,9281,9283,9285]

# Abaixo de 60%
classes = [119, 305, 314, 322, 323, 402, 405, 409, 411, 807, 808, 812, 1206, 1325, 1706, 1803,
 2202, 3012, 3015, 3016, 3020, 3021, 3030, 3031, 3033, 3036, 3039, 3041, 3045, 3046,
 3048, 3049, 3052, 3054, 3055, 3056, 3058, 3061, 3203, 3209, 3210, 3212, 3301, 3505,
 3506, 3507, 3508, 3509, 3514, 3601, 3603, 3606, 3607, 3608, 3612, 3613, 3615, 3620,
 3629, 3630, 3634, 3637, 3639, 3641, 3645, 3646, 3647, 3650, 3706, 3710, 3907, 3916,
 3952, 3954, 3958, 3961, 3967, 3970, 3976, 3979, 4013, 4016, 4028, 4082, 4115, 4123,
 4125, 4305, 4515, 4701, 5111, 5115, 5117, 5201, 5204, 5205, 5206, 5208, 5212, 5213,
 5214, 5217, 5218, 5219, 5220, 5221, 5223, 5225, 5233, 5234, 5235, 5241, 5242, 9203,
 9205, 9206, 9208, 9213, 9222, 9225, 9227, 9232, 9233, 9234, 9239, 9240, 9241, 9242,
 9245, 9246, 9247, 9248, 9249, 9250, 9251, 9252, 9254, 9255, 9256, 9257, 9259, 9262,
 9263, 9264, 9266, 9267, 9269, 9270, 9271, 9272, 9274, 9275, 9277, 9278, 9279, 9280,
 9281, 9283, 9285, 9290, 9312]



result = [0]*len(classes)
for i in range(len(classes)):
    if(len(str(classes[i]))==3):
        elemento = label_elemento[0].where(label_elemento[0]==str(classes[i])[:1]).dropna().value_counts().values[0]
        sub = label[0].where(label[0]==classes[i]).dropna().value_counts().values[0]
        result[i]= [[classes[i]],[sub/elemento]]
    else:
        elemento = label_elemento[0].where(label_elemento[0]==str(classes[i])[:2]).dropna().value_counts().values[0]
        sub = label[0].where(label[0]==classes[i]).dropna().value_counts().values[0]
        result[i]= [[classes[i]],[sub/elemento]]
    
result = pd.DataFrame(result)

# =============================================================================
# DISTANCIA PARA O CENTROID
# =============================================================================
from preparacaoDados import tratamentoDados
from sklearn.neighbors import NearestCentroid
from scipy.spatial import distance
import pandas as pd
data, label = tratamentoDados("OHE")

clf = clf = NearestCentroid()
clf.fit(data, label[0])
centroids = pd.DataFrame(clf.centroids_)
labels = pd.DataFrame(clf.classes_)

distancia_media =[0]*len(classes)
for i in range(len(classes)):
    docs = list(label[0].where(label[0]==classes[i]).dropna().index)
    documentos = data[data.columns].iloc[docs]
    centroid = centroids.iloc[labels[0].where(labels[0]==classes[i]).dropna().index[0]]
    distancia = 0
    for j in range(len(documentos)):
        distancia += distance.euclidean(documentos.iloc[j],centroid)
    distancia = distancia/len(documentos)
    distancia_media[i]=distancia

# Qunatidade de elementos e subelementos das classes erroneas
sub_quant = [0]*len(classes)
elem_quant = [0]*len(classes)
for i in range(len(classes)):
    sub_quant[i] = label[0].where(label[0]==classes[i]).dropna().value_counts().values[0]
    if(len(str(classes[i]))==3):
        elem_quant[i] = label_elemento[0].where(label_elemento[0]==str(classes[i])[0]).dropna().value_counts().values[0]
    else:
        elem_quant[i] = label_elemento[0].where(label_elemento[0]==str(classes[i])[:2]).dropna().value_counts().values[0]

# Relacao F1 X Porcentagem de documentos (subelementos)
import matplotlib.pyplot as plt
informacoes = pd.DataFrame()
#informacoes = pd.concat([informacoes,pd.DataFrame(label[0].value_counts().sort_index().index), pd.DataFrame(label[0].value_counts().sort_index().values/324705 *100) ],axis = 1)
informacoes = pd.concat([informacoes,pd.DataFrame(label[0].value_counts().sort_index().index), pd.DataFrame(label[0].value_counts().sort_index().values/324705 *100) ],axis = 1)
media_acertos = [0.82,0.88,0.92,0.90,0.89,0.94,0.77,0.87,0.99,0.97,0.88,0.85,0.65,0.88,0.33,0.98,0.93,0.69,0.97,0.76,0.90,0.89,0.82,0.48,0.97,0.85,0.81,0.70,0.82,0.48,0.96,0.86,0.68,0.53,0.22,0.86,0.58,0.89,0.56,0.91,0.93,0.99,0.53,0.00,0.95,0.95,0.98,0.89,0.66,0.43,0.27,0.86,0.93,0.95,0.48,0.95,0.67,0.83,0.82,0.92,0.91,0.91,0.92,0.95,0.89,0.78,0.98,0.88,0.94,0.95,0.96,0.97,0.94,0.92,0.93,0.97,0.97,0.98,0.88,0.96,0.93,0.99,0.96,0.97,0.72,0.71,0.95,0.90,0.83,0.17,0.91,0.80,0.89,0.86,0.95,0.87,0.79,0.95,0.95,0.92,0.88,0.87,0.93,0.96,0.90,0.97,0.92,0.94,0.96,0.92,0.56,0.88,0.83,0.97,0.78,0.73,0.80,0.80,0.89,0.88,0.97,0.85,0.92,0.91,0.96,0.89,0.86,0.77,0.92,0.89,0.91,0.00,0.77,0.60,0.83,0.72,0.53,0.99,0.90,1.00,0.62,0.98,0.90,0.71,0.86,0.53,0.89,0.87,0.77,0.77,0.75,0.75,0.78,0.70,0.13,0.63,0.53,0.54,0.50,0.60,0.69,0.61,0.81,0.76,0.79,0.81,0.48,0.38,0.68,0.58,0.66,0.67,0.00,0.49,0.68,0.56,0.62,0.68,0.63,0.39,0.54,0.34,0.42,0.62,0.79,0.59,0.60,0.44,0.07,0.37,0.63,0.00,0.96,0.83,0.36,0.84,0.91,0.66,0.66,0.55,0.73,0.95,0.84,0.44,0.22,0.68,0.46,0.86,0.48,0.82,0.74,0.89,0.83,0.79,0.95,0.87,0.87,0.89,0.35,0.53,0.00,0.35,0.58,0.81,0.89,0.69,0.00,0.23,0.80,0.33,0.91,0.25,0.31,0.11,0.99,0.23,0.55,0.00,0.93,0.96,0.95,0.98,0.13,0.83,0.89,0.52,0.40,0.69,0.49,0.67,0.22,0.90,0.56,0.52,0.74,0.96,0.59,0.00,0.46,0.78,0.00,0.86,0.90,0.76,0.86,0.90,0.12,0.70,0.73,0.12,0.83,0.89,0.86,0.93,0.82,0.78,0.47,0.83,0.77,0.87,0.79,0.77,0.68,0.59,0.73,0.90,0.60,0.75,0.94,0.97,0.94,0.69,0.79,0.67,0.78,0.90,0.85,0.91,0.80,0.72,0.78,0.92,0.71,0.90,0.99,0.81,0.94,0.71,0.64,0.91,0.73,0.90,0.00,0.00,0.88,0.88,0.19,0.78,0.00,0.92,0.80,0.81,0.66,0.11,0.67,0.79,0.32,0.85,0.79,0.87,0.33,0.66,0.43,0.97,0.74,0.91,0.78,0.62,0.89,0.88,0.72,0.97,0.92,1.00,0.51,0.61,0.33,0.59,0.78,0.51,0.69,0.79,0.97,0.97,0.83,0.47,0.97,0.96,0.96,0.67,0.55,0.33,0.80,0.98,0.86,0.98,0.88,0.00,0.98,0.38,0.95,0.73,0.00,0.94,0.93,0.89,0.83,0.93,0.95,0.84,0.87,0.92,0.92,0.62,0.76,0.89,0.94,0.83,0.86,0.82,0.96,0.96,0.87,0.66,0.58,0.70,0.80,1.00,0.47,0.86,0.58,0.91,0.90,0.00,0.79,0.65,0.36,0.57,0.43,0.82,0.29,0.79,0.77,0.72,0.57,0.52,0.57,0.69,0.88,0.12,0.07,0.33,0.21,0.57,0.68,0.48,0.76,0.55,0.97,0.79,0.74,0.88,0.59,0.51,0.54,0.83,0.17,0.00,0.80,0.97,0.96,0.99,0.99,0.97,0.82,0.96,0.62,0.67,1.00,1.00,1.00,0.94,1.00,1.00,0.96,0.86,0.92,0.90,0.76,0.86,0.75,0.80,0.93,0.74,0.67,0.28,0.85,0.51,0.00,0.76,0.13,0.74,0.73,0.64,0.91,0.47,0.81,0.78,0.72,0.42,0.78,0.89,0.49,0.77,0.48,0.67,0.67,0.70,0.11,0.33,0.32,0.69,0.76,0.77,0.34,0.36,0.29,0.28,0.89,0.77,0.42,0.40,0.31,0.33,0.50,0.59,0.23,0.57,0.67,0.58,0.39,0.44,0.48,0.83,0.37,0.66,0.64,0.42,0.39,0.00,0.61,0.40,0.46,0.53,0.32,0.38,0.36,0.62,0.52,0.34,0.82,0.47,0.50,0.49,0.46,0.28,0.68,0.46,0.64,0.46,0.73,0.73,0.56,0.94,0.81,0.77,0.84,0.90,0.67,0.84,0.60,0.86,0.99,0.93,0.85,0.93,0.87]
informacoes = pd.concat([informacoes,pd.DataFrame(media_acertos)],axis =1)
informacoes.columns = ["natureza","porcentagem","acuracia"]
X = informacoes["natureza"]
Y = informacoes["porcentagem"]
linha = informacoes["acuracia"]
plt.bar(X,Y)
plt.plot(linha)
plt.show()

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
informacoes['porcentagem']= min_max_scaler.fit_transform(informacoes['porcentagem'].values.reshape(-1, 1))

# Qunatidade de elementos as classes erroneas
elem_quant = [0]*len(classes)
for i in range(len(classes)):
    elem_quant[i] = label.where(label[0]==classes[i]).dropna().value_counts().values[0]

# Materialidade das classes erroneas
materialidade = [0]*len(classes)
for i in range(len(classes)):
    # Media da materialidade da natureza
    materialidade[i] = int(sum(data["valor_saldo_do_empenho"].iloc[label.where(label[0]==classes[i]).dropna().index].values)/label.where(label[0]==classes[i]).dropna().value_counts().values[0])


# Balanceamento das classses
from yellowbrick.target import ClassBalance
visualizer = ClassBalance(labels=label[0].value_counts().index)
visualizer.fit(label[0])        # Fit the data to the visualizer
visualizer.show()   




valor = 9237
data['natureza_despesa_nome'].iloc[label.where(label[0]==valor).dropna().index].value_counts().count()
data['natureza_despesa_nome'].iloc[label.where(label[0]==valor).dropna().index].value_counts()
label.where(label[0]==817).dropna().value_counts()

# Código que quantos nomes cada natureza tem
label = []
for i in range(data.shape[0]):
  label.append(data['natureza_despesa_cod'].iloc[i].split(".")[3]+data['natureza_despesa_cod'].iloc[i].split(".")[4] )
label = pd.DataFrame(label).astype("int16")
label = pd.DataFrame(list(label[0].value_counts().index))
label = label.sort_values(by=[0])
label.reset_index(drop=True, inplace=True)

contador = [0]*label[0].value_counts().count()
for i in range(label[0].value_counts().count()):
    contador[i] =data['natureza_despesa_nome'].where(label[0] == label[0].value_counts().sort_index().index[i]).dropna().value_counts().count()

resultado = pd.concat([pd.DataFrame(label[0].value_counts().sort_index().index),pd.DataFrame(contador)],axis = 1)
resultado.columns = ["0","1"]
data['natureza_despesa_nome'].where(label[0] == 9302).dropna().value_counts()

# Porcentagem e taxa de acerto dos errantes
indexes = []
for i in range(len(informacoes)):
    if((informacoes["natureza"].iloc[i] == classes).any()):
        indexes.append(i)
errantes = informacoes.iloc[indexes]
errantes['porcentagem'] = min_max_scaler.fit_transform(errantes['porcentagem'].values.reshape(-1, 1))
