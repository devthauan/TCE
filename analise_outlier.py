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
data, label = tratamentoDados("OHE")


# =============================================================================
# Achar melhor eps
# =============================================================================
#min_sample = 2
#from sklearn.neighbors import NearestNeighbors
#nbrs = NearestNeighbors(n_neighbors=min_sample).fit(data)
#distances, indices = nbrs.kneighbors(data)
#distanceDec = sorted(distances[:,min_sample-1], reverse=True)
#plt.figure(figsize=(20,15))
#plt.plot(indices[:,0], distanceDec)
#plt.grid()
#plt.xticks(range(0,len(indices[:,0]),50))
##plt.yticks(np.arange(0,max(distanceDec),0.1))
#plt.show()

#for min_sample in np.arange(2,10,1):
#    for eps in np.arange(0.1,2,0.1):
#        labels_dbscan = dbscan.dbscan(data,eps,min_sample)
#        shilhouette_result = silhouette_score(data, labels_dbscan)
#        pd.DataFrame(labels_dbscan)[0].value_counts()
#        print("eps: ",eps," min_sample: ",min_sample," Silhouette score DBscan é: ",shilhouette_result," e foi dividido em ",pd.DataFrame(labels_dbscan)[0].value_counts().count()," clusters")

eps = 1
min_sample =2

#X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label)
clusters_dbscan = pd.DataFrame(dbscan.dbscan(data,eps,min_sample))
# Posicao dos outliers
position_outliers = clusters_dbscan.where(clusters_dbscan == -1).dropna().index

#outliers = pd.DataFrame(data.iloc[position_outliers])
# naturezas dos outliers
#outliers_labels = pd.DataFrame(label[0].iloc[position_outliers])
#outliers_labels[0].value_counts()

#analise_cluster.analisecluster(label,clusters_dbscan)
f = open('outliers_DBSCAN.xlsx','a+')
f.write("Index dos Outliers"+'\n')
for pos in position_outliers:
    f.write(str(pos)+'\n')
f.flush()
f.close()
print(len(position_outliers)," Outliers DBSCAN")
# =============================================================================
# # Gaussian Mixture
# =============================================================================
# Separa os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label,random_state= 0)


#for i in [10,20,30,50,100,150,200,300,400,500,550,600]:
#    num_mixtures = i
#    predicao_gaussiana, prob = gaussian_mixtures.gaussian(X_train, X_test, num_mixtures)
#    shilhouette_result = silhouette_score(X_test, predicao_gaussiana)
#    print(i,"Silhouette Gaussian Mixtures é: ",shilhouette_result," e foi dividido em ",pd.DataFrame(predicao_gaussiana)[0].value_counts().count()," clusters")
#    analise_cluster.analisecluster(y_train,predicao_gaussiana)

num_mixtures = 200
predicao_gaussiana, probabilidade_gaussiana = gaussian_mixtures.gaussian(X_train, X_test, num_mixtures)
probabilidade_gaussiana = pd.DataFrame(probabilidade_gaussiana)
#probabilidade_gaussiana.iloc[17].max()
linhas_outlier_gaussiano = []
for i in range(probabilidade_gaussiana.shape[0]):
    if(probabilidade_gaussiana.iloc[i].max() < 0.5):
        linhas_outlier_gaussiano.append(i)

f = open('outliers_GauMix.xlsx','a+')
f.write("Index dos Outliers"+'\n')
for pos in linhas_outlier_gaussiano:
    f.write(str(pos)+'\n')
f.flush()
f.close()
print(len(linhas_outlier_gaussiano)," Outliers Gaussian Mixtures")




