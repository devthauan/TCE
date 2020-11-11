from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

#silhouette score para descidir o melhor K do algorítmo KMeans
def bestK(data):
    resultados = []
    valores_de_K = [2,4,10,15,20,50,100,120,140,170,200,250,300,350,400,450,500,550,600,650,700]
    for i in (valores_de_K):
        kmenas_model = KMeans(n_clusters=i, init='k-means++',n_init = 1,random_state = 0)
        cluster_labels = kmenas_model.fit_predict(data)  
        resultados.append(silhouette_score(data, cluster_labels))
    plt.plot(valores_de_K,resultados)
    plt.show()
    print(resultados)
    maior = [valores_de_K[resultados.index(max(resultados))],max(resultados)]
    return maior

#Após a escolha do K a implementação do KMeans #n_init numero de vezes que vai rodar em diferentes centroids
def kmeans(best_k,data):
    kmenas_model = KMeans(n_clusters=best_k, init='k-means++',n_init = 1)
    kmenas_model.fit(data)
    labels = kmenas_model.labels_
    
    return labels
