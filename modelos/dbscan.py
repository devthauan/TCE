from sklearn.cluster import DBSCAN
from modelos import analise_cluster
import pandas as pd

def dbscan(data,epsilon=0.5,minimum_sample=5):
    dbscan_model = DBSCAN(eps=epsilon, min_samples=minimum_sample,n_jobs=-1)
    dbscan_model.fit(data)
    # return the predicted clusters
    labels = dbscan_model.labels_    
    return labels

def cluster(X_train, X_test, y_train, y_test,epsilon=0.5,minimum_sample=5):
    dbscan_model = DBSCAN(eps=epsilon, min_samples=minimum_sample,n_jobs=-1)
    dbscan_model.fit(X_train)
    y_predito = dbscan_model.labels_
    analise_cluster.analisecluster(y_test,y_predito)
