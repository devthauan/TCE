from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from modelos import analise_cluster
import pandas as pd

def gaussian(X_train, X_test, num_component = 1):
    clf = GaussianMixture(n_components = num_component)
    clf.fit(X_train)
    prob = clf.predict_proba(X_test)
    y_predito = clf.predict(X_test)
    return y_predito , prob

def cluster(X_train, X_test, y_train, y_test, num_components):
    clf = GaussianMixture(n_components = num_components, covariance_type ='full', init_params = 'kmeans')
    clf.fit(X_train)
    y_predito = clf.predict(X_test)
    analise_cluster.analisecluster(y_test,y_predito)

#y_train[0].value_counts().count()