import numpy as np
import pandas as pd
import xgboost as xgb
from modelos import knn
from modelos import sgd
from sklearn.svm import SVC
from scipy import sparse
from modelos import xboost
from modelos import naive_bayes
from tratamentos import pickles
from modelos import randomforest
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from modelos import supportVectorMachine
from sklearn.model_selection import KFold
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

vetor = ['estrategia[1, 1]','estrategia[1, 2]','estrategia[2, 1]','estrategia[2, 2]']
for i in range(len(vetor)):
    modelo = pickles.carregaPickle(vetor[i])
    plt.figure(figsize=(16,11))
    plt.title(vetor[i],fontsize= 20)#y=1.08
    plt.rcParams.update({'font.size': 12})
    print("Tamanho: ",modelo[0].value_counts().count())
    plt.bar(modelo[0].value_counts(ascending=True).index.astype("str"),(modelo[0].value_counts(ascending=True).values))
    if(i>1):
        plt.xticks([])
    else:
        plt.xticks()
    plt.show()
    