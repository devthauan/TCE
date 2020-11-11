import numpy as np
import pandas as pd
from modelos import knn
from modelos import cenknn
from modelos import randomforest
from modelos import supportVectorMachine
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from scipy.sparse import csr_matrix
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split

data, label = tratamentoDados("sem OHE")
texto = tratamentoDados("texto") 


cv = TfidfVectorizer(dtype=np.float32, ngram_range=(1,2))
data_cv = cv.fit_transform(texto)
tfidf = pd.DataFrame.sparse.from_spmatrix(data_cv, columns = cv.get_feature_names())
# Pegando 15% estratificado
X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.15,stratify = label,random_state =4)
print("calculou o tfidf")
from modelos import feature_importance
print("Fazendo selecao de features")
selecao, colunasMaisImportantes = feature_importance.featureImportance(X_test,y_test,1,-1)
print("fez a selecao")
tfidf = tfidf[selecao.columns]
print("tamanho do tfidf ",tfidf.shape)
tfidf = csr_matrix(tfidf)
print("Rodando o randomforest")
X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.3,stratify = label,random_state =4)
randomforest.randomForest(X_train, X_test, y_train, y_test,"tfidf 2grama no RF é")


#valores = [1,2,3,5,10,20,50,100]
#for valor in valores:
#    knn.knn(X_train, X_test, y_train, y_test,"sem selecao",valor)
print("Rodando o knn")
knn.knn(X_train, X_test, y_train, y_test,"sem selecao",1)
