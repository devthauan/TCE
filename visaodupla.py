import numpy as np
import pandas as pd
from modelos import knn
from scipy import sparse
from modelos import rocchio
from modelos import radiusknn
from modelos import randomforest
from scipy.sparse import csr_matrix
from modelos import supportVectorMachine
from modelos import feature_importance
from preparacaoDados import tratamentoDados
#from preparacaoDados2 import tratamentoDados
from sklearn.model_selection import train_test_split

data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
#dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
#del data,tfidf

## =============================================================================
## EXCLUINDO O 92
## =============================================================================
#index = label.where(label['natureza_despesa_cod'].str.contains('.\..\...\.92\...',regex=True)).dropna().index
#data.drop(index,inplace=True)
#label.drop(index,inplace=True)
#tfidf.drop(index,inplace=True)
#
#data.reset_index(drop=True,inplace=True)
#label.reset_index(drop=True,inplace=True)
#tfidf.reset_index(drop=True,inplace=True)
## =============================================================================
## EXCLUINDO O 92
## =============================================================================
aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
dados =  pd.DataFrame.sparse.from_spmatrix(aux)
dados, colunasMaisImportantes = feature_importance.featureImportance(dados,label,1,0.91)

dados = csr_matrix(dados)
X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)

# Random forest
randomforest.randomForest(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS")
# Rocchio
rocchio.rocchio(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS")

#valores = [1,2,3,5,10,20,50,100]
#for valor in valores:
#    knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",valor)
knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",1)

#
#possibilidade = [0.001,1,10,100,1000,2000,2500,3000]
#for pos in possibilidade:
#    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",pos)
supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",10)

# =============================================================================
# RADIUS_NEIGHBOR
# =============================================================================
#valores = np.arange(0.1,3,0.1)
#for valor in valores:
#    try:
#        radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",round(valor,1))
#    except:
#        continue
radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",2.7)
