import numpy as np
import pandas as pd
from modelos import knn
from scipy import sparse
from modelos import rocchio
from modelos import rocchio2
from modelos import radiusknn
from modelos import randomforest
from scipy.sparse import csr_matrix
from modelos import supportVectorMachine
from preparacaoDados import tratamentoDados
#from preparacaoDados2 import tratamentoDados
from sklearn.model_selection import train_test_split

data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
#data = data.astype('float16')
dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
del data,tfidf
#dados =  pd.DataFrame.sparse.from_spmatrix(dados)
#dados = pd.concat([data,tfidf],axis=1 )
#dados = dados.astype(pd.SparseDtype("float16", 0))
#dados = csr_matrix(dados)
#dados = dados.astype(pd.SparseDtype("float16", 0))
#del data, tfidf
#from modelos import feature_importance
#print("tamanho original "+str(dados.shape))
#data_feature_importance, colunasMaisImportantes = feature_importance.featureImportance(dados,label,1,0.92)
#print("tamanho reduzido "+str(data_feature_importance.shape))
#X_train, X_test, y_train, y_test = train_test_split(csr_matrix(data_feature_importance), label,test_size=0.3,stratify = label,random_state =2)
#
X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)
#randomforest.randomForest(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS")

#rocchio.rocchio(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS")

#valores = [1,2,3,5,10,20,50,100]
#for valor in valores:
#    knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",valor)
#knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",1)#testado com 10k

#rocchio2.rocchio2(X_train.toarray(), X_test.toarray(), y_train, y_test,"COM DADOS COMBINADOS")

#possibilidade = [0.001,1,10,100,1000,2000,2500,3000]
#for pos in possibilidade:
#    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",pos)
supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",10)
#y_visaodupla = supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",10)#testado com 10k


#from tratamentos import pickles
#pickles.criaPickle(X_train,"X_train")
#pickles.criaPickle(X_test,"X_test")   
#pickles.criaPickle(y_train,"y_train")
#pickles.criaPickle(y_test,"y_test")
# =============================================================================
# RADIUS_NEIGHBOR
# =============================================================================

#ja testei com 1.8 e nao deu
#valores = np.arange(0.1,3,0.1)
#for valor in valores:
#    try:
#        radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",round(valor,1))
#    except:
#        continue
#radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",valor)
