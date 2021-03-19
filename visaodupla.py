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
from modelos import xboosting
from modelos import xboost
from tratamentos import pickles
from modelos import sgd
from modelos import naive_bayes
from sklearn import preprocessing
#from preparacaoDados2 import tratamentoDados
from sklearn.model_selection import train_test_split

data = pickles.carregaPickle("data")
label = pickles.carregaPickle("label")
tfidf = pickles.carregaPickle("tfidf")
aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
dados =  pd.DataFrame.sparse.from_spmatrix(aux)
dados = csr_matrix(dados)
del data
print(dados.shape)
#X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)



le = preprocessing.LabelEncoder()
label['natureza_despesa_cod'] = le.fit_transform(label['natureza_despesa_cod'])#LABEL ENCODER
X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =10)
#valores = [1,10,100,1000,2000,2500,3000]
#for valor in valores:
#    xboost.xboost(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",valor)
xboost.xboost(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",100,label.value_counts().count())

naive_bayes.naivebayes(X_train.toarray(), X_test.toarray(), y_train, y_test,"COM DADOS COMBINADOS")
sgd.sgd(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",100)
xboosting.xboost(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",200)


# Random forest
randomforest.randomForest(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS")
# Rocchio
#rocchio.rocchio(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS")

#valores = [1,2,3,5,10,20,50,100]
#for valor in valores:
#    knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",valor)
knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",1)

#SVC com parametro menor
#supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",0.1)
#possibilidade = [0.001,1,10,100,1000,2000,2500,3000]
#for pos in possibilidade:
#    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",pos)
#supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",10)
# =============================================================================
# RADIUS_NEIGHBOR
# =============================================================================
#valores = np.arange(3,3.6,0.1)
#for valor in valores:
#    try:
#        radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",round(valor,1))
#    except:
#        continue
#radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",3.2)


#from tratamentos import pickles
#pickles.criaPickle(dados,"dados_complex")
#pickles.criaPickle(label,"label_complex")
#dados["labels"]= label
#dados = dados.drop(0,axis = 1)
#pickles.criaPickle(dados,"dados_complex_combinado")
