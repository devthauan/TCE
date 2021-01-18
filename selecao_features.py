# Packages python
import pandas as pd
from sklearn.model_selection import train_test_split

# Meus packages
from modelos import single_value_decomposition
from preparacaoDados import tratamentoDados
from modelos import supportVectorMachine
from modelos import feature_importance
from scipy.sparse import csr_matrix
from modelos import randomforest
from scipy import sparse
from modelos import knn
import numpy as np



data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf")
aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
data =  pd.DataFrame.sparse.from_spmatrix(aux)
del tfidf
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"sem feature selection com os dados combinados no RF")
#valores = [1,2,3,5,10,20,50,100]
#for valor in valores:
#    knn.knn(X_train, X_test, y_train, y_test,"sem selecao",valor)
knn.knn(X_train, X_test, y_train, y_test,"sem selecao",1)
#possibilidade = [0.001,1,10,100,1000,2000,2500,3000]
#for pos in possibilidade:
#    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",pos)
supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM TFIDF",10)
print("")
# =============================================================================
# =============================================================================
for i in np.arange(0.1,1,0.1):
    # Seleção de atributos usando Feature Importance
    print("tamanho original "+str(data.shape))
    data_feature_importance, colunasMaisImportantes = feature_importance.featureImportance(data,label,1,i)
    print("tamanho reduzido "+str(data_feature_importance.shape))
    X_train, X_test, y_train, y_test = train_test_split(csr_matrix(data_feature_importance), label,test_size=0.3,stratify = label,random_state =5)
    randomforest.randomForest(X_train, X_test, y_train, y_test,"feature importance com os dados combinados no RF")

    knn.knn(X_train, X_test, y_train, y_test,"feature importance",1)
    X_train, X_test, y_train, y_test = train_test_split(data_feature_importance, label,test_size=0.3,stratify = label,random_state =5)
    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM TFIDF",10)

#
#    num_atributos = data_feature_importance.shape[1]
#    del data_feature_importance
#    print("")
#    # =============================================================================
#    # =============================================================================
#    # Seleção de atributos usando Principal Component Analysis
#    from sklearn.decomposition import PCA
#    pca = PCA(n_components = num_atributos).fit(data.to_numpy())
#    #print(pca.explained_variance_ratio_)
#    data_pca = pca.transform(data.to_numpy())
#
#    X_train, X_test, y_train, y_test = train_test_split(data_pca, label,test_size=0.3,stratify = label,random_state =2)
#    randomforest.randomForest(X_train, X_test, y_train, y_test,"PCA com os dados combinados no RF")
#    #valores = [1,2,3,5,10,20,50,100]
#    #for valor in valores:
#    #    knn.knn(X_train, X_test, y_train, y_test,"PCA",valor)
#
#    knn.knn(X_train, X_test, y_train, y_test,"PCA",1)
#
#    del data_pca
#    print("")
