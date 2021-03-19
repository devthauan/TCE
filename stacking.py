import numpy as np
import pandas as pd
from modelos import knn
from sklearn.svm import SVC
from modelos import rocchio
from modelos import randomforest
from modelos import supportVectorMachine
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from preparacaoDados import tratamentoDados
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split

data, label = tratamentoDados("OHE")
data = pd.DataFrame.sparse.from_spmatrix(csr_matrix(data))
tfidf = tratamentoDados("tfidf")
       
print(data.shape)
print(tfidf.shape)
#possibilidade = [["rf","rf","rf"],["rf","rf","knn"],["rf","rf","svc"],["rf","rf","rocchio"],["rf","knn","rf"],["rf","knn","knn"],["rf","knn","svc"],["rf","knn","rocchio"],["rf","svc","rf"],["rf","svc","knn"],["rf","svc","svc"],["rf","svc","rocchio"],["knn","rf","rf"],["knn","rf","knn"],["knn","rf","svc"],["knn","rf","rocchio"],["knn","knn","rf"],["knn","knn","knn"],["knn","knn","svc"],["knn","knn","rocchio"],["knn","svc","rf"],["knn","svc","knn"],["knn","svc","svc"],["knn","svc","rocchio"],["svc","rf","rf"],["svc","rf","knn"],["svc","rf","svc"],["svc","rf","rocchio"],["svc","knn","rf"],["svc","knn","knn"],["svc","knn","svc"],["svc","knn","rocchio"],["svc","svc","rf"],["svc","svc","knn"],["svc","svc","svc"],["svc","svc","rocchio"]]
possibilidade = [['svc', 'rf', 'svc']]
for i in range(len(possibilidade)):
    #iniciando vetores
    resultadoOHE = pd.DataFrame([])
    resultadoTFIDF = pd.DataFrame([])
    rotulo = pd.DataFrame([])
    prob = possibilidade[i]
    #escolhendo os parametros
    algoritmo1 = prob[0]
    algoritmo2 = prob[1]
    algoritmo3 = prob[2]
    kf = KFold(n_splits=5,shuffle=True,random_state=0)
    for train_index, test_index in kf.split(data):
# =============================================================================
#         Primeira parte
# =============================================================================
#        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        X_train_text, X_test_text = tfidf.iloc[train_index], tfidf.iloc[test_index]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]
        # Gerando o vetor de probabilidade
        if(algoritmo1 =="rf"):
            y_prob_predito = randomforest.randomForest(X_train, X_test, y_train, y_test,"prob")
            resultadoOHE = pd.concat([resultadoOHE,pd.DataFrame(y_prob_predito)],axis=0)
        elif(algoritmo1 == "knn"):
#            for i in [1,2,3,5,10,20,50,100]: #teste de hiperparametro
            y_prob_predito = knn.knn(X_train, X_test, y_train, y_test,"prob",1)
            resultadoOHE = pd.concat([resultadoOHE,pd.DataFrame(y_prob_predito)],axis=0)
        else:
#            for i in [0.001,1,10,100,1000,2000,2500,3000]: #teste de hiperparametro
            y_prob_predito = supportVectorMachine.svc(X_train, X_test, y_train, y_test,"prob_sparse",100)
            resultadoOHE = pd.concat([resultadoOHE,pd.DataFrame(y_prob_predito)],axis=0)
# =============================================================================
#         Segunda parte
# =============================================================================
        if(algoritmo2 =="rf"):
            y_prob_predito_text = randomforest.randomForest(X_train_text, X_test_text, y_train, y_test,"prob")
            resultadoTFIDF = pd.concat([resultadoTFIDF,pd.DataFrame(y_prob_predito_text)],axis=0)
        elif(algoritmo2 == "knn"):
#            for i in [1,2,3,5,6,7,8,9,10]: #teste de hiperparametro
            y_prob_predito_text = knn.knn(X_train_text, X_test_text, y_train, y_test,"prob",1)
            resultadoTFIDF = pd.concat([resultadoTFIDF,pd.DataFrame(y_prob_predito_text)],axis=0)
        else:
#            for i in [0.001,1,10,100,1000,2000,2500,3000]: #teste de hiperparametro
            y_prob_predito_text = supportVectorMachine.svc(X_train_text, X_test_text, y_train, y_test,"prob_sparse",10)
            resultadoTFIDF = pd.concat([resultadoTFIDF,pd.DataFrame(y_prob_predito_text)],axis=0)
        #rotulo do test na ordem correta
        rotulo = pd.concat([rotulo,y_test],axis = 0)
# =============================================================================
#     Terceira parte
# =============================================================================
    dados = pd.concat([pd.DataFrame(resultadoOHE),pd.DataFrame(resultadoTFIDF)],axis = 1)
    dados = dados.fillna(0)
    # Salva os dados do stacking para usar no active learning
#    dados.to_csv('dados_stacking.csv', index=False)
    X_train, X_test, y_train, y_test = train_test_split(dados, rotulo,test_size=0.3,stratify = rotulo,random_state= 0)
    print(prob)
    if(algoritmo3 == "rf"):
        randomforest.randomForest(X_train, X_test, y_train, y_test,"stacking")
    elif(algoritmo3 == "rocchio"):
        rocchio.rocchio(X_train, X_test, y_train, y_test,"stacking")
    elif(algoritmo3 == "knn"):
#        for i in [1,2,3,5,6,7,8,9,10]: #teste de hiperparametro
        knn.knn(X_train, X_test, y_train, y_test,"stacking",1)
#        y_stacking = knn.knn(X_train, X_test, y_train, y_test,"stacking",1)
    else:
#        for i in [0.001,1,10,100,1000,2000,2500,3000]: #teste de hiperparametro
        supportVectorMachine.svc(X_train, X_test, y_train, y_test,"stacking",100)
