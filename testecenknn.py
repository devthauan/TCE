import numpy as np
import pandas as pd
from modelos import cenknn
from modelos import randomforest
from modelos import supportVectorMachine
from scipy import sparse
from scipy.sparse import csr_matrix
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split



data, label = tratamentoDados("sem OHE")
tfidf = tratamentoDados("tfidf") 

# =============================================================================
label = label.astype("str")
classes = list(label.where(label[0].str.contains('11..',regex = True)).dropna().value_counts().index)
index = []
for i in range(len(classes)):
    index.append(label.where(label[0] == classes[i][0]).dropna().index)
index = [item for sublist in index for item in sublist]
tfidf.drop(index,inplace = True)
label.drop(index,inplace = True)
tfidf.reset_index(drop=True, inplace=True)
label.reset_index(drop=True, inplace=True)
del classes,index,i,data
# =============================================================================


#X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)
X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.07,stratify = label,random_state =5)
# pegando apenas 16k docs
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test,test_size=0.3,random_state =5)




centroids = cenknn.centroid_calc(X_train,y_train) #Ok    
X_train = pd.DataFrame(cenknn.similarity_paralela(X_train,centroids)) #OK

centroids = cenknn.centroid_calc(X_test,y_test)
X_test = pd.DataFrame(cenknn.similarity_paralela(X_test,centroids)) #OK

randomforest.randomForest(X_train, X_test, y_train, y_test,"com os dados do cenknn no RF")
#
#possibilidade = [0.001,1,10,100,1000,2000,2500,3000]
#for pos in possibilidade:
#    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"com os dados do cenknn no SVC",pos)
supportVectorMachine.svc(X_train, X_test, y_train, y_test,"com os dados do cenknn no SVC",340)

#valores = [1,2,3,5,10,20,50,100]
#for valor in valores:
#    knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",valor)
#knn.knn(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",1)#testado com 10k