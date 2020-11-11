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
from sklearn.model_selection import train_test_split

from modelos import feature_importance
data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf")
aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
data =  pd.DataFrame.sparse.from_spmatrix(aux)
del tfidf
dados, colunasMaisImportantes = feature_importance.featureImportance(data,label,1,0.9)
print("tamanho reduzido "+str(dados.shape))


#data, label = tratamentoDados("OHE")
#tfidf = tratamentoDados("tfidf") 
#dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
#dados =  pd.DataFrame.sparse.from_spmatrix(dados)
#dados = pd.concat([data,tfidf],axis = 1)
#del data,tfidf
# outliers para serem removidos
outliers = pd.read_csv("outliers_DBSCAN.xlsx")

# deletando outliers da label
label = label.drop(outliers["Index dos Outliers"].values)
label.reset_index(drop=True, inplace=True)

#deletando os outliers do dataset
dados.drop(list(outliers["Index dos Outliers"].values),inplace=True)
dados.reset_index(drop=True, inplace=True)

# pegando classes com so 1 doc
dropar = []
for i in range(label[0].value_counts().count()):
      if(label[0].value_counts().iloc[i]  == 1):
          dropar.append(label[0].value_counts().index[i])

# pegando indices dessas classes
indices =[0]*len(dropar)         
for i in range(len(dropar)):
    indices[i] = label[0].where(label[0]==dropar[i]).dropna().index[0]

#deletando classes com 1 so doc da label
label = label.drop(indices)
label.reset_index(drop=True, inplace=True)

#deletando classes com 1 so doc do dataset
dados = dados.drop(indices)
dados.reset_index(drop=True, inplace=True)
del indices, outliers, i, dropar

dados = csr_matrix(dados)
X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)
#ja testei com 1.8 e nao deu
valores = np.arange(0.1,10,0.1)
for valor in valores:
    try:
        radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",round(valor,1))
    except:
        continue
#radiusknn.radius(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",3.1)
#from tratamentos import pickles
##pickles.criaPickle(dados,"dados_radius")
##pickles.criaPickle(label,"label_radius")
#dados = pickles.carregaPickle("dados_radius")
#label = pickles.carregaPickle("label_radius")
#pickles.criaPickle(X_train,"X_train_radius")
#pickles.criaPickle(y_train,"y_train_radius")
#pickles.criaPickle(X_test,"X_test_radius")
#pickles.criaPickle(y_test,"y_test_radius")
