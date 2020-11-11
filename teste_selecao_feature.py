import numpy as np
import pandas as pd
from modelos import knn
from scipy import sparse
from modelos import rocchio
from modelos import rocchio2
from modelos import radiusknn
from tratamentos import pickles
from modelos import randomforest
from scipy.sparse import csr_matrix
from modelos import feature_importance
from modelos import supportVectorMachine
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split

data = csr_matrix(pickles.carregaPickle("data_reduzida"))
label = pickles.carregaPickle("label_reduzida")
print(data.shape)
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label,random_state =2)

#randomforest.randomForest(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS")
supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",10)
