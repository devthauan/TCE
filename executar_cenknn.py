from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados
from scipy.sparse import csr_matrix
from modelos import cenknn
from scipy import sparse
import pandas as pd

data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
dados =  pd.DataFrame.sparse.from_spmatrix(aux)
X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)
#for i in range(1,6,1):
#    cenknn.cenknn(X_train, X_test, y_train, y_test,"com Cenknn", k_value = i) # numero de vizinhos
#import time
#start = time.time()
print(X_train.shape," X_train")
print(X_test.shape," X_test")
cenknn.cenknn(X_train, X_test, y_train, y_test,"com Cenknn", k_value = 2) # numero de vizinhos
#final = time.time()
#print(final-start)