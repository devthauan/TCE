from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados
from scipy.sparse import csr_matrix
from modelos import cenknn
from scipy import sparse
import pandas as pd

data, label = tratamentoDados("sem OHE")
tfidf = tratamentoDados("tfidf") 
#dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
#dados =  pd.DataFrame.sparse.from_spmatrix(dados)
#tfidf =  pd.DataFrame.sparse.from_spmatrix(csr_matrix(tfidf))

#data = pickles.carregaPickle("data_reduzida")
#label = pickles.carregaPickle("label_reduzida")

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
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test,test_size=0.3,stratify = y_test,random_state =5)
#for i in range(1,6,1):
#    cenknn.cenknn(X_train, X_test, y_train, y_test,"com Cenknn", k_value = i) # numero de vizinhos
#import time
#start = time.time()
print(X_train.shape," X_train")
print(X_test.shape," X_test")
cenknn.cenknn(X_train, X_test, y_train, y_test,"com Cenknn", k_value = 2) # numero de vizinhos
#final = time.time()
#print(final-start)