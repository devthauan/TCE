### Meus pacotes ###
from modelos import rocchio
from tratamentos import pickles
from modelos import randomforest
from scipy.sparse import csr_matrix
#from tratamentos import tratar_texto
from modelos import supportVectorMachine
from preparacaoDados import tratamentoDados
### Imports
from sklearn.model_selection import train_test_split

# Dados do OneHotEncoding
data, label = tratamentoDados("OHE")
#del data
data = csr_matrix(data)
#data = data.astype('float16')
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label,random_state =2)
#del data
# Salva as variáveis em formato pkl
#pickles.criaPickle(X_train,"X_train_OHE")
#pickles.criaPickle(X_test,"X_test_OHE")
#pickles.criaPickle(y_train,"y_train_OHE")
#pickles.criaPickle(y_test,"y_test_OHE")

# Salvando em arquivo as naturezas
#f = open('Resultados.txt','a+')
#f.write("naturezas"+'\n')
#for index in (label[0].value_counts().sort_index().index):
#    f.write(str(index)+'\n')
#f.write('\n')
#f.flush()
#f.close()

# Random Forest COM OneHotEncoding    
randomforest.randomForest(X_train, X_test, y_train, y_test,"COM OHE")

# Rocchio COM oneHotEncoding
#rocchio.rocchio(X_train, X_test, y_train, y_test,"COM OHE")

## Support Vector Classifier COM oneHotEncoding
### Valores para o parametro C do SVC
#parametroc = [0.001,1,10,100,1000,1200,1500,1800,2000,2100,2200,2300,2400,2500,3000,3500,4000]
#for parC in parametroc:
#    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM OHE",parC)
#supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM OHE",2000)
#del X_train, X_test, y_train, y_test
################################################################################
## Dados do TF-IDF
tfidf = tratamentoDados("tfidf") 
X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.3,stratify = label,random_state =4)
#del tfidf
## Salva as variáveis em formato pkl
#pickles.criaPickle(X_train,"X_train_tfidf")
#pickles.criaPickle(X_test,"X_test_tfidf")
#pickles.criaPickle(y_train,"y_train_tfidf")
#pickles.criaPickle(y_test,"y_test_tfidf")
#
## Random Forest COM TF-IDF
randomforest.randomForest(X_train, X_test, y_train, y_test,"COM TFIDF")
#
## Rocchio COM TF-IDF
#rocchio.rocchio(X_train, X_test, y_train, y_test,"COM TFIDF")
#
#### Support Vector Classifier COM TF-IDF
#parametroc = [0.001,1,10,100,1000,1200,1500,1800,2000,2100,2200,2300,2400,2500,3000,3500,4000]
#for parC in parametroc:
#    supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM TFIDF",parC)
#supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM TFIDF",10)
###############################################################################