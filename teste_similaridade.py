import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from tratamentos import pickles
from scipy.sparse import csr_matrix
from tratarDados import tratarDados
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

data = csr_matrix(pickles.carregaPickle("data")[:10000])
tfidf = csr_matrix(pickles.carregaPickle("tfidf")[:10000])
data = sparse.hstack((data,tfidf))
del tfidf
label = pickles.carregaPickle("label")[:10000]
dados_novos = pickles.carregaPickle("dados_test")
dados_novos.reset_index(drop= True,inplace = True)
dados_novos, label_nova = tratarDados(dados_novos)

# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,random_state =10)
nbrs = NearestNeighbors().fit(X_train)
distances, indices = nbrs.kneighbors(X_test)
distances = pd.DataFrame(distances)
media_distancias = [0]*len(distances)
for i in range(len(media_distancias)):
    media_distancias[i] = np.mean(distances.iloc[i].values)
media_distancias = pd.DataFrame(media_distancias)
# =============================================================================
distances, indices = nbrs.kneighbors(dados_novos)
distances = pd.DataFrame(distances)
media_distancias_novas = [0]*len(distances)
for i in range(len(media_distancias_novas)):
    media_distancias_novas[i] = np.mean(distances.iloc[i].values)
media_distancias_novas = pd.DataFrame(media_distancias_novas)
# =============================================================================
result = pd.concat([ pd.DataFrame(np.sort(media_distancias[0])), pd.DataFrame(np.sort(media_distancias_novas[0]))],axis = 1)
result.to_csv("similaridade",index = False)
# =============================================================================
data =  sparse.vstack((data,csr_matrix(dados_novos)))
label = pd.concat([label,label_nova],axis = 0)
label.reset_index(drop = True, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,random_state =10)
modelo = SVC(kernel="linear",C= 10,random_state=0)
modelo.fit(X_train, y_train)
#X_test = pd.concat([pd.DataFrame(X_test),dados_novos],axis = 0)
y_predito = modelo.predict(X_test)
micro = f1_score(y_test,y_predito,average='micro')
macro = f1_score(y_test,y_predito,average='macro')
string = ""
valor_c = 10
print("O f1Score micro do SVC ",string," com parametro C = ",valor_c,"é: ",micro)
print("O f1Score macro do SVC ",string," com parametro C = ",valor_c,"é: ",macro)
# =============================================================================
y_predito = pd.DataFrame(y_predito)
index_original = y_test.index
y_test.reset_index(drop = True, inplace = True)
y_predito.reset_index(drop = True, inplace = True)
comparacao = pd.concat([y_test,y_predito],axis = 1)
comparacao.columns = ['real','predito']
comparacao['resultado'] = comparacao['real'] == comparacao['predito']
#comparacao['resultado'].value_counts()
comparacao['index'] = index_original
#
index = []
for i in range(len(comparacao)):
    if(comparacao['index'][i] >9999):
        index.append(i)

comparacao_novos = comparacao.iloc[index]
comparacao_novos['resultado'].value_counts()

