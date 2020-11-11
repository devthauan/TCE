from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados
from scipy.sparse import csr_matrix
from gensim.models import Word2Vec
from modelos import randomforest
from scipy import sparse
import pandas as pd
import numpy as np
import re
# retorna o texto tratado e limpo
texto_tratado = tratamentoDados("texto")
data,label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf")
# tokeniza os textos
sentences_ted = []
for sent_str in texto_tratado:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)
del tokens,sent_str

model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=1, workers=4, sg=0)
#model_ted.wv.most_similar("pensao")

# =============================================================================
# TIPO1
# =============================================================================
resultado = [0]*len(sentences_ted)
#pega o vetor de cada documento
for i in range(len(sentences_ted)):
    aux = [0]*len(sentences_ted[i])
    #para cada documento percorre as palavras pegando seus vetores de embedding
    for j in range(len(sentences_ted[i])):
        aux[j] = list(model_ted.wv.get_vector(sentences_ted[i][j]))
    media_vec = [0]* len(aux[0])
    # para cada doc vai ter vetores de vetores, faz a media desses vetores
    for k in range(len(aux)):
        media_vec = [ (a + b) for a, b in zip(media_vec, aux[k]) ]
    # divide pela quantidade de elementos
    media_vec = list(map(lambda x: x/len(aux), media_vec))
    resultado[i] = media_vec

resultado = pd.DataFrame(resultado)

X_train, X_test, y_train, y_test = train_test_split(resultado, label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding entre palavras do documento W2V")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(pd.concat([resultado,data],axis =1), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding entre palavras do documento W2V+OHE")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding entre palavras do documento W2V+TFIDF")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(data),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding entre palavras do documento W2V+OHE+TFIDF")
print("\n")

# =============================================================================
# TIPO2
# =============================================================================
resultado = [0]*len(sentences_ted)
for i in range(len(sentences_ted)):
    aux = [0]*len(sentences_ted[i])
    #para cada documento percorre as palavras pegando seus vetores de embedding
    for j in range(len(sentences_ted[i])):
        aux[j] = np.mean(list(model_ted.wv.get_vector(sentences_ted[i][j])))
    resultado[i] = aux
resultado = pd.DataFrame(resultado).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(resultado, label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding da palavra individual W2V")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(pd.concat([resultado,data],axis =1), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding da palavra individual W2V+OHE")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding da palavra individual W2V+TFIDF")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(data),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Word2vec com media do embbeding da palavra individual W2V+OHE+TFIDF")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(csr_matrix(tfidf), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"TFIDF")
