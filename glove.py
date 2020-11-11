from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados
from gensim.models import KeyedVectors
from scipy.sparse import csr_matrix
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

embeddings_dict = {}
with open("glove.6B/glove_s100.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

del word,values,vector
#glove_file = pd.read_table('glove.6B/glove_s100.txt', sep=" ",index_col = 0)
#glove_file.columns = list(np.arange(0,100,1))
#index = glove_file.index.value_counts().where(glove_file.index.value_counts().values == 2).dropna()
#dicionario = glove_file.to_dict('index')
#model = KeyedVectors.load_word2vec_format(glove_file)
# =============================================================================
# TIPO1
# =============================================================================
resultado = [0]*len(sentences_ted)
#pega o vetor de cada documento
for i in range(len(sentences_ted)):
    aux = [0]*len(sentences_ted[i])
    #para cada documento percorre as palavras pegando seus vetores de embedding
    for j in range(len(sentences_ted[i])):
        # tenta achar o embedding da palavra no dicionario caso nao encontre coloque 0
        try:
            aux[j] = list(embeddings_dict[sentences_ted[i][j]])
        except:
            aux[j] = [0]*100
    media_vec = [0]* len(aux[0])
    # para cada doc vai ter vetores de vetores, faz a media desses vetores
    for k in range(len(aux)):
        media_vec = [ (a + b) for a, b in zip(media_vec, aux[k]) ]
    # divide pela quantidade de elementos
    media_vec = list(map(lambda x: x/len(aux), media_vec))
    resultado[i] = media_vec

resultado = pd.DataFrame(resultado)

X_train, X_test, y_train, y_test = train_test_split(resultado, label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding entre palavras do documento Glove")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(pd.concat([resultado,data],axis =1), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding entre palavras do documento Glove+OHE")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding entre palavras do documento Glove+TFIDF")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(data),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding entre palavras do documento W2V+OHE+TFIDF")
print("\n")

# =============================================================================
# TIPO2
# =============================================================================
resultado = [0]*len(sentences_ted)
for i in range(len(sentences_ted)):
    aux = [0]*len(sentences_ted[i])
    #para cada documento percorre as palavras pegando seus vetores de embedding
    for j in range(len(sentences_ted[i])):
        try:
            aux[j] = np.mean(list(embeddings_dict[sentences_ted[i][j]]))
        except:
             aux[j] =0
    resultado[i] = aux
resultado = pd.DataFrame(resultado).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(resultado, label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding da palavra individual Glove")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(pd.concat([resultado,data],axis =1), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding da palavra individual Glove+OHE")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding da palavra individual Glove+TFIDF")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(data),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding da palavra individual Glove+OHE+TFIDF")
print("\n")
X_train, X_test, y_train, y_test = train_test_split(csr_matrix(tfidf), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"TFIDF")
