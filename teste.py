from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados
from modelos import supportVectorMachine
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

X_train, X_test, y_train, y_test = train_test_split(sparse.hstack((csr_matrix(resultado),csr_matrix(tfidf))), label,test_size=0.3,stratify = label,random_state =5)
randomforest.randomForest(X_train, X_test, y_train, y_test,"Glove com media do embbeding da palavra individual Glove+TFIDF")
supportVectorMachine.svc(X_train, X_test, y_train, y_test,"COM DADOS COMBINADOS",10)
