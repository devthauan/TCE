import pandas as pd
from scipy import sparse
from tratamentos import pickles
from scipy.sparse import csr_matrix
from modelos import feature_importance
from preparacaoDados import tratamentoDados


data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf")
aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
data =  pd.DataFrame.sparse.from_spmatrix(aux)
data_feature_importance, colunasMaisImportantes = feature_importance.featureImportance(data,label,1,0.92)
print("tamanho reduzido "+str(data_feature_importance.shape))
data = csr_matrix(data_feature_importance)

pickles.criaPickle(data_feature_importance,"data_reduzida")
pickles.criaPickle(label,"label_reduzida")