import numpy as np
import pandas as pd
from scipy import sparse
from tratamentos import pickles
from scipy.sparse import csr_matrix
from preparacaoDados import tratamentoDados
from tratamentos import tratar_texto
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.feature_extraction.text import CountVectorizer

# Separa os valores em 100 intervalos igualmente espacados
def create_bins(lower_bound, higher_bound):
    bins = []
    width = higher_bound/100
    for low in np.arange(lower_bound, higher_bound, width):
        bins.append((low, low+width))
    return bins

# Retorna em qual intervalo o valor esta
def find_bin(value, bins):  
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1


texto = tratamentoDados("texto")
dados, label = tratamentoDados("sem OHE")
# Trata os valores numericos colocando-os em intervalos e os valores categoricos limpando o texto
for col in dados.columns:
    # Caso seja numerico entre
    if(dados[col].dtype == "float" or dados[col].dtype != "O"):
        # Se tiver apenas 1 valor
        if(dados[col].value_counts().count()==1):
            dados[col]="coluna_"+col+"_intervalo_0 "
            continue
        # Separa o range dos valores em 100 intervalos iguais
        bins = create_bins(lower_bound=dados[col].min(),higher_bound=dados[col].max()+0.000001)
        column = [0]*dados.shape[0]
        # Verifica em qual intervalo o doc esta e coloca uma classe para ele
        for i in range(dados.shape[0]):
            column[i] = "coluna_"+col+"_intervalo_"+str(find_bin(dados[col].iloc[i],bins))
        dados[col]=column
    else:
        # Limpa o texto das colunas categoricas
        dados[col] = tratar_texto.cleanTextData(dados[col]) 

# Concatena os textos de todas as colunas categoricas em uma string por documento
texto_colunas = [0]*dados.shape[0]
for i in range(dados.shape[0]):
    texto_colunas[i] = str(dados[dados.columns].iloc[i].values)

# Separa as palavras de todas as colunas categoricas
vectorizer_col = CountVectorizer(binary=True)
df_col = vectorizer_col.fit_transform(pd.DataFrame(texto_colunas)[0])
dataset_col = pd.DataFrame.sparse.from_spmatrix(df_col, columns = vectorizer_col.get_feature_names())

# Separa as palavras da colunas de texto
vectorizer = CountVectorizer(binary=True)
df = vectorizer.fit_transform(texto)
dataset = pd.DataFrame.sparse.from_spmatrix(df, columns = vectorizer.get_feature_names())

# Concatena os dados de texto das colunas categoricas com a coluna de texto
#dados_completos =  pd.concat([dataset,dataset_col],axis=1).astype('int8')

# vetorizando a label
vectorizer3 = CountVectorizer(binary=True)
df = vectorizer3.fit_transform(label[0].astype('str'))
label_vect = pd.DataFrame.sparse.from_spmatrix(df, columns = vectorizer3.get_feature_names())

# Usa apriori para pegar os itemsets
#frequent_itemsets = apriori(dataset_col.astype('int8').sparse.to_dense(), min_support=0.6, use_colnames=False)
frequent_itemsets = apriori(pd.concat([label_vect,dataset],axis=1), min_support=0.3, use_colnames=True)
#frequent_itemsets = apriori(pd.concat([label_vect,dataset_col],axis=1), min_support=0.3, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Faz a regra de associacao devolvendo support e confidence
association_rule = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

association_rule.to_excel("association_rule.xlsx")
