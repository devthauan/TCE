import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# Separa as colunas que são formatadas em códigos (atomiza as colunas)

def oneHotEncoding(data):
    colunas = data.columns
    for i in range (len(colunas)):
        if(data[colunas[i]].dtypes == 'O'):
            enc = OneHotEncoder(handle_unknown = 'ignore')
            enc.fit(data[colunas[i]].values.reshape(-1, 1))
            with open('pickles/modelos_tratamentos/'+"OHE_"+colunas[i]+'.pk', 'wb') as fin:
                pickle.dump(enc, fin)
            dummies = pd.DataFrame(enc.transform(data[colunas[i]].values.reshape(-1, 1)).toarray())
            #concatenando a tabela de dummy criada com o dataset
            data = pd.concat([data,dummies],axis='columns')
            #dropando a antiga coluna
            data = data.drop(colunas[i],axis='columns')
    return data

def aplyOHE(data):
    colunas = data.columns
    for i in range (len(colunas)):
        if(data[colunas[i]].dtypes == 'O'):
            with open('pickles/modelos_tratamentos/'+"OHE_"+colunas[i]+'.pk', 'rb') as pickle_file:
                enc = pickle.load(pickle_file)
            dummies = pd.DataFrame(enc.transform(data[colunas[i]].values.reshape(-1, 1)).toarray())
            #concatenando a tabela de dummy criada com o dataset
            data = pd.concat([data,dummies],axis='columns')
            #dropando a antiga coluna
            data = data.drop(colunas[i],axis='columns')
    return data
    