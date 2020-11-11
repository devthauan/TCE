import pandas as pd
# Separa as colunas que são formatadas em códigos (atomiza as colunas)

def oneHotEncoding(data):
    colunas = data.columns
    for i in range (len(colunas)):
        if(data[colunas[i]].dtypes == 'O'):
            #colocar nome na frente do codigo
            dummies = pd.get_dummies(data[colunas[i]],prefix=colunas[i], prefix_sep='-')
            #concatenando a tabela de dummy criada com o dataset
            data = pd.concat([data,dummies],axis='columns')
            #dropando a antiga coluna
            data = data.drop(colunas[i],axis='columns')
    return data