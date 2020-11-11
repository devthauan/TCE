import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#feature_importances_quanto maior o valor mais importante
def featureImportance(data, label, num_features_mais_importantes,porcentagem):
    # aplica o randomforest para acessar o feature importance depois
    rnd_clf = RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1)
    rnd_clf.fit(data, label.values.ravel())
    #feature_importances_quanto maior o valor mais importante
    importancia = list(rnd_clf.feature_importances_ )
    if(porcentagem == -1):
        excluir = []
        while(len(excluir) < len(importancia)-65000):
            #pega o index do menor valor
            indice = importancia.index( min(importancia))        
            importancia[indice]=2
            excluir.append(indice)
    else:
        excluir = [0]*int(len(importancia)*porcentagem)
        # Retorna a posição das X % das colunas menos importantes
        for i in range(int(len(importancia)*porcentagem)):
            #pega o index do menor valor
            indice = importancia.index( min(importancia))        
            importancia[indice]=2
            excluir[i] = indice
    # Pega as colunas
    colunas = data.columns
    # Pega o nome das colunas a excluir
    para_excluir = colunas[excluir]  
    # Exclui a coluna
    data = data.drop(para_excluir,axis="columns")   
    #pegando as N features mais importantes
    features_mais_importantes = []
    valor = [0,0]
    for j in range(num_features_mais_importantes):
      for i in range(len(importancia)):
        if(importancia[i]>valor[1]):
          valor[0]=i
          valor[1]=importancia[i]
      importancia[valor[0]]=-1   
      features_mais_importantes.append(valor[0])
      valor = [0,0]
    colunas_importantes = colunas[features_mais_importantes]
    return data, colunas_importantes