from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split
import mlpack
import pandas as pd

data, label = tratamentoDados("sem OHE")
dados = pd.read_csv("dados_stacking.csv", encoding = "utf-8")

#Calculando EMST
result = mlpack.emst(dados)
emst = pd.DataFrame(*result.values())
emst.to_csv('emst.csv', index=False, header=False)


# Convertendo os dados para o formato correto
data = dados.astype("str")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data.iloc[i].iloc[j] = "FEAT"+str(j+1)+":"+str(data.iloc[i].iloc[j])
data.reset_index(drop=True, inplace=True)
dados_preparados = pd.concat([label,data],axis = 1)

#dados_preparados = dados_preparados.astype("str")
dados_preparados.to_csv('dados_preparados.csv', index=False, header=False,sep=" ")
