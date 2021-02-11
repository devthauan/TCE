from preparacaoDados import tratamentoDados
import pandas as pd
import mlpack

data, label = tratamentoDados("sem OHE")#  Carrega os dados
dados = pd.read_csv("dados_stacking.csv", encoding = "utf-8")# Carrega os dados do Stacking

# Calculando EMST
result = mlpack.emst(dados)
emst = pd.DataFrame(*result.values())
emst.to_csv('emst.csv', index=False, header=False)

# Retira o ponto do codigo do rotulo
label['natureza_despesa_cod'] = [label['natureza_despesa_cod'].iloc[i].replace(".", "") for i in range(data.shape[0])]
# Convertendo os dados para o formato correto
data = dados.astype("str")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data.iloc[i].iloc[j] = "FEAT"+str(j+1)+":"+str(data.iloc[i].iloc[j])
data.reset_index(drop=True, inplace=True)
dados_preparados = pd.concat([label,data],axis = 1)

# Salva os dados
dados_preparados.to_csv('dados_preparados.csv', index=False, header=False,sep=" ")