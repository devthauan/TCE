import numpy as np
import pandas as pd
from preparacaoDados import tratamentoDados
from tratamentos import pickles
data['Empenho (Sequencial Empenho)(EOF)'] = [data['Empenho (Sequencial Empenho)(EOF)'].iloc[i].replace(".", "") for i in range(data.shape[0])]
dados_tce = pd.read_csv("dados_analise_TCE_2.csv", encoding = "utf-8")

#excluindo documentos ja selecionados
index = []
for i in range(len(dados_tce)):
    try:
        index.append(data.where(data["Empenho (Sequencial Empenho)(EOF)"] == str(dados_tce["Empenho (Sequencial Empenho)(EOF)"].iloc[i])).dropna().index[0])
    except:
        print("ja excluido")

pickles.criaPickle(pd.DataFrame(index),"index")
data.drop(index,inplace = True)
label.drop(index,inplace = True)
index = data["Valor Saldo do Empenho(EOF)"].where(data["Valor Saldo do Empenho(EOF)"] == 0).dropna().index
len(index)



dados_tce = pd.read_csv("dados_analise_TCE_2.csv", encoding = "utf-8")
index = pickles.carregaPickle("index")
data.drop(index,inplace = True)
label.drop(index,inplace = True)