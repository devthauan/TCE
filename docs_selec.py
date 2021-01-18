import numpy as np
import pandas as pd
from preparacaoDados import tratamentoDados
from tratamentos import pickles

#data, label = tratamentoDados("sem OHE")
#
#validados = pd.read_excel("ANÁLISE CONSOLIDADA DOS EMPENHOS.xlsx")
#validados = validados.where(validados['ANÁLISE'] != np.nan).dropna()
#identificador = validados['Empenho (Sequencial Empenho)(EOF)']
#identificador = list(identificador.astype("str"))
#
#
#data['empenho_sequencial_empenho'] = [data['empenho_sequencial_empenho'].iloc[i].replace(".", "") for i in range(data.shape[0])]
#
#index = []
#for i in range(data.shape[0]):
#    if(data['empenho_sequencial_empenho'].iloc[i] in identificador):
#        index.append(i)
#index = pd.DataFrame(index)
#index.to_csv("index.csv",index = False)
index = pd.read_csv("index.csv")
index = list(index['0'])
dados_preparados = pd.read_csv("dados_preparados.csv")
dados_preparados.drop(index,inplace = True)
dados_preparados.to_csv("dados_preparados2.csv")

#data.drop(index,inplace = True)
#label.drop(index,inplace = True)
#data.reset_index(inplace = True, drop = True)
#label.reset_index(inplace = True, drop = True)
#
#data['Empenho (Sequencial Empenho)(EOF)'] = [data['Empenho (Sequencial Empenho)(EOF)'].iloc[i].replace(".", "") for i in range(data.shape[0])]
#dados_tce = pd.read_csv("dados_analise_TCE_2.csv", encoding = "utf-8")
#
##excluindo documentos ja selecionados
#index = []
#for i in range(len(dados_tce)):
#    try:
#        index.append(data.where(data["Empenho (Sequencial Empenho)(EOF)"] == str(dados_tce["Empenho (Sequencial Empenho)(EOF)"].iloc[i])).dropna().index[0])
#    except:
#        print("ja excluido")
#
#pickles.criaPickle(pd.DataFrame(index),"index")
#data.drop(index,inplace = True)
#label.drop(index,inplace = True)
#index = data["Valor Saldo do Empenho(EOF)"].where(data["Valor Saldo do Empenho(EOF)"] == 0).dropna().index
#len(index)
#
#
#
#dados_tce = pd.read_csv("dados_analise_TCE_2.csv", encoding = "utf-8")
#index = pickles.carregaPickle("index")
#data.drop(index,inplace = True)
#label.drop(index,inplace = True)
#
#
#validados = pd.read_excel("ANÁLISE CONSOLIDADA DOS EMPENHOS.xlsx")
#prontos = validados.where(validados['ANÁLISE'] != np.nan).dropna()
#classes = prontos['Natureza Despesa (Cod)(EOF)'].value_counts().index
#classes = list(classes)
