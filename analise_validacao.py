import pandas as pd
import numpy as np

dados_tce = pd.read_csv("/home/devthauan/Documentos/ProjetoTCE/arquivos/dadosTCE.csv",low_memory=False)
dados_recebidos2 = pd.read_excel("/home/devthauan/Documentos/selecao de documentos TCE/Juliete/resposta/VERSÃO 2 ANÁLISE DOS EMPENHOS.xlsx")

indexes_fora = []
# Colocando a pontuacao e verificando se o empenho esta dentro do conjunto principal de dados
for i in range(len(dados_recebidos2)):
    dados_recebidos2['empenho_sequencial_empenho'].iloc[i] = str(dados_recebidos2['empenho_sequencial_empenho'].iloc[i])[:4]+"."+str(dados_recebidos2['empenho_sequencial_empenho'].iloc[i])[4:8]+"."+str(dados_recebidos2['empenho_sequencial_empenho'].iloc[i])[8:11]+"."+str(dados_recebidos2['empenho_sequencial_empenho'].iloc[i])[11:]
    if(dados_recebidos2['empenho_sequencial_empenho'].iloc[i] not in dados_tce["Empenho (Sequencial Empenho)(EOF)"].values):
        indexes_fora.append(i)

dados_recebidos1 = pd.read_excel("/home/devthauan/Documentos/selecao de documentos TCE/Juliete/resposta/ANÁLISE CONSOLIDADA DOS EMPENHOS.xlsx")
# Considerando apenas a coluna correta
dados_recebidos1["Empenho (Sequencial Empenho)(EOF)"] = dados_recebidos1["Empenho (Sequencial Empenho)(EOF).1"]
# retirando os empenhos nao validados
dados_recebidos1 = dados_recebidos1.dropna()
dados_recebidos1.reset_index(drop = True, inplace = True)
# arrumando a diferenca de nome das colunas
dados_recebidos2.columns = dados_recebidos1.columns
indexes_fora = []
# verificando se os emepnhos estao dentro do conjunto principal de dados
for i in range(len(dados_recebidos1)):
    if(dados_recebidos1['Empenho (Sequencial Empenho)(EOF)'].iloc[i] not in dados_tce["Empenho (Sequencial Empenho)(EOF)"].values):
        indexes_fora.append(i)
# =============================================================================
# Juntando os dados
# =============================================================================
dados_analisados = pd.concat([dados_recebidos2,dados_recebidos1],axis = 0)
dados_analisados.reset_index(drop = True, inplace = True)
del dados_recebidos2,dados_recebidos1
indexes_fora = []
# verificando se os emepnhos estao dentro do conjunto principal de dados
for i in range(len(dados_analisados)):
    if(dados_analisados['Empenho (Sequencial Empenho)(EOF)'].iloc[i] not in dados_tce["Empenho (Sequencial Empenho)(EOF)"].values):
        indexes_fora.append(i)
# passando para maiuscula para evitar diferencas
dados_analisados['ANÁLISE'] = dados_analisados['ANÁLISE'].str.upper()
## existem empenhos repetidos verificando se estao com a mesma analise em cada copia
empenhos_repetidos = list(dados_analisados["Empenho (Sequencial Empenho)(EOF)"].value_counts()[dados_analisados["Empenho (Sequencial Empenho)(EOF)"].value_counts().values >1].index)
empenhos_inconsistentes = []
# salva os empenhos repetidos inconsistentes e exclui os apenas repetidos
for i in range(len(empenhos_repetidos)):
    if(len(dados_analisados["ANÁLISE"][dados_analisados["Empenho (Sequencial Empenho)(EOF)"] == empenhos_repetidos[i]].value_counts())!=1):
        empenhos_inconsistentes.append(empenhos_repetidos[i])

# removendo as repeticoes
index_empenhos_repetidos = []
for i in range(len(empenhos_repetidos)):
    index_empenhos_repetidos.append(list(dados_analisados["ANÁLISE"][dados_analisados["Empenho (Sequencial Empenho)(EOF)"] == empenhos_repetidos[i]].index[:len(dados_analisados["ANÁLISE"][dados_analisados["Empenho (Sequencial Empenho)(EOF)"] == empenhos_repetidos[i]].index)-1]))
# achatando o vetor
index_empenhos_repetidos = [item for sublist in index_empenhos_repetidos for item in sublist]   
dados_analisados.drop(index_empenhos_repetidos, inplace = True)

# retirando o campo de analise dos dados
analise = pd.DataFrame(dados_analisados["ANÁLISE"])
analise.reset_index(drop = True, inplace = True)
dados_analisados.drop("ANÁLISE", axis = 1, inplace = True)
## pegando o index no dataset principal dos dados avaliados
indexes = [0]*dados_analisados.shape[0]
for i in range(dados_analisados.shape[0]):
    indexes[i] = dados_tce['Empenho (Sequencial Empenho)(EOF)'][dados_tce['Empenho (Sequencial Empenho)(EOF)'] == dados_analisados['Empenho (Sequencial Empenho)(EOF)'].iloc[i]].index[0]
# pegando os dados e salvando
dados_analisados = dados_tce.iloc[indexes]
dados_analisados.reset_index(drop = True, inplace = True)
dados_analisados = pd.concat([dados_analisados,analise], axis = 1)
dados_analisados.drop("Empenho (Sequencial Empenho)(EOF).1",axis =1,inplace = True)
dados_analisados.to_excel('dados_analisados.xlsx', index=False)
