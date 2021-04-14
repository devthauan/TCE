import pandas as pd

dados_juliete = pd.read_excel("/home/devthauan/Documentos/selecao de documentos TCE/Juliete/ANÁLISE CONSOLIDADA DOS EMPENHOS.xlsx")
dados_juliete2 = pd.read_excel("/home/devthauan/Documentos/selecao de documentos TCE/Juliete/VERSÃO 2 ANÁLISE DOS EMPENHOS.xlsx")
dados_juliete.dropna(inplace = True)
dados_juliete2.columns = dados_juliete.columns
dados_analisados = pd.concat([dados_juliete2,dados_juliete],axis = 0)
del dados_juliete, dados_juliete2

dados_analisados.to_csv('/home/devthauan/Documentos/selecao de documentos TCE/Juliete/dados_analisados.xlsx', index=False)
