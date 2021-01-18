import numpy as np
import pandas as pd
from tratamentos import tratar_texto
from tratamentos import tratar_label

# Carrega os dados na variavel 'data' utilizando o 
nomeArquivoDadosBrutos = 'arquivos/dadosTCE.csv';
data = pd.read_csv(nomeArquivoDadosBrutos, encoding = "utf-8",low_memory = False)
del nomeArquivoDadosBrutos
# Trata o nome das colunas para trabalhar melhor com os dados
data.columns = [c.lower().replace(' ', '_') for c in data.columns]
data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
# Deletando empenhos sem relevancia devido ao saldo zerado
index = data["valor_saldo_do_empenho"].where(data["valor_saldo_do_empenho"] == 0).dropna().index
data.drop(index,inplace = True)
data.reset_index(drop=True, inplace=True)
## Deleta colunas que atraves de analise foram identificadas como nao uteis
#data = data.drop(['empenho_sequencial_empenho.1','classificacao_orcamentaria_descricao',
#                  'natureza_despesa_nome','valor_estorno_anulacao_empenho',
#                  'valor_anulacao_cancelamento_empenho','fonte_recurso_cod',
#                  'elemento_despesa','grupo_despesa','empenho_sequencial_empenho'], axis='columns')
# Funcao que gera o rotulo e retorna as linhas com as naturezas de despesa que so aparecem em 1 empenho
label,linhas_label_unica = tratar_label.tratarLabel(data)
label = pd.DataFrame(label)
# Excluindo as naturezas de despesas que so tem 1 empenho
data = data.drop(linhas_label_unica)
data.reset_index(drop=True, inplace=True)
del linhas_label_unica
# Excluindo empenhos irrelevantes devido nao estarem mais em vigencia
sem_relevancia = pd.read_excel("analise/Naturezas de despesa com vigência encerrada.xlsx")
sem_relevancia = sem_relevancia['Nat. Despesa']
sem_relevancia = pd.DataFrame(sem_relevancia)
excluir = []
for i in range(len(sem_relevancia['Nat. Despesa'])):
    excluir.append( label.where( label['natureza_despesa_cod'] == sem_relevancia['Nat. Despesa'].iloc[i] ).dropna().index )
excluir = [item for sublist in excluir for item in sublist]
# Excluindo as naturezas que nao estao mais vigentes
label.drop(excluir,inplace =True)
label.reset_index(drop=True, inplace=True)
data.drop(excluir,inplace = True)
data.reset_index(drop=True, inplace=True)
del excluir, sem_relevancia

#docs = pd.DataFrame(np.loadtxt('resultados/finais/resultado perl/clusthist-3000.txt'))
#rows = docs[0]
#data_analisar = data[data.columns].iloc[rows]
#del docs,rows

## =============================================================================
## comparar com outliers
## =============================================================================
## outliers para serem removidos
#outliers = pd.read_csv("outliers_DBSCAN.xlsx")
#
##verifica interseccao
#intersect = set(outliers["Index dos Outliers"].values).intersection(data_analisar.index)
#len(intersect) #33 docs sao outliers e representativos
#
#dados_outliers = data.iloc[outliers["Index dos Outliers"].values]
#intersect = pd.DataFrame(intersect).astype(int)
#dados_outliers = dados_outliers.drop(intersect[0].values)
#dados_completos =  pd.concat([data_analisar,dados_outliers],axis =0)
#del intersect, outliers, dados_outliers,data_analisar
#
#
## verificando se nao tem documentos repetidos
#dados_completos['Empenho (Sequencial Empenho)(EOF)'].value_counts().count() == dados_completos.shape[0]
#
#
#
#
#label_analisar = label[0].iloc[dados_completos.index]
##label_analisar.value_counts().count() #219 labels de fora
#label_analisar.reset_index(drop=True, inplace=True)
#dados_completos.reset_index(drop=True, inplace=True)
##verificar classes com menos de 21 documentos
#deletar = []
## verifica em todas as classes se a quantidade excede 20
#for i in range(label_analisar.shape[0]):
#    if(label[0].where(label[0] == label_analisar[i]).dropna().value_counts().values[0] <=20):
#        deletar.append(i)
#
#dados_completos = dados_completos.drop(deletar)
#dados_completos.reset_index(drop=True, inplace=True)
#del deletar, label_analisar
## =============================================================================
## naturezas
## =============================================================================
##label_outlier = label[0].iloc[outliers["Index dos Outliers"].values]
##label_outlier.value_counts().count() #198 labels de fora
#
##label_completa = pd.concat([label_outlier,label_analisar],axis = 0)
##label_analisar.value_counts().count() #112 labels de fora
#
##label_completa
##diferenca = set(label_completa).symmetric_difference(set(label[0].values))
##len(diferenca)
#
## retirando pintuacao
#dados_completos['Empenho (Sequencial Empenho)(EOF)'] = [dados_completos['Empenho (Sequencial Empenho)(EOF)'].iloc[i].replace(".", "") for i in range(dados_completos.shape[0])]
#
#
## empenhos ja validados
#empenhos_classificados = pd.read_csv("empenhosClassificados.csv",sep = ";")
#
##categorioas = []
##for i in range(len(num)):
##     categorioas.append(data["Natureza Despesa (Nome)(EOF)"].iloc[data['Empenho (Sequencial Empenho)(EOF)'].where(data['Empenho (Sequencial Empenho)(EOF)']==str(num[i])).dropna().index[0]])
## pd.DataFrame(categorioas).value_counts()   
#
#docs_mauricio = (empenhos_classificados["key_nr_empenho"].where(empenhos_classificados["e_natureza_publicidade"]==1).dropna()).astype("int64").astype("str")
#docs_mauricio.reset_index(drop=True, inplace=True)
#del empenhos_classificados
#
## pega os docs ja validados
#repeticoes = set(dados_completos['Empenho (Sequencial Empenho)(EOF)']).intersection(docs_mauricio)
#len(repeticoes)
#repeticoes = list(repeticoes)
## pega o index dos docs ja validados
#index_validados = []
#for i in range(len(repeticoes)):
#    index_validados.append(int(dados_completos.index.where(dados_completos["Empenho (Sequencial Empenho)(EOF)"]==(repeticoes[i])).dropna().values[0]))
#
#dados_completos = dados_completos.drop(index_validados)
#dados_completos.reset_index(drop=True, inplace=True)
#
#dados_completos = dados_completos["Empenho (Sequencial Empenho)(EOF)"]
#dados_completos.to_csv("dados_analise_TCE_2.csv",index = False)
##
#data['Empenho (Sequencial Empenho)(EOF)'] = [data['Empenho (Sequencial Empenho)(EOF)'].iloc[i].replace(".", "") for i in range(data.shape[0])]
#docs_mauricio_tce = set(docs_mauricio).intersection(data["Empenho (Sequencial Empenho)(EOF)"])
#len(docs_mauricio_tce)


docs = pd.DataFrame(np.loadtxt('resultados/finais/resultado perl/clusthist-4000.txt'))
rows = docs[0][:3500]
data_analisar = data[data.columns].iloc[rows]
data_analisar.reset_index(drop=True, inplace=True)
outliers = pd.read_csv("outliers_DBSCAN.xlsx")
dados_outliers = data.iloc[outliers["Index dos Outliers"].values]
dados_outliers = dados_outliers.iloc[:200]
dados_outliers.reset_index(drop=True, inplace=True)

repeticoes = set(data_analisar['empenho_sequencial_empenho']).intersection(dados_outliers['empenho_sequencial_empenho'])
len(repeticoes)
index = []
for i in range(len(data_analisar)):
    if(data_analisar['empenho_sequencial_empenho'].iloc[i] in repeticoes):
        index.append(i)

data_analisar.drop(index,inplace=True)
data_analisar = pd.concat([data_analisar,dados_outliers],axis =0)
data_analisar.reset_index(drop=True, inplace=True)

del docs,rows
dados = pd.read_excel("ANÁLISE CONSOLIDADA DOS EMPENHOS.xlsx")
dados = dados.where(dados['ANÁLISE'] != np.nan).dropna()


data_analisar['empenho_sequencial_empenho'] = [data_analisar['empenho_sequencial_empenho'].iloc[i].replace(".", "") for i in range(data_analisar.shape[0])]
repetidos = set(data_analisar['empenho_sequencial_empenho']).intersection(dados["Empenho (Sequencial Empenho)(EOF)"].astype("str"))
print("tamanho dos repetidos",len(repetidos))

index = []
for i in range(len(data_analisar)):
    if(data_analisar['empenho_sequencial_empenho'].iloc[i] in repetidos):
        index.append(i)

data_analisar.drop(index,inplace=True)
data_analisar.reset_index(drop=True, inplace=True)


#ind_dict = dict((k,i) for i,k in enumerate(data_analisar["empenho_sequencial_empenho"]))
#repetidos = set(ind_dict).intersection(dados["empenho_sequencial_empenho"].astype("str"))
#indices = [ ind_dict[x] for x in repetidos ]
#len(indices)

index = data_analisar["valor_saldo_do_empenho"].where(data_analisar["valor_saldo_do_empenho"] == 0).dropna().index
print("index dos com saldo zerado",len(index))
#data_analisar.drop(index,inplace = True)

# Removendo naturezas 3.1
import re
pattern = re.compile("3\.1\...\...\...")
data_analisar = pd.read_csv('data_analisar.csv')
classificados = []
for i in range(len(data_analisar)):
    if(pattern.match(data_analisar['natureza_despesa_cod'].iloc[i])):
        classificados.append(i)

data_analisar.drop(data_analisar,inplace = True)
data_analisar.reset_index(inplace=True, drop=True)


print("tamanho final",data_analisar.shape)
data_analisar.to_csv("data_analisar.csv",index = False)

