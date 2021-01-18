import numpy as np
import pandas as pd
from modelos import knn
from scipy import sparse
from modelos import rocchio
from modelos import radiusknn
from modelos import randomforest
from scipy.sparse import csr_matrix
from modelos import supportVectorMachine
from modelos import feature_importance
from preparacaoDados import tratamentoDados
from modelos import xboosting
from modelos import sgd
from modelos import naive_bayes
from sklearn import preprocessing
#from preparacaoDados2 import tratamentoDados
from sklearn.model_selection import train_test_split
# =============================================================================
# PRIMEIRA PARTE
# =============================================================================
data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
dados =  pd.DataFrame.sparse.from_spmatrix(aux)
print("Size ",dados.shape[0])
print("Feat. ",dados.shape[1])
print("Classes. ",label['natureza_despesa_cod'].value_counts().count())
print("Minor Classes. ",label['natureza_despesa_cod'].value_counts()[-1])
print("Median ",np.median(label['natureza_despesa_cod'].value_counts().values))
print("Mean ",np.mean(label['natureza_despesa_cod'].value_counts().values))
print("Major Classes. ",label['natureza_despesa_cod'].value_counts()[0])
# =============================================================================
# SEGUNDA PARTE
# =============================================================================
from tratamentos import tratar_label
from tratamentos import tratar_texto
nomeArquivoDadosBrutos = 'arquivos/dadosTCE.csv';
# Carrega os dados na variavel 'data' utilizando o Pandas
data = pd.read_csv(nomeArquivoDadosBrutos, encoding = "utf-8",low_memory = False)
del nomeArquivoDadosBrutos
# Trata o nome das colunas para trabalhar melhor com os dados
data.columns = [c.lower().replace(' ', '_') for c in data.columns]
data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
# Excluindo empenhos diferentes aglomerados na classe 92
exercicio_anterior = data['natureza_despesa_cod'].str.contains(".\..\...\.92\...", regex= True, na=False)
index = exercicio_anterior.where(exercicio_anterior==True).dropna().index
data.drop(index,inplace = True)
data.reset_index(drop=True, inplace=True)
# Deletando empenhos sem relevancia devido ao saldo zerado
index = data["valor_saldo_do_empenho"].where(data["valor_saldo_do_empenho"] == 0).dropna().index
data.drop(index,inplace = True)
data.reset_index(drop=True, inplace=True)
#data = data[:500] #limitando os dados para fazer testes
# Deleta colunas que atraves de analise foram identificadas como nao uteis
data = data.drop(['empenho_sequencial_empenho.1','classificacao_orcamentaria_descricao',
                  'natureza_despesa_nome','valor_estorno_anulacao_empenho',
                  'valor_anulacao_cancelamento_empenho','fonte_recurso_cod',
                  'elemento_despesa','grupo_despesa','empenho_sequencial_empenho'], axis='columns')
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

tamanho_doc = [0]*data.shape[0]
for i in range(data.shape[0]):
    tamanho_doc[i] = len(data["empenho_historico"].iloc[i].split(" "))
print("Avg Doc. Size (words) . ",np.mean(tamanho_doc)) 

datasets = pd.DataFrame([["WEBKB",8199,23047,7,137,926,1171,3705,209]])
datasets = pd.concat([datasets,pd.DataFrame([["REUT",13327,27302,90,2,29,148,3964,171]])])
datasets = pd.concat([datasets,pd.DataFrame([["20NG",18846,97401,20,628,984,942,999,296]])])
datasets = pd.concat([datasets,pd.DataFrame([["ACM",24897,48867,11,63,2041,2263,6562,65]])])
datasets = pd.concat([datasets,pd.DataFrame([["AGNEWS",127600,39837,4,31900,31900,31900,31900,37]])])
datasets = pd.concat([datasets,pd.DataFrame([["IMDB review",348415,115831,10,12836,31551,64841,63266,326]])])
datasets = pd.concat([datasets,pd.DataFrame([["SOGOU",510000,98974,5,102000,102000,102000,102000,588]])])
datasets = pd.concat([datasets,pd.DataFrame([["YELP Full",700000,115371,5,140000,140000,140000,140000,136]])])
datasets = pd.concat([datasets,pd.DataFrame([["Yahoo Answer",1460000,1554607,10,146000,146000,146000,146000,92]])])
datasets.columns = ["Dataset","Size","Features","Classes","Minor Class", "Median","Mean", "Major Class", "Avg Doc Size (words)"]

meu_dataset = pd.DataFrame([["Siofinet (TCE)",247896,72617,495,2,120,500,5978,69]])
meu_dataset.columns = ["Dataset","Size","Features","Classes","Minor Class", "Median","Mean", "Major Class", "Avg Doc Size (words)"]
from sklearn.metrics.pairwise import euclidean_distances

datasets_test = datasets.drop("Dataset",axis = 1)
meu_dataset_test = meu_dataset.drop("Dataset",axis = 1)
distancias = euclidean_distances(meu_dataset_test, datasets_test)
datasets["Dataset"].iloc[list(distancias[0]).index(min(distancias[0]))]

