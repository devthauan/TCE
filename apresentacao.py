import numpy as np
import pandas as pd
from modelos import knn
from scipy import sparse
from modelos import rocchio
from modelos import rocchio2
from modelos import radiusknn
from modelos import randomforest
from scipy.sparse import csr_matrix
from modelos import supportVectorMachine
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split

data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
#dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
dados = pd.concat([data,tfidf],axis=1 )
dados = dados.astype(pd.SparseDtype("float16", 0))
del data,tfidf
#
X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)

from matplotlib import pyplot as plt
label = label.astype("str")
x = list(label.where(label[0].str.contains("92..")).dropna().value_counts(ascending = True).index)
x = [x[i][0] for i in range(len(x))]
y = label.where(label[0].str.contains("92..")).dropna().value_counts(ascending = True).values

plt.figure(figsize=(8,6))
plt.bar(x,y)
plt.xticks([])
plt.xlabel("Elementos 92 (81 subelementos diferentes)",fontsize = 14)
plt.ylabel("Quantidade",fontsize = 14)
plt.title("Balanceamento dos subelementos do elemento 92 (Despesas de Exercícios Anteriores)",fontsize = 14)

# =============================================================================
# # SEM RELEVANCIA
# =============================================================================
sem_relevancia = pd.read_excel("analise/Naturezas de despesa com vigência encerrada.xlsx")
sem_relevancia = sem_relevancia['Nat. Despesa']
sem_relevancia = [sem_relevancia.iloc[i].split('.')[3:][0]+sem_relevancia.iloc[i].split('.')[3:][1] for i in range(len(sem_relevancia))]
sem_relevancia = pd.DataFrame(sem_relevancia)

excluir = []
for i in range(len(sem_relevancia[0].value_counts())):
    excluir.append( label.where(label[0] == int(sem_relevancia[0].value_counts().index[i])).dropna().index )
    
excluir = [item for sublist in excluir for item in sublist]
label.drop(excluir,inplace =True)
label[0].value_counts().count()

# =============================================================================
# #Verificando se tem empenhos com subelemento 00 R: nao
# =============================================================================
#label_elemento = label.copy()
#label_subelemento = label.copy()
## Separando os codigos do elemento e subelemento
#for i in range(label_elemento.shape[0]):  
#    if(len(str(label_elemento[0].iloc[i]))==3):
#        label_elemento[0].iloc[i] = str(label[0].iloc[i])[:1]
#        label_subelemento[0].iloc[i] = str(label[0].iloc[i])[1:]
#    else:
#        label_elemento[0].iloc[i] = str(label[0].iloc[i])[:2]
#        label_subelemento[0].iloc[i] = str(label[0].iloc[i])[2:]


mais_importantes = pd.read_excel("analise/Naturezas de Despesa vigentes em 2020.xls")
utilizadas = pd.read_excel("analise/Naturezas utilizadas 2015 a 2020.xlsx")
from sklearn.metrics import accuracy_score

y_test =    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3]
y_predito = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,2,2,2,1,3,3,3,2,2,2]

print( f1_score(y_test,y_predito,average='micro'))
print( f1_score(y_test,y_predito,average='macro'))


y_test =    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
y_predito = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3]

print( f1_score(y_test,y_predito,average='micro'))
#print( f1_score(y_test,y_predito,average='macro'))

y_test =    [2,2,2,2]
y_predito = [2,2,2,1]

print( f1_score(y_test,y_predito,average='micro'))
#print( f1_score(y_test,y_predito,average='macro'))

y_test =    [3,3,3,3,3,3]
y_predito = [3,3,3,2,2,2]

print( f1_score(y_test,y_predito,average='micro'))
#print( f1_score(y_test,y_predito,average='macro'))


plt.figure(figsize=(15,10))
plt.title("Natureza de despesa",fontsize= 20)#y=1.08
plt.rcParams.update({'font.size': 12})
plt.bar(label['natureza_despesa_cod'].value_counts(ascending=True).index.astype("str"),label['natureza_despesa_cod'].value_counts(ascending=True).values )
#plt.xticks(rotation=90)
#plt.text(0, 0 + .25, str(1), color='black')
#plt.text(48.5, 65000+ .25, str(64456), color='black')
plt.xticks([0,50])
plt.ylabel('Quantidade',fontsize=15)
plt.xlabel('Elementos',fontsize=15)
plt.show()

