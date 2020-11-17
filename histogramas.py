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
#from preparacaoDados2 import tratamentoDados
from sklearn.model_selection import train_test_split

data, label = tratamentoDados("sem OHE")


# Codigo para plotar o histograma dos dados categoricos
from matplotlib import pyplot as plt
for col in data.columns:
    if(data[col].dtypes == 'O'):
        plt.figure(figsize=(16,11))
        plt.title(col,fontsize= 20)#y=1.08
        plt.rcParams.update({'font.size': 12})
        plt.bar(data[col].value_counts(ascending=True).index.astype("str"),( (data[col].value_counts(ascending=True).values) /label.shape[0])*100)
        plt.yticks( list(range(0,int(np.ceil(np.ceil((data[col].value_counts(ascending=True).values[-1]/label.shape[0])*100)))+int(np.ceil((data[col].value_counts(ascending=True).values[-1]/label.shape[0])*10)),int(np.ceil((data[col].value_counts(ascending=True).values[-1]/label.shape[0])*10)))) )
        if(data[col].value_counts().count()>40):
            for i in range(5):
                print(data[col].value_counts(ascending=True).index[data[col].value_counts().count()-5+i])
            plt.xticks([data[col].value_counts().count()-1])
            plt.xticks(rotation=90)
        else:plt.xticks(rotation=90)
        plt.ylabel('Porcentagem',fontsize=15)
        plt.xlabel(col,fontsize=15)
        plt.show()

# Código que plota o histograma da label
from matplotlib import pyplot as plt

plt.figure(figsize=(15,10))
plt.title("Histograma Natureza de despesa codigo (Rótulo)",fontsize= 20)#y=1.08
plt.rcParams.update({'font.size': 12})
plt.bar(label['natureza_despesa_cod'].value_counts(ascending=True).index.astype("str"),( (label['natureza_despesa_cod'].value_counts(ascending=True).values) /label.shape[0])*100)
#plt.xticks(rotation=90)
plt.xticks([])
plt.ylabel('Porcentagem',fontsize=15)
plt.xlabel('Código natureza despesa',fontsize=15)
plt.show()