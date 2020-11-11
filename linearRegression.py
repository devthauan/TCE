import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from preparacaoDados import tratamentoDados
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dados do OneHotEncoding
data, label = tratamentoDados("OHE") 
data = csr_matrix(data)
#data = data.astype('float16')
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label)
del data
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predito = reg.predict(X_test)
y_predito = pd.DataFrame((y_predito)).astype("int")
reg.score(X_test,y_test)
micro = f1_score(y_test,y_predito,average='micro')
macro = f1_score(y_test,y_predito,average='macro')
print("O f1Score micro do LinearRegression é: ",micro)
print("O f1Score macro do LinearRegression é: ",macro)



