import pandas as pd
from tratamentos import pickles
from modelos import randomforest
from sklearn.metrics import f1_score
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split

# Carrega os dados e separa eles
data, label = tratamentoDados("OHE")
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label)


def evaluate(y_test,y_predito):
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    print("O f1Score micro é: ",micro)
    print("O f1Score macro é: ",macro)


# Resultados da predicao flat elemento e subelemento
y_predito_rf = randomforest.randomForest(X_train, X_test, y_train, y_test,"COM OHE")



# Pega o elemento do y_flat
y_predito_rf_elemento = y_predito_rf.copy()
for i in range(y_predito_rf_elemento.shape[0]):
    if(len(str(y_predito_rf_elemento[i]))==3):
        y_predito_rf_elemento[i] = str(y_predito_rf_elemento[i])[:1]
    else:
        y_predito_rf_elemento[i] = str(y_predito_rf_elemento[i])[:2]

# Pega o elemento do y_test
y_test_elemento = y_test.copy()
for i in range(y_test_elemento.shape[0]):
    if(len(str(y_test_elemento[0].iloc[i]))==3):
        y_test_elemento[0].iloc[i]=str(y_test_elemento[0].iloc[i])[:1]
    else:
       y_test_elemento[0].iloc[i]=str(y_test_elemento[0].iloc[i])[:2]
    
y_predito_rf_elemento = pd.DataFrame(y_predito_rf_elemento).astype('str')
y_test_elemento = pd.DataFrame(y_test_elemento).astype('str')

# Comparando a predicao flat com o real
evaluate(y_test_elemento ,y_predito_rf_elemento)


# =============================================================================
# stacking 
# =============================================================================
# Pega o elemento do y_test
#y_test_elemento = y_test.copy()
#for i in range(y_test_elemento.shape[0]):
#    if(len(str(y_test_elemento[0].iloc[i]))==3):
#        y_test_elemento[0].iloc[i]=str(y_test_elemento[0].iloc[i])[:1]
#    else:
#        y_test_elemento[0].iloc[i]=str(y_test_elemento[0].iloc[i])[:2]
#    
## Pega o elemento do stacking
#y_stacking_elemento = y_stacking.copy()
#for i in range(y_stacking_elemento.shape[0]):
#    if(len(str(y_stacking_elemento[i]))==3):
#        y_stacking_elemento[i] = str(y_stacking_elemento[i])[:1]
#    else:
#        y_stacking_elemento[i] = str(y_stacking_elemento[i])[:2]
# 
#y_stacking_elemento = pd.DataFrame(y_stacking_elemento).astype('str')
#y_test_elemento = pd.DataFrame(y_test_elemento).astype('str')
#
## Comparando a predicao flat com o real
#evaluate(y_test_elemento ,y_stacking_elemento)
#
## =============================================================================
## Visao Dupla
## =============================================================================
## Pega o elemento do y_test
#y_test_elemento = y_test.copy()
#for i in range(y_test_elemento.shape[0]):
#    if(len(str(y_test_elemento[0].iloc[i]))==3):
#        y_test_elemento[0].iloc[i]=str(y_test_elemento[0].iloc[i])[:1]
#    else:
#        y_test_elemento[0].iloc[i]=str(y_test_elemento[0].iloc[i])[:2]
#    
## Pega o elemento do visao dupla
#y_visaodupla_elemento = y_visaodupla.copy()
#for i in range(y_visaodupla_elemento.shape[0]):
#    if(len(str(y_visaodupla_elemento[i]))==3):
#        y_visaodupla_elemento[i] = str(y_visaodupla[i])[:1]
#    else:
#        y_visaodupla_elemento[i] = str(y_visaodupla_elemento[i])[:2]
# 
#y_visaodupla_elemento = pd.DataFrame(y_visaodupla_elemento).astype('str')
#y_test_elemento = pd.DataFrame(y_test_elemento).astype('str')
#
## Comparando a predicao flat com o real
#evaluate(y_test_elemento ,y_visaodupla_elemento)
#
