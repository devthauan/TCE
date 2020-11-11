import numpy as np
import pandas as pd
from modelos import rocchio
from sklearn.svm import SVC
from tratamentos import pickles
from modelos import randomforest
from scipy.sparse import csr_matrix
from tratamentos import salvar_dados
from sklearn.metrics import f1_score
from modelos import supportVectorMachine
from preparacaoDados import tratamentoDados
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def meuModelo_fit_elemento(X_train,y_train):
    # copiando a label para pegar apenas o elemento
    label_elemento = y_train.copy()
    # Pegando apenas o codigo do elemento
    for i in range(label_elemento.shape[0]): 
        if(len(str(label_elemento[0].iloc[i]))==3):
            label_elemento[0].iloc[i] = (str(y_train[0].iloc[i])[:1])
        else:
            label_elemento[0].iloc[i] = (str(y_train[0].iloc[i])[:2])
    if(modelo1 == "rf"):
        rnd_clf = RandomForestClassifier(n_jobs=-1,n_estimators=200,random_state=0,max_samples=int((len(X_train)+len(X_test))*0.3))
        if(TIPO_DADO =="combinado"):
            rnd_clf.fit(csr_matrix(X_train.astype('float16').fillna(0)), label_elemento.values.ravel())
        else:
            rnd_clf.fit(X_train, label_elemento.values.ravel())
        pickles.criarModelo(rnd_clf,"meusmodelos/randomForest_modelo_elemento")
    elif(modelo1 == "rocchio"):
        rnd_clf = NearestCentroid()
        if(TIPO_DADO == "combinado"):
            rnd_clf.fit(csr_matrix(X_train.astype('float16').fillna(0)), label_elemento.values.ravel())
        else:
            rnd_clf.fit(X_train, label_elemento.values.ravel())
        pickles.criarModelo(rnd_clf,"meusmodelos/rocchio_modelo_elemento")
    elif(modelo1 == "knn"):
        if(TIPO_DADO == "tfidf"):
            rnd_clf = KNeighborsClassifier(n_jobs = -1,n_neighbors = 3)
            rnd_clf.fit(X_train, label_elemento.values.ravel())
        elif(TIPO_DADO == "OHE"):
            rnd_clf = KNeighborsClassifier(n_jobs = -1,n_neighbors = 1)#antigo valor 1
            rnd_clf.fit(X_train, label_elemento.values.ravel())
        else:
            rnd_clf = KNeighborsClassifier(n_jobs = -1,n_neighbors = 1)#antigo valor 1
            rnd_clf.fit(csr_matrix(X_train.astype('float16').fillna(0)), label_elemento.values.ravel())
        pickles.criarModelo(rnd_clf,"meusmodelos/knn_modelo_elemento")
    else:
        if(TIPO_DADO == "tfidf"):
            rnd_clf = SVC(kernel = "linear", C = 10)
        else:
            rnd_clf = SVC(kernel = "linear", C = 1)# 10 para TFIDF ;1 para OHE ;1 para Combinado
        if(TIPO_DADO =="tfidf"):
            rnd_clf.fit(csr_matrix(X_train), label_elemento.values.ravel())
        elif(TIPO_DADO == "OHE"):
            rnd_clf.fit(X_train, label_elemento.values.ravel())
        else:
            rnd_clf.fit(csr_matrix(X_train.astype('float16').fillna(0)), label_elemento.values.ravel())
        pickles.criarModelo(rnd_clf,"meusmodelos/svc_modelo_elemento")
        
    
def meuModelo_predict_elemento(X_test):
    if(modelo1 == "rf"):
        modelo = pickles.carregarModelo("meusmodelos/randomForest_modelo_elemento")
        if(TIPO_DADO == "combinado"):
            y_predito = modelo.predict(csr_matrix(X_test.astype("float16").fillna(0)))
        else:
            y_predito = modelo.predict(X_test)
        return y_predito
    elif(modelo1 == "rocchio"):
        modelo = pickles.carregarModelo("meusmodelos/rocchio_modelo_elemento")
        if(TIPO_DADO == "combinado"):
            y_predito = modelo.predict(csr_matrix(X_test.astype("float16").fillna(0)))
        else:
            y_predito = modelo.predict(X_test)
        return y_predito
    elif(modelo1 == "knn"):
        modelo = pickles.carregarModelo("meusmodelos/knn_modelo_elemento")
        if(TIPO_DADO == "combinado"):
            y_predito = modelo.predict(csr_matrix(X_test.astype("float16").fillna(0)))
        else:
            y_predito = modelo.predict(X_test)
        return y_predito
    else:
        modelo = pickles.carregarModelo("meusmodelos/svc_modelo_elemento")
        if(TIPO_DADO == "combinado"):
            y_predito = modelo.predict(csr_matrix(X_test.astype("float16").fillna(0)))
        else:
            y_predito = modelo.predict(X_test)
        return y_predito
            
            

def meuModelo_fit_subelemento(X_train,y_train):
    # Copiando a label para separar em elemento e subelemento
    label_elemento = y_train.copy()
    label_subelemento = y_train.copy()
    # Separando os codigos do elemento e subelemento
    for i in range(label_elemento.shape[0]):  
        if(len(str(label_elemento[0].iloc[i]))==3):
            label_elemento[0].iloc[i] = str(y_train[0].iloc[i])[:1]
            label_subelemento[0].iloc[i] = str(y_train[0].iloc[i])[1:]
        else:
            label_elemento[0].iloc[i] = str(y_train[0].iloc[i])[:2]
            label_subelemento[0].iloc[i] = str(y_train[0].iloc[i])[2:]
    # Para cada elemento pegue a posicao de todos seus subelementos e faca o fit    
    for i in range(pd.DataFrame(label_elemento)[0].value_counts().count()):
        elemento = pd.DataFrame(label_elemento)[0].value_counts().index[i]
        posicao = label_elemento.where(label_elemento==str(elemento)).dropna().index
        dados = X_train.iloc[posicao].fillna(0)# adicao para o tfidf
        rotulo = label_subelemento.iloc[posicao]
        # Caso o elemento tenha apenas um subelemento nao precisa de modelo apenas salva no dicionario
        if(rotulo[0].value_counts().count() == 1):
            elemento_um_subelemento[elemento] = rotulo.iloc[0].values[0]
            continue
        if(modelo2 == "rf"):
            modelo = RandomForestClassifier(n_jobs=-1,n_estimators=200,random_state=0,max_samples=int(np.ceil(len(rotulo)*0.3)) )
            if(TIPO_DADO == "combinado"):
                modelo.fit(csr_matrix(dados.astype('float16').fillna(0)), rotulo.values.ravel())
            else:
                modelo.fit(dados, rotulo.values.ravel())
            pickles.criarModelo(modelo,"meusmodelos/randomForest_modelo_"+str(elemento))
        elif(modelo2 == "rocchio"):
            modelo = NearestCentroid()
            if(TIPO_DADO == "combinado"):
                modelo.fit(csr_matrix(dados.astype('float16').fillna(0)), rotulo.values.ravel())
            else:
                modelo.fit(dados, rotulo.values.ravel())
            pickles.criarModelo(modelo,"meusmodelos/rocchio_modelo_"+str(elemento))
        elif(modelo2 == "knn"):
            modelo = KNeighborsClassifier(n_jobs = -1,n_neighbors = 1)
            if(TIPO_DADO == "combinado"):
                modelo.fit(csr_matrix(dados.astype('float16').fillna(0)), rotulo.values.ravel())
            else:
                modelo.fit(dados, rotulo.values.ravel())
            pickles.criarModelo(modelo,"meusmodelos/knn_modelo_"+str(elemento))
        elif(TIPO_DADO=="tfidf"):
            modelo = SVC(kernel = "linear", C = 10)
            modelo.fit(csr_matrix(dados), rotulo.values.ravel())
            pickles.criarModelo(modelo,"meusmodelos/svc_modelo_"+str(elemento))
        elif(TIPO_DADO == "combinado"):
            modelo = SVC(kernel = "linear", C = 1)
            modelo.fit(csr_matrix(dados.astype('float16').fillna(0)), rotulo.values.ravel())
            pickles.criarModelo(modelo,"meusmodelos/svc_modelo_"+str(elemento))
        else:
            modelo = SVC(kernel = "linear", C = 1000)#OHE 1000
            modelo.fit(dados, rotulo.values.ravel())
            pickles.criarModelo(modelo,"meusmodelos/svc_modelo_"+str(elemento))

def meuModelo_predict_subelemento(X_test,y_predito_modelo):
    # criando vetor para salvar as lables na ordem correta
    label_subelemento = [0]*len(y_predito_modelo)
    # para cada elemento leia o modelo e faca o predict para as posicoes que ele aparece 
    for i in range(pd.DataFrame(y_predito_modelo)[0].value_counts().count()):
        elemento = pd.DataFrame(y_predito_modelo)[0].value_counts().index[i]
        # Tenta carregar o modelo caso nao consiga é porque so tem uma classe e esta no dicionario        
        try:            
            if(modelo2 == "rf"):
                modelo = pickles.carregarModelo("meusmodelos/randomForest_modelo_"+str(elemento))
            elif(modelo2 == "rocchio"):
                modelo = pickles.carregarModelo("meusmodelos/rocchio_modelo_"+str(elemento))
            elif(modelo2 == "knn"):
                modelo = pickles.carregarModelo("meusmodelos/knn_modelo_"+str(elemento))
            else:
                modelo = pickles.carregarModelo("meusmodelos/svc_modelo_"+str(elemento))   
        except:
            posicao = pd.DataFrame(y_predito_modelo).where(pd.DataFrame(y_predito_modelo)==elemento).dropna().index
            rotulo = elemento_um_subelemento[elemento]
            for pos in posicao:
                label_subelemento[pos]=rotulo
            continue        
        # Pega todas as posicoes onde foi tem o elemento i
        posicao = pd.DataFrame(y_predito_modelo).where(pd.DataFrame(y_predito_modelo)==elemento).dropna().index
        # Pega os dados dessas posicoes
        dados = X_test.iloc[posicao].fillna(0)#adicao por causa do tfidf
        # Prediz o subelemento para esses dados
        if(TIPO_DADO == "combinado"):
            rotulo = modelo.predict(csr_matrix(dados.astype("float16").fillna(0)))
        else:
            rotulo = modelo.predict(dados)
        # calcula o f1_score individual de cada subelemento do elemento
        if(teste2):
            #pegando apenas o subelmento da label
            y_real = y_test.iloc[posicao].copy()
            for k in range(y_real.shape[0]):
                if(len(str(y_real[0].iloc[k]))==3):
                    y_real[0].iloc[k] = (str(y_real[0].iloc[k])[1:])
                else:
                    y_real[0].iloc[k] = (str(y_real[0].iloc[k])[2:])
            f1_individual = f1_score(y_real,rotulo,average=None)
            #alinhando os subelementos com a f1_individual
            subelementos =pd.concat([pd.DataFrame(rotulo),y_real],axis=0)
            subelementos = subelementos[0].value_counts().sort_index().index
            #salvando a f1_individual de cada subelemento em arquivo
            f = open('f1_elementos.xlsx','a+')
            f.write("Elemento: "+elemento+'\n')
            f.write("Subelemento"+";"+"Score"+'\n')
            for sub in range(len(subelementos)):
                f.write(str(subelementos[sub])+";"+str(f1_individual[sub])+'\n')
            f.write("\n")
            f.flush()
            f.close()
        j=0
        # label na posicao dos dados recebe o subelemento
        for pos in posicao:
            label_subelemento[pos]=rotulo[j]
            j=j+1    
    # Junta elemento com subelemento
    for i in range(len(y_predito_modelo)):
        label_subelemento[i] = int(str(y_predito_modelo[i])+str(label_subelemento[i]))
    return label_subelemento

def meuModelo_evaluate(y_test,y_predito):
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    print("O f1Score micro do meu modelo é: ",micro)
    print("O f1Score macro do meu modelo é: ",macro)
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," randomForest ")

#TIPO_DADO = "OHE"
#TIPO_DADO = "tfidf"
TIPO_DADO = "combinado" ###


if(TIPO_DADO == "tfidf"):
    data, label = tratamentoDados("sem OHE")
    tfidf = tratamentoDados("tfidf")
    X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.3,stratify = label,random_state= 0)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
elif(TIPO_DADO == "OHE"):
    data, label = tratamentoDados("OHE")
    X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label,random_state= 0)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
else:
    data, label = tratamentoDados("OHE")
    tfidf = tratamentoDados("tfidf")
    dados = pd.concat([data.astype(pd.SparseDtype("float32", 0.0)),tfidf], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state= 0)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
# Dicionario com os elementos que tem apenas 1 subelemento
elemento_um_subelemento = {}



#possibilidade = [["rf","rf"],["rf","knn"],["rf","rocchio"],["rf","svc"],["knn","rf"],["knn","knn"],["knn","rocchio"],["knn","svc"],["rocchio","rf"],["rocchio","knn"],["rocchio","rocchio"],["rocchio","svc"],["svc","rf"],["svc","knn"],["svc","rocchio"],["svc","svc"]]
#possibilidade = [["rf","rf"]]#OHE
#possibilidade = [["rf","knn"]]#TFIDF
possibilidade = [["svc","svc"]]#Combinados

teste = False
teste2 = False
#teste = True
#i=0

for i in range(len(possibilidade)):
    prob = possibilidade[i]
    modelo1 = prob[0]
    modelo2 = prob[1]
    meuModelo_fit_elemento(X_train,y_train)
    y_predito_modelo = meuModelo_predict_elemento(X_test)
    if(teste):
        # Verificando a porcentagem de acerto do elemento
        y_test_elemento = [0]*y_test.shape[0]
        for i in range(y_test.shape[0]):
            if(len(str(y_test_elemento[i]))==3):
                y_test_elemento[i]=str(y_test.iloc[i].values[0])[:1]
            else:
                y_test_elemento[i]=str(y_test.iloc[i].values[0])[:2]
        print(modelo1)
        meuModelo_evaluate(y_predito_modelo,y_test_elemento)
    meuModelo_fit_subelemento(X_train,y_train)
    y_predito_completo = meuModelo_predict_subelemento(X_test,y_predito_modelo)
    # Quantidade de acerto total
    print(prob)
    meuModelo_evaluate(y_test,y_predito_completo)