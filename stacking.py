import sys
import pandas as pd
from modelos import knn
from sklearn.svm import SVC
from tratamentos import pickles
from modelos import randomforest
from preparacaoDados import filtro
from scipy.sparse import csr_matrix
from tratarDados import tratarDados
from tratamentos import tratar_label
from tratamentos import tratar_texto
from sklearn.metrics import f1_score
from modelos import supportVectorMachine
from preparacaoDados import tratamentoDados
from sklearn.neighbors import KNeighborsClassifier
from tratarDados import refinamento_hiperparametros
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
#sys.argv = ["stacking.py","treino"]

data = pd.read_csv("dadosTCE.csv",low_memory = False)[:500]
data.drop("Empenho (Sequencial Empenho)(EOF).1", axis = 1, inplace = True)
colunas = data.columns
data.columns = [c.lower().replace(' ', '_') for c in data.columns]
data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
data = filtro(data.copy())
data.columns = colunas
label = data["Natureza Despesa (Cod)(EOF)"]
# Retirando naturezas com numero de empenhos menor ou igual a x
label, index_label_x_empenhos = tratar_label.label_elemento(label, 6)
data.drop(index_label_x_empenhos,inplace = True, axis = 0)
data.reset_index(drop = True, inplace = True)
del index_label_x_empenhos
if(sys.argv[1]=="treino"):
    # Separando 40% dos dados para selecao de hiperparametros
    data, data_teste, label, label_teste = train_test_split(data, label, test_size = 0.6,stratify = label, random_state = 10)
    del data_teste, label_teste
    # Resetando os indexes dos dados
    data.reset_index(drop = True, inplace = True)
    label.reset_index(drop = True, inplace = True)
    print(data.shape)


#iniciando vetores
resultadoOHE = pd.DataFrame([])
resultadoTFIDF = pd.DataFrame([])
rotulo = pd.DataFrame([])
kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 10)
for train_index, test_index in kf.split(data, label):
# =============================================================================
#         Primeira parte
# =============================================================================
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    data_train.reset_index(drop = True, inplace = True)
    data_test.reset_index(drop = True, inplace = True)
    # Tratando os dados de treino
    tratamentoDados(data_train.copy(),'tfidf')
    tratamentoDados(data_train.copy(),'OHE')
    # Carregando os dados tratados
    X_train = csr_matrix(pickles.carregarPickle("data"))
    X_train_text = csr_matrix(pickles.carregarPickle("tfidf"))
    y_train = pickles.carregarPickle("label")
    # Aplicando o tratamento nos dados de teste
    X_test, y_test = tratarDados(data_test.copy(),'OHE')
    X_test = csr_matrix(X_test)
    X_test_text, y_test = tratarDados(data_test.copy(),'tfidf')
    X_test_text = csr_matrix(X_test_text)
    y_test.reset_index(drop = True, inplace = True)
    # Algoritmo 1
    y_prob_predito = pd.DataFrame(randomforest.randomForest(X_train, X_test, y_train, y_test,"prob",1020))
    y_prob_predito.index = test_index
    resultadoOHE = pd.concat([resultadoOHE,y_prob_predito],axis=0)
# =============================================================================
#         Segunda parte
# =============================================================================
    y_prob_predito_text = pd.DataFrame(supportVectorMachine.svc(X_train_text, X_test_text, y_train, y_test,"prob",10))
    y_prob_predito_text.index = test_index
    resultadoTFIDF = pd.concat([resultadoTFIDF,y_prob_predito_text],axis=0)
    #rotulo do test na ordem correta
    y_test.index = test_index
    rotulo = pd.concat([rotulo,y_test],axis = 0)
# =============================================================================
#     Terceira parte
# =============================================================================
def testeModelo(X_train, X_test, y_train, y_test,modelo):
    modelo.fit(X_train, y_train.values.ravel())
    y_predito = modelo.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    return micro, macro

resultadoOHE.sort_index(inplace = True)
resultadoTFIDF.sort_index(inplace = True)
rotulo.sort_index(inplace = True)
dados = pd.concat([resultadoOHE,resultadoTFIDF],axis = 1)
# Salva os dados do stacking para usar no active learning
dados.to_csv('dados_stacking.csv', index=False)
del data,label, data_test, data_train,resultadoOHE,resultadoTFIDF, test_index,train_index,y_prob_predito,y_prob_predito_text,y_test,y_train
#

if(sys.argv[1]=="treino"):
    # Retirando naturezas com numero de empenhos menor ou igual a x
    rotulo, index_label_x_empenhos = tratar_label.label_elemento(rotulo["natureza_despesa_cod"], 9)
    dados.drop(index_label_x_empenhos,inplace = True, axis = 0)
    dados.reset_index(drop = True, inplace = True)
    del index_label_x_empenhos
    print("tamanho dos dados: ",dados.shape)
    f = open("stacking_hiperparametro.txt","a+")
    possibilidade = ["rf","knn","svc"]
    X_train, X_test, y_train, y_test = train_test_split(dados, rotulo,test_size=0.3,stratify = rotulo,random_state= 0)
    maior_macro = 0
    melhor_modelo = 0
    for i in range(len(possibilidade)):
        print(possibilidade[i])
        espalhamento = 3
        hiperparametros = {}
        if(possibilidade[i] == "rf"):
            modelo = RandomForestClassifier(n_jobs=-1,random_state=10,max_samples=int((X_train.shape[0]+(X_test.shape[0]))*0.3))
            hiperparametros["n_estimators"]=[100,300,500,700,1000]
            hiper_refinado = refinamento_hiperparametros(X_train, y_train, modelo, hiperparametros, espalhamento)
            modelo.set_params(**hiper_refinado)
            micro, macro = testeModelo(X_train, X_test, y_train, y_test,modelo)
            if(macro >maior_macro):
                maior_macro = macro
                melhor_modelo = modelo
            print("micro: ",micro," macro: ",macro)
            f.write("RF n_estimators: "+str(hiper_refinado)+"\n")
            f.write("\n")
            f.flush()
        elif(possibilidade[i] == "knn"):
            modelo = knn_modelo = KNeighborsClassifier(n_jobs = -1)
            hiperparametros['n_neighbors'] = [1,3,5,7] 
            hiper_refinado = refinamento_hiperparametros(X_train, y_train, modelo, hiperparametros, espalhamento)
            modelo.set_params(**hiper_refinado)
            micro, macro = testeModelo(X_train, X_test, y_train, y_test,modelo)
            if(macro >maior_macro):
                maior_macro = macro
                melhor_modelo = modelo
            print("micro: ",micro," macro: ",macro)
            f.write("KNN n_neighbors: "+str(hiper_refinado)+"\n")
            f.write("\n")
            f.flush()
        else:
            modelo = SVC(kernel="linear",random_state=10)
            hiperparametros['C']=[0.1,1,10,100] 
            hiper_refinado = refinamento_hiperparametros(X_train, y_train, modelo, hiperparametros, espalhamento)
            modelo.set_params(**hiper_refinado)
            micro, macro = testeModelo(X_train, X_test, y_train, y_test,modelo)
            if(macro >maior_macro):
                maior_macro = macro
                melhor_modelo = modelo
            print("micro: ",micro," macro: ",macro)
            f.write("SVM C: "+str(hiper_refinado)+"\n")
            f.write("\n")
            f.flush()
    pickles.criarModelo(melhor_modelo,"melhor_modelo")
        
else:
    print("tamanho dos dados: ",dados.shape)
    X_train, X_test, y_train, y_test = train_test_split(dados, rotulo,test_size=0.3,stratify = rotulo,random_state= 0)
    modelo = pickles.carregarModelo("melhor_modelo")
    micro, macro = testeModelo(X_train, X_test, y_train, y_test,modelo)
    print("Melhor modelo foi",type(modelo).__name__," com a micro: ",micro," e a macro: ", macro)
