import time
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from tratamentos import pickles
from tratarDados import tratarDados
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from tratamentos import tratar_label
from preparacaoDados import tratamentoDados
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tratarDados import refinamento_hiperparametros
from sklearn.model_selection import train_test_split

def load_dados(pouca_natureza, porcentagem_split):
    # Importando os dados do tribunal
    data = pd.read_csv("dadosTCE.csv", low_memory = False)[:500]
    data.drop("Empenho (Sequencial Empenho)(EOF).1", axis = 1, inplace = True)
    label = data["Natureza Despesa (Cod)(EOF)"]
    # Retirando naturezas com numero de empenhos menor ou igual a x
    label, index_label_x_empenhos = tratar_label.label_elemento(label, pouca_natureza)
    data.drop(index_label_x_empenhos,inplace = True, axis = 0)
    data.reset_index(drop = True, inplace = True)
    # Separando X% dos dados para selecao de hiperparametros
    data_treino, data_teste, label_treino, label_teste = train_test_split(data, label, test_size = porcentagem_split,stratify = label, random_state = 10)
    del data, label, label_teste, label_treino
    # Resetando os indexes dos dados
    data_treino.reset_index(drop = True, inplace = True)
    data_teste.reset_index(drop = True, inplace = True)
    return data_treino, data_teste

def tratar_dados(data_treino, data_teste, teste):
    tratamentoDados(data_treino.copy(), "OHE")
    tratamentoDados(data_treino.copy(), "tfidf")
    # Carregar os dados tratados
    data_treino = pickles.carregarPickle("data")
    label_treino = pickles.carregarPickle("label")
    tfidf_treino = pickles.carregarPickle("tfidf")
    # Retirando naturezas com numero de empenhos menor que X depois da limpesa
    label_treino, index_label_x_empenhos = tratar_label.label_elemento(label_treino["natureza_despesa_cod"], 2)
    label_treino = pd.DataFrame(label_treino)["natureza_despesa_cod"]
    data_treino.drop(index_label_x_empenhos,inplace = True, axis = 0)
    data_treino.reset_index(drop = True, inplace = True)
    tfidf_treino.drop(index_label_x_empenhos,inplace = True, axis = 0)
    tfidf_treino.reset_index(drop = True, inplace = True)
    del index_label_x_empenhos
    # Tamanhos dos dados de treino tratados
    print("OHE_treino",data_treino.shape)
    print("TF-IDF_treino",tfidf_treino.shape)
    visao_dupla_treino = csr_matrix( sparse.hstack((csr_matrix(data_treino),csr_matrix(tfidf_treino) )) )
    print("Visao dupla, dados combinados do treino",visao_dupla_treino.shape)
    if(teste):
        # Aplicar o tratamento no teste
        tfidf_teste, label_teste = tratarDados(data_teste.copy(),'tfidf')
        data_teste, label_teste = tratarDados(data_teste.copy(),'OHE')
        # Retirando naturezas com numero de empenhos menor que X depois da limpesa
        label_teste, index_label_x_empenhos = tratar_label.label_elemento(label_teste["natureza_despesa_cod"], 2)
        label_teste = pd.DataFrame(label_teste)["natureza_despesa_cod"]
        data_teste.drop(index_label_x_empenhos,inplace = True, axis = 0)
        data_teste.reset_index(drop = True, inplace = True)
        tfidf_teste.drop(index_label_x_empenhos,inplace = True, axis = 0)
        tfidf_teste.reset_index(drop = True, inplace = True)
        # Tamanhos dos dados de treino tratados
        print("OHE_teste",data_teste.shape)
        print("TF-IDF_teste",tfidf_teste.shape)
        visao_dupla_teste = csr_matrix( sparse.hstack((csr_matrix(data_teste),csr_matrix(tfidf_teste) )) )
        print("Visao dupla, dados combinados do testes",visao_dupla_teste.shape)
        return data_treino, label_treino, data_teste, label_teste, tfidf_treino, tfidf_teste, visao_dupla_treino, visao_dupla_teste
    else:
        return data_treino, label_treino, tfidf_treino, visao_dupla_treino

data_treino, data_teste = load_dados(6, 0.6)
teste = 0
data_treino, label_treino, tfidf_treino, visao_dupla_treino = tratar_dados(data_treino, data_teste, teste)
# =============================================================================
# Gridsearch
# =============================================================================

# Especificando os modelos
modelo_1 = RandomForestClassifier(n_jobs = -1, random_state = 10, max_samples = int(label_treino.shape[0]*0.3) )
modelo_2 = knn_modelo = KNeighborsClassifier(n_jobs = -1)
modelo_3 = SGDClassifier(loss = "log", random_state = 10)
modelo_4 = SVC(kernel = "linear", random_state = 10)
#
modelos = [modelo_1, modelo_2, modelo_3, modelo_4]
modelos_nome = ["Random Forest", "Knn", "SGD", "SVC"]
resultados_modelos = [0]* len(modelos)
# Abrindo arquivo de texto para salvar os resultados da avaliacao de hiperparametros
f = open('HiperParametros.txt','a+')
strings = ["TF-IDF","visao dupla","OHE"]
OHE_hiperparametro = TFIDF_hiperparametro = VISAO_DUPLA_hiperparametro = []
melhores_parametros = [1]*len(modelos)
# Rodando o codigo para cada tipo de dado
for string in strings:
    ''' Definindo a lista de todos hiperparametros e intervalo de valores a serem
    avaliados por todos os algoritmos. A ordem dos hiperparametros se referem aos 
    seguintes algoritmos: Random Forest, KNN, SGD, SVC ''' 
    parametros_todos_modelos = [{'n_estimators':[100,300,500,700,1000] },
                        {'n_neighbors':[1,3,5,7] },
                        {'max_iter':[100,300,500,700] },
                        {'C':[0.1,1,10,100] },]
    
    if("OHE" in string):
        data_grid =  csr_matrix(data_treino.copy())
        f.write(string +"\n")
        print(string)
    elif("TF-IDF" in string):
        data_grid = csr_matrix(tfidf_treino.copy())
        f.write(string +"\n")
        print(string)
    else:
        data_grid = csr_matrix(visao_dupla_treino.copy())
        f.write(string +"\n")
        print(string)
    
    # Fazendo o refinamento de hiperparametros para cada algoritmo
    for i in range(len(parametros_todos_modelos)):
        # Aplicando o refinamento
        if(i == 3 and string == "OHE"):
            break
        else:
            melhores_parametros[i] = refinamento_hiperparametros(data_grid, label_treino, modelos[i], parametros_todos_modelos[i], 3)
            # Obtendo a melhor acuracia e os melhores valores dos hiperparametros
        print("Melhor hiperparametro do modelo %s: %s" % (modelos_nome[i],melhores_parametros[i]))
        # Salvando resultado em arquivo
        f.write("Melhor acuracia do modelo %s: %s" % (modelos_nome[i],melhores_parametros[i]))
        f.write('\n')
    # Salvando os hiperparametros
    if("OHE" in string):
        OHE_hiperparametro = melhores_parametros.copy()
    elif("TF-IDF" in string):
        TFIDF_hiperparametro = melhores_parametros.copy()
    else:
        VISAO_DUPLA_hiperparametro = melhores_parametros.copy()
    f.write('\n')
    f.flush()

# =============================================================================
# Aplicando os hiperparametros
# =============================================================================
data_treino, data_teste = load_dados(2, 0.2)
teste = 1
data_treino, label_treino, data_teste, label_teste, tfidf_treino, tfidf_teste, visao_dupla_treino, visao_dupla_teste = tratar_dados(data_treino, data_teste, teste)
resultado_modelo_refinado = open('Resultado_modelos_refinados.txt','a+')
for string in strings:
    if("OHE" in string):
        treino = csr_matrix(data_treino.copy())
        teste = csr_matrix(data_teste.copy())
        hiperparametros = OHE_hiperparametro
        resultado_modelo_refinado.write(string +"\n")
        print(string)
    elif("TF-IDF" in string):
        treino = csr_matrix(tfidf_treino.copy())
        teste = csr_matrix(tfidf_teste.copy())
        hiperparametros = TFIDF_hiperparametro
        resultado_modelo_refinado.write(string +"\n")
        print(string)
    else:
        treino = csr_matrix(visao_dupla_treino.copy())
        teste = csr_matrix(visao_dupla_teste.copy())
        hiperparametros = VISAO_DUPLA_hiperparametro
        resultado_modelo_refinado.write(string +"\n")
        print(string)
    # Obtendo a melhor acuracia e os melhores valores dos hiperparametros
    for i in range(len(modelos)):
        if(modelos_nome[i] == "SVC" and string == "OHE"):
            break
        modelos[i].set_params(**hiperparametros[i])
        inicio = time.time()
        modelos[i].fit(treino,label_treino)
        fim = time.time()
        tempo = fim - inicio
        y_predito = modelos[i].predict(teste)
        micro = f1_score(label_teste,y_predito,average='micro')
        macro = f1_score(label_teste,y_predito,average='macro')
        print("O f1Score micro do ",modelos_nome[i]," com ",hiperparametros[i]," é: ",micro)
        print("O f1Score macro do ",modelos_nome[i]," com ",hiperparametros[i]," é: ",macro)
        print("Tempo de execucao: ",tempo)
        resultado_modelo_refinado.write("O f1Score micro do "+modelos_nome[i]+" com "+str(hiperparametros[i])+" é: "+str(micro)+"\n")
        resultado_modelo_refinado.write("O f1Score macro do "+modelos_nome[i]+" com "+str(hiperparametros[i])+" é: "+str(macro)+"\n")
        resultado_modelo_refinado.write("Tempo de execucao: "+str(tempo)+"\n")
    resultado_modelo_refinado.write("\n")
    resultado_modelo_refinado.flush()
resultado_modelo_refinado.close()