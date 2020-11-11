from sklearn.neighbors import NearestCentroid
#from tratamentos import salvar_dados
from sklearn.metrics import f1_score
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from sklearn import preprocessing
import pandas as pd
import numpy as np
import operator

import multiprocessing 
from functools import partial
import time

def centroid_calc(X,y):
    clf = clf = NearestCentroid()
    clf.fit(X, y.values.ravel())
    return pd.DataFrame(clf.centroids_)
    
def classeCheck(x,y):
    if(x==y):
        return 1
    else:
        return 0
        

def search(testDocument,indices,normalized_projected_data,y_train,k):
    # todas as classes dos vizinhos
    classes = y_train[0].iloc[indices]
    if(classes.value_counts().count() ==1):
        return classes.iloc[0]
    # o score de cada classe
    classesScore ={}
    # calcula o score de cada classe
    for i in range(classes.value_counts().count()):
        # pega a classe i
        classe = classes.value_counts().index[i]
        valor_classe_i = 0
        # calcula o score da classe i para todos os vizinhos
        for ind in indices:
            valor_classe_i+= 1-distance.euclidean(testDocument,normalized_projected_data.iloc[ind])*(classeCheck(y_train.iloc[ind].values[0],classe))
        classesScore[classe] = valor_classe_i
    # pega a classe com maior score
    argMax =  max(classesScore.items(), key=operator.itemgetter(1))[0]
    return argMax

def funcao_demorada(doc,centroids):
    # linha projetada
    row = [0]*centroids.shape[0]  
    for j in range(centroids.shape[0]):
        # calculo do cosine similarity para o centroid j
        sim = sum(np.transpose(doc)* centroids.iloc[j].values) / ( np.sqrt(sum(list((x**2 for x in doc)))) *  np.sqrt(sum(list((x**2 for x in centroids.iloc[j].values)))) )
        row[j]=sim
    #salva a linha de todas as similaridades entre doc i e todos os centroids J
    global variavel
    variavel +=1
    if(variavel%100 == 0):
        print(variavel," Documentos finalizados\n")
    return row
variavel=0
def similarity_paralela(docs,centroids):
    # dados projetados
    projected_data = [0]* docs.shape[0]
    # para cada documento calcule a similaridade para cada centroid
    p = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()/2))
    arg_fix = partial(funcao_demorada, centroids=centroids)
    projected_data = p.map(arg_fix, [docs.iloc[i] for i in range(docs.shape[0])])
#    projected_data = p.map(arg_fix, [docs.iloc[i] for i in range(6)])#
#    projected_data = p.map(funcao_demorada, [docs.iloc[i] for i in range(docs.shape[0])])
    p.close()
    return projected_data

def cenknn(X_train, X_test, y_train, y_test,string,k_value):
# =============================================================================
#     Train
# =============================================================================
    # CentroidDR
    centroids = centroid_calc(X_train,y_train) #Ok    
    ini = time.time()
    projected_data = pd.DataFrame(similarity_paralela(X_train,centroids)) #OK
    fim = time.time()
    print("Função paralela: ", fim-ini)
    
#    ini = time.time()
#    projected_data = funcao_multipla(X_train,centroids) #OK
#    fim = time.time()
#    print("Função splited: ", fim-ini)
#    
#    resultado = pd.DataFrame()
#    for i in range(len(projected_data)):
#        resultado = pd.concat([resultado, pd.DataFrame(projected_data[i])],axis =0)
#    resultado.reset_index(drop=True, inplace=True)
#    projected_data = resultado
#    del resultado
    print("1 projecao finalizada")
    # CenKnn
    min_max_scaler = preprocessing.MinMaxScaler() #OK
    normalized_projected_data = pd.DataFrame(min_max_scaler.fit_transform(projected_data)) #OK
    tree = KDTree(normalized_projected_data) #OK
# =============================================================================
#     Test
# =============================================================================
    centroids = centroid_calc(X_test,y_test)
    projected_data = pd.DataFrame(similarity_paralela(X_test,centroids)) #OK
    print("2 projecao finalizada")
    
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_projected_data_test = pd.DataFrame(min_max_scaler.fit_transform(projected_data))
    y_predito =[0]*normalized_projected_data_test.shape[0]
    # para cada documento calcule sua classe de acordo com a equacao do artigo
    for i in range(normalized_projected_data_test.shape[0]):
        testDocument = normalized_projected_data_test.iloc[i]
        # pega a distancia e indice dos k vizinhos mais proximos do doc teste
        dist, indices  = tree.query([testDocument],k=k_value)
        # retorna a classe do doc teste de acordo com o cenknn
        result = search(testDocument,indices[0],normalized_projected_data,y_train,k_value)
        # salva o valor predito
        y_predito[i]= result
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," Knn "+string)
    print("O f1Score micro do cenKnn ", string ," com ",k_value," de K é: ",micro)
    print("O f1Score macro do cenKnn ", string ," com ",k_value," de K é: ",macro)