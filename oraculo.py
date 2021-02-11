import numpy as np
import pandas as pd
import xgboost as xgb
from modelos import knn
from modelos import sgd
from scipy import sparse
from modelos import xboost
from modelos import naive_bayes
from tratamentos import pickles
from modelos import randomforest
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from modelos import supportVectorMachine
from sklearn.model_selection import KFold
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split
# Aquisicao dos dados
data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
del data
# Separacao dos dados em trino e teste
dados =  pd.DataFrame.sparse.from_spmatrix(dados)
dados, dados_test, label, label_test = train_test_split(dados, label,test_size=0.2,stratify = label,random_state =5)
# Resetando os indexes
dados.reset_index(drop= True, inplace = True)
label.reset_index(drop= True, inplace = True)
dados_test.reset_index(drop= True, inplace = True)
label_test.reset_index(drop= True, inplace = True)
print(dados.shape)
# Label Encoder
le = preprocessing.LabelEncoder()
label['natureza_despesa_cod'] = le.fit_transform(label['natureza_despesa_cod'])
# Variaveis
TREINAR_MODELOS = 0
folds_index = [] #Armazena a divisao dos folds
resultados = pd.DataFrame() # Onde ficara os resultados da primeira parte do algoritmo
fold_number = 0
kf = KFold(n_splits=5, random_state=0, shuffle=True)
for train_index, test_index in kf.split(dados):
    X_train, X_test = dados[:].iloc[train_index], dados[:].iloc[test_index]
    y_train, y_test = label[:].iloc[train_index], label[:].iloc[test_index]
    folds_index.append(test_index) 
#    break
# =============================================================================
#     TREINAMENTO DOS MODELOS
# =============================================================================
    if(TREINAR_MODELOS):
        randomforest.randomForest(X_train, X_test, y_train, y_test,"Randon Forest Fold "+ str(fold_number))
        sgd.sgd(X_train, X_test, y_train, y_test,"SGD Fold "+ str(fold_number),100)
        knn.knn(X_train, X_test, y_train, y_test,"Knn Fold "+ str(fold_number),1)
        naive_bayes.naivebayes(X_train.to_numpy(), X_test.to_numpy(), y_train, y_test,"NB Fold "+ str(fold_number))
        xboost.xboost(X_train, X_test, y_train, y_test,"XGBoost Fold "+ str(fold_number),10,label.value_counts().count())
        X_train_svm =  csr_matrix(X_train); X_test_svm =  csr_matrix(X_test)
        supportVectorMachine.svc(X_train_svm, X_test_svm, y_train, y_test,"SVC 0.1 Fold "+ str(fold_number),0.1)
        supportVectorMachine.svc(X_train_svm, X_test_svm, y_train, y_test,"SVC 10 Fold "+ str(fold_number),10)
# =============================================================================
#     PREDICAO DOS MODELOS
# =============================================================================
    resultados_fold_prob = pd.DataFrame()
    modelos = ["XGBoost Fold ","Randon Forest Fold ","Knn Fold ","NB Fold ","SGD Fold ","SVC 0.1 Fold ","SVC 10 Fold "]
    for i in range(len(modelos)):
        rf_model = pickles.carregarModelo("oraculo/"+modelos[i]+ str(fold_number))
        try:
            y_result_prob = rf_model.predict_proba(X_test.to_numpy())
            novo_y = [0]*len(y_result_prob)
            for m in range(len(y_result_prob)):
                novo_y[m] = [y_result_prob[m]]
            y_result_prob = pd.DataFrame(novo_y,columns = [modelos[i]])
        except:
            dtest = xgb.DMatrix(X_test)
            y_result_prob = rf_model.predict(dtest)
            novo_y = [0]*len(y_result_prob)# Transforma varias colunas em um array
            for m in range(len(y_result_prob)):
                novo_y[m] = [y_result_prob[m]]
            y_result_prob = pd.DataFrame(novo_y,columns = [modelos[i]])
        y_result_prob.index = X_test.index
        resultados_fold_prob = pd.concat([resultados_fold_prob,y_result_prob], axis = 1)
# =============================================================================
    resultados = pd.concat([resultados,resultados_fold_prob],axis = 0)    
    fold_number += 1
resultados = resultados.sort_index()# Resultados
# =============================================================================
# =============================================================================
# # SEGUNDA PARTE
# =============================================================================
# =============================================================================
possibilidades = [[1,1],[1,2],[2,1],[2,2]]
for pos in range(len(possibilidades)):
    # Estragetia estocastica 1, Estrategia comite 2
    Estrategia = possibilidades[pos][0]
    Algoritmo_oraculo =  possibilidades[pos][1]
    # Inicio
    DOCUMENTOS_SEM_ACERTO = 0
    algoritmos = []
    for i in range(dados.shape[0]):
        algs = []# Algoritmos que acertaram para o documento i
        for j in range(len(resultados.columns)):# Verifica os algoritmos que acertaram a predicao
            if(list(resultados[resultados.columns[j]].iloc[i]).index(np.float64(max(resultados[resultados.columns[j]].iloc[i]))) == label["natureza_despesa_cod"].iloc[i]):
                algs.append(resultados.columns[j])
        if(Estrategia == 1):
            if(len(algs)>0):# Se mais de um alg acertar selecionar um aleatoriamente
                aleat = np.random.randint(0,len(algs))
                algoritmos.append(algs[aleat])
            else:# Se nenhum acertar seleciona o SVC
                algoritmos.append('SVC 10 Fold ')
                DOCUMENTOS_SEM_ACERTO+=1
        elif(Estrategia == 2):
            if(len(algs)>0):# Armazena os algoritmos que acertaram
                algoritmos.append(algs)
            else:# Se nenhum alg acertar coloca todos para fazer o comite na proxima etapa
                algoritmos.append(modelos)
                DOCUMENTOS_SEM_ACERTO+=1
    
    #Funcao que transforma lista em string       
    def listToString(algoritmos):  
        lista = [0]*len(algoritmos)
        for i in range(len(algoritmos)):
            linha = ""  
            if(len(algoritmos[i])>1):
                for alg in range(len(algoritmos[i])):
                    if(alg != len(algoritmos[i])-1):
                        linha += algoritmos[i][alg]+","
                    else:
                        linha += algoritmos[i][alg]
            else:
                linha = algoritmos[i][0]
            lista[i] = linha
        return lista
    
    if(Estrategia == 2):
        algoritmos = listToString(algoritmos)
    # Label Encoder
    encoder = preprocessing.LabelEncoder()
    algoritmos = pd.DataFrame(encoder.fit_transform(algoritmos))
    # =============================================================================
    # ALGORITMO DO ORACULO
    # =============================================================================
    if(Algoritmo_oraculo == 1):
        melhor_alg = "SVC 10 -fixed"
        from sklearn.svm import SVC
        clf = SVC(kernel="linear",C = 10,random_state = 10)
        clf.fit(csr_matrix(dados), algoritmos.values.ravel())
        y_predito = clf.predict(csr_matrix(dados_test))
        y_alg = pd.DataFrame(encoder.inverse_transform(y_predito))
    elif(Algoritmo_oraculo == 2):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.linear_model import SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier
        import xgboost as xgb
        # Pega os algoritmos
        y_alg = pd.DataFrame(encoder.inverse_transform(algoritmos.values.ravel()))
        # Transforma a string em vetor
        y_aux = [0]*len(y_alg)
        for l in range(len(y_alg)):
            y_aux[l] = y_alg[0].iloc[l].split(",")
        # Achata o vetor
        flat_list = [item for sublist in y_aux for item in sublist]
        # Pega o com maior frequencia
        melhor_alg = pd.DataFrame(flat_list).value_counts().index[0][0]
        # Treina o modelo de acordo com o algoritmo
        if("Randon" in melhor_alg):
            clf = RandomForestClassifier(n_jobs=-1,n_estimators=200,random_state=10)
        elif("Knn" in melhor_alg):
            clf = KNeighborsClassifier(n_neighbors=1,n_jobs = -1)
        elif("NB" in melhor_alg):
            clf = GaussianNB()
        elif("SGD" in melhor_alg):
            clf = SGDClassifier(loss="log", penalty="l2", max_iter = 100,random_state = 10)
        elif("SVC 0.1" in melhor_alg):
            clf = SVC(kernel="linear",C = 0.1,random_state = 10)
        elif("SVC 10" in melhor_alg):
            clf = SVC(kernel="linear",C = 10,random_state = 10)   
        
        if("XGBoost" in melhor_alg):
            param = {'max_depth':6, 'objective': 'multi:softmax', "num_class":algoritmos.value_counts().count()}
            dtrain = xgb.DMatrix(dados, algoritmos)
            bst = xgb.train(param, dtrain, 10)
            dtest = xgb.DMatrix(dados_test)
            y_predito = bst.predict(dtest)
            y_predito = pd.DataFrame(y_predito).astype("int16")
            y_alg = pd.DataFrame(encoder.inverse_transform(y_predito.values.ravel()))
        elif("NB" in melhor_alg):
            clf.fit(dados, algoritmos.values.ravel())
            y_alg = clf.predict(dados_test)
            y_alg = pd.DataFrame(encoder.inverse_transform(y_alg))
        else:
            clf.fit(csr_matrix(dados), algoritmos.values.ravel())
            y_alg = clf.predict(csr_matrix(dados_test))
            y_alg = pd.DataFrame(encoder.inverse_transform(y_alg))
    # =============================================================================
    # Estrategia II
    # =============================================================================
    if(Estrategia == 2):
        # String to list
        y_aux = [0]*len(y_alg)
        for l in range(len(y_alg)):
            y_aux[l] = y_alg[0].iloc[l].split(",")
        y_alg = y_aux
        #Atribuindo o numero dos folds
        for i in range(len(y_alg)):
            if(i in folds_index[0]):
                y_alg[i] = [y_alg[i][x]+"0" for x in range(len(y_alg[i]))]
            elif(i in folds_index[1]):
                y_alg[i] = [y_alg[i][x]+"1" for x in range(len(y_alg[i]))]
            elif(i in folds_index[2]):
                y_alg[i] = [y_alg[i][x]+"2" for x in range(len(y_alg[i]))]
            elif(i in folds_index[3]):
                y_alg[i] = [y_alg[i][x]+"3" for x in range(len(y_alg[i]))]
            else:
                y_alg[i] = [y_alg[i][x]+"4" for x in range(len(y_alg[i]))]
        # Para cada linha faz a predicao da classe e da porcentagem de certeza para cada alg
        y_final = pd.DataFrame([0]*len(y_alg))
        y_final_porcent = pd.DataFrame([0]*len(y_alg))
        for k in range(len(y_alg)):# Para cada linha
            linha = [0]*len(y_alg[k])# Para cada vetor de algoritmos 
            linha_porcent = [0]*len(y_alg[k])# Para cada vetor de porcentagem
            for m in range(len(y_alg[k])):# Cada item do vetor de algoritmos
                model = pickles.carregarModelo("oraculo/"+y_alg[k][m])
                if('XGBoost' in y_alg[k][m]):
                    dtest = xgb.DMatrix(pd.DataFrame(dados_test.iloc[k]).T)# Converte para o formato aceito
                    result_xgb = pd.DataFrame(model.predict(dtest))# Faz a predicao_prob do documento
                    linha[m] = result_xgb.idxmax(axis = 1)[0]# Seleciona qual a predicao
                    linha_porcent[m] = result_xgb[result_xgb.idxmax(axis = 1)[0]][0]# Seleciona a porcentagem de certeza
                else:
                     linha[m] = pd.DataFrame(model.predict([dados_test.iloc[k]]))[0][0]# Faz a predicao do documento
                     linha_porcent[m] = max(model.predict_proba([dados_test.iloc[k]])[0])# Seleciona a porcentagem de certeza
            y_final[0].iloc[k] = str(linha)[1:-1]# Vetor dos resultados das predicoes
            y_final_porcent[0].iloc[k] = str(linha_porcent)[1:-1]# Vetor dos resultados das porcentagens
        # String to list
        for l in range(len(y_final)):
            y_final[0].iloc[l] = y_final[0].iloc[l].split(",")
            y_final_porcent[0].iloc[l] = y_final_porcent[0].iloc[l].split(",")
        y_final = pd.DataFrame(y_final)
        y_final_porcent = pd.DataFrame(y_final_porcent)
        # Comite de decisoes
        for h in range(len(y_final)):
            # Se existe mais de um resultado verificar se existe um mais frequente, em caso de empate verificar porcentagem
            if(len(pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts()) >1):
                # Verifica se os 3 mais frequentes tem a mesma frequencia
                if((len(pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts()) >=3) and (pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().values[0] == pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().values[1])  and (pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().values[1] == pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().values[2])):
                    disputantes = [0]*3# Vao disputar para ver qual vai ser escolhida
                    for w in range(len(disputantes)):
                        disputantes[w] =  pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().index[w][0]
                    # Vetor com as porcentagens de certeza referente a cada algoritmo
                    index_disputante_0 = []
                    index_disputante_1 = []
                    index_disputante_2 = []
                    for u in range(len(y_final[0].iloc[h])):
                        if(list(pd.DataFrame(y_final[0].iloc[h]).astype("int16")[0].values)[u] == disputantes[0]):
                            index_disputante_0.append(u)
                        elif(list(pd.DataFrame(y_final[0].iloc[h]).astype("int16")[0].values)[u] == disputantes[1]):
                            index_disputante_1.append(u)
                        elif(list(pd.DataFrame(y_final[0].iloc[h]).astype("int16")[0].values)[u] == disputantes[2]):
                            index_disputante_2.append(u)
                    # Calcula a media da porcentagem para determinada classe
                    porcentagem_disputante_0 = np.mean(pd.DataFrame(y_final_porcent[0].iloc[h]).astype("float16").iloc[index_disputante_0])[0]
                    porcentagem_disputante_1 = np.mean(pd.DataFrame(y_final_porcent[0].iloc[h]).astype("float16").iloc[index_disputante_1])[0]
                    porcentagem_disputante_2 = np.mean(pd.DataFrame(y_final_porcent[0].iloc[h]).astype("float16").iloc[index_disputante_2])[0]
                    #Verifica a classe com maior porcentagem                
                    if(porcentagem_disputante_0 > porcentagem_disputante_1 and porcentagem_disputante_0 > porcentagem_disputante_2):
                        y_final[0].iloc[h] = disputantes[0]
                    elif(porcentagem_disputante_1 > porcentagem_disputante_0 and porcentagem_disputante_1 > porcentagem_disputante_2):
                        y_final[0].iloc[h] = disputantes[1]
                    else:
                        y_final[0].iloc[h] = disputantes[2]
                # Caso haja apenas empate duplo
                elif(pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().values[0] == pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().values[1]):
                    disputantes = [0]*2
                    disputantes[0], disputantes[1] = pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().index[0][0], pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().index[1][0]
                    index_disputante_0 = []
                    index_disputante_1 = []
                    for u in range(len(y_final[0].iloc[h])):
                        if(list(pd.DataFrame(y_final[0].iloc[h]).astype("int16")[0].values)[u] == disputantes[0]):
                            index_disputante_0.append(u)
                        elif(list(pd.DataFrame(y_final[0].iloc[h]).astype("int16")[0].values)[u] == disputantes[1]):
                            index_disputante_1.append(u)
                    porcentagem_disputante_0 = np.mean(pd.DataFrame(y_final_porcent[0].iloc[h]).astype("float16").iloc[index_disputante_0])[0]
                    porcentagem_disputante_1 = np.mean(pd.DataFrame(y_final_porcent[0].iloc[h]).astype("float16").iloc[index_disputante_1])[0]
                    if(porcentagem_disputante_0> porcentagem_disputante_0):
                        y_final[0].iloc[h] = disputantes[0]
                    else:
                        y_final[0].iloc[h] = disputantes[1]
                # Caso tenha mais de um valor mas tenha um mais frequente que os outros
                else:
                    y_final[0].iloc[h] = pd.DataFrame(y_final[0].iloc[h]).astype("int16").value_counts().index[0][0]
            else:#caso so exista um resultado utilizar ele
                y_final[0].iloc[h] = int(y_final[0].iloc[h][0])
    # =============================================================================
    # Estrategia I 
    # =============================================================================
    elif(Estrategia == 1):
        #Atribuindo o numero dos folds
        for i in range(len(y_alg)):
            if(i in folds_index[0]):
                y_alg[0].iloc[i] = y_alg[0].iloc[i] + "0"
            elif(i in folds_index[1]):
                y_alg[0].iloc[i] = y_alg[0].iloc[i] + "1"
            elif(i in folds_index[2]):
                y_alg[0].iloc[i] = y_alg[0].iloc[i] + "2"
            elif(i in folds_index[3]):
                y_alg[0].iloc[i] = y_alg[0].iloc[i] + "3"
            else:
                y_alg[0].iloc[i] = y_alg[0].iloc[i] + "4"
        # Para cada algoritmo pegue os documentos que sao relacionados e faz o predict
        y_final = pd.DataFrame()
        for k in range(y_alg.value_counts().count()):
            model = pickles.carregarModelo("oraculo/"+y_alg.value_counts().index[k][0])# Carrega o modelo
            # Pega o index dos documentos pertencentes aquele modelo
            docs_classe_k_index = y_alg.where(y_alg[0] == y_alg.value_counts().index[k][0]).dropna().index
            # Pega os documentos para aquele modelo
            docs_classe_k = dados_test.iloc[docs_classe_k_index]
            # Faz as predicoes
            if("XGBoost" in y_alg.value_counts().index[k][0]):
                dtest = xgb.DMatrix(docs_classe_k)
                y_pred = pd.DataFrame(model.predict(dtest))
                y_pred.index = docs_classe_k_index
                y_final= pd.concat([y_final,y_pred.idxmax(axis = 1)],axis = 0)
            else:
                y_pred = pd.DataFrame(model.predict(docs_classe_k.to_numpy()))
                y_pred.index = docs_classe_k_index
                y_final= pd.concat([y_final,y_pred],axis = 0)
    # =============================================================================
    # RESULTADOS
    # =============================================================================
    y_final = y_final.sort_index()
    y_final = y_final.astype("int16")
    y_final = pd.DataFrame(le.inverse_transform(y_final.values.ravel()))
    # Calcula a Micro e Macro F1
    micro = f1_score(label_test,y_final,average='micro')
    macro = f1_score(label_test,y_final,average='macro')
    print("Estrategia: ", Estrategia, "Algoritmo: ", Algoritmo_oraculo, " ", melhor_alg)
    print("O f1Score micro do Oraculo é: ",micro)
    print("O f1Score macro do Oraculo é: ",macro)   
    # Quantidade de documentos que nao tiveram acerto por nenhum algoritmo
    print("Documentos sem acerto ",DOCUMENTOS_SEM_ACERTO)