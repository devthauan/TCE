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
from modelos import supportVectorMachine
from sklearn.model_selection import KFold
from preparacaoDados import tratamentoDados
# Aquisicao dos dados
data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
dados =  pd.DataFrame.sparse.from_spmatrix(dados)
dados_stacking =  pd.read_csv("pickles/oraculo/dados_stacking.csv")
print(dados.shape)
# Label Encoder
le = preprocessing.LabelEncoder()
label['natureza_despesa_cod'] = le.fit_transform(label['natureza_despesa_cod'])

# Onde ficara os resultados finais
resultados = pd.DataFrame()

fold_number = 0
kf = KFold(n_splits=5, random_state=0, shuffle=True)
for train_index, test_index in kf.split(dados):
#    DADOS NORMAIS
    X_train, X_test = dados[:].iloc[train_index], dados[:].iloc[test_index]
    y_train, y_test = label[:].iloc[train_index], label[:].iloc[test_index]
    X_train_text, X_test_text = tfidf[:].iloc[train_index], tfidf[:].iloc[test_index]
#    DADOS STACKING
    X_train_stacking, X_test_stacking = dados_stacking[:].iloc[train_index], dados_stacking[:].iloc[test_index]
    y_train_stacking, y_test_stacking = label[:].iloc[train_index], label[:].iloc[test_index]
#    break
# =============================================================================
#     TREINAMENTO DOS MODELOS
# =============================================================================
    randomforest.randomForest(X_train, X_test, y_train, y_test,"Randon Forest Fold "+ str(fold_number))
    sgd.sgd(X_train, X_test, y_train, y_test,"SGD Fold "+ str(fold_number),100)
    knn.knn(X_train, X_test, y_train, y_test,"Knn Fold "+ str(fold_number),1)
    naive_bayes.naivebayes(X_train.to_numpy(), X_test.to_numpy(), y_train, y_test,"NB Fold "+ str(fold_number))
    xboost.xboost(X_train, X_test, y_train, y_test,"XGBoost Fold "+ str(fold_number),10,label.value_counts().count())
    X_train_svm =  csr_matrix(X_train); X_test_svm =  csr_matrix(X_test)
    supportVectorMachine.svc(X_train_svm, X_test_svm, y_train, y_test,"SVC 0.1 Fold "+ str(fold_number),0.1)
    supportVectorMachine.svc(X_train_svm, X_test_svm, y_train, y_test,"SVC 10 Fold "+ str(fold_number),10)
    X_train_svm_stacking =  csr_matrix(X_train_stacking); X_test_svm_stacking =  csr_matrix(X_test_stacking)
    supportVectorMachine.svc(X_train_svm_stacking, X_test_svm_stacking, y_train, y_test,"Stacking SVC 100 Fold "+ str(fold_number),100)
# =============================================================================
#     PREDICAO DOS MODELOS
# =============================================================================
    resultados_fold = pd.DataFrame()
    resultados_fold_prob = pd.DataFrame()
    modelos = ["Randon Forest Fold ","Knn Fold ","NB Fold ","SGD Fold "]
    modelos_especiais = ["Stacking SVC 100 Fold ","XGBoost Fold ","SVC 0.1 Fold ","SVC 10 Fold "]
    for i in range(len(modelos)):
        rf_model = pickles.carregarModelo("oraculo/"+modelos[i]+ str(fold_number))
        #class
        y_result = pd.DataFrame(rf_model.predict(X_test.to_numpy()))
        y_result.index = X_test.index
        y_result.columns = [modelos[i]+ str(fold_number)]
        resultados_fold = pd.concat([resultados_fold,y_result], axis = 1)
        #prob
        y_result_prob =pd.DataFrame(rf_model.predict_proba(X_test.to_numpy()))
        y_result_prob.index = X_test.index
        y_result_prob_maior =[0]*len(y_result_prob)
        for j in range(len(y_result_prob)):
            y_result_prob_maior[j] = max(y_result_prob.iloc[j])
        y_result_prob = pd.DataFrame(y_result_prob_maior)
        y_result_prob.columns = [modelos[i]+ str(fold_number)]
        resultados_fold_prob = pd.concat([resultados_fold_prob,y_result_prob], axis = 1)
# =============================================================================
#     STACKING
# =============================================================================
    rf_model = pickles.carregarModelo("oraculo/"+modelos_especiais[0]+ str(fold_number))
    y_result_prob = pd.DataFrame(rf_model.predict_proba(X_test_stacking.to_numpy()))
    y_result  = pd.DataFrame(y_result_prob.idxmax(axis = 1))
    y_result.index = X_test_stacking.index
    y_result.columns = [modelos_especiais[0]+ str(fold_number)]
    resultados_fold = pd.concat([resultados_fold,y_result], axis = 1)
    #prob
    y_result_prob_maior =[0]*len(y_result_prob)
    y_result_prob.index = X_test_stacking.index
    for j in range(len(y_result_prob)):
        y_result_prob_maior[j] = max(y_result_prob.iloc[j])
    y_result_prob = pd.DataFrame(y_result_prob_maior)
    y_result_prob.columns = [modelos_especiais[0]+ str(fold_number)]
    resultados_fold_prob = pd.concat([resultados_fold_prob,y_result_prob], axis = 1)
# =============================================================================
#     XGBOOST
# =============================================================================
    rf_model = pickles.carregarModelo("oraculo/"+modelos_especiais[1]+ str(fold_number))
    dtest = xgb.DMatrix(X_test)
    y_result_prob = pd.DataFrame(rf_model.predict(dtest))
    y_result  = pd.DataFrame(y_result_prob.idxmax(axis = 1))
    y_result.index = X_test.index
    y_result.columns = [ modelos_especiais[1]+ str(fold_number)]
    resultados_fold = pd.concat([resultados_fold,y_result], axis = 1)
    #prob
    y_result_prob_maior =[0]*len(y_result_prob)
    y_result_prob.index = X_test.index
    for j in range(len(y_result_prob)):
        y_result_prob_maior[j] = max(y_result_prob.iloc[j])
    y_result_prob = pd.DataFrame(y_result_prob_maior)
    y_result_prob.columns = [modelos_especiais[1]+ str(fold_number)]
    resultados_fold_prob = pd.concat([resultados_fold_prob,y_result_prob], axis = 1)
# =============================================================================
#     SVC
# =============================================================================
    for k in [2,3]:
        rf_model = pickles.carregarModelo("oraculo/"+modelos_especiais[k]+ str(fold_number))
        y_result_prob = pd.DataFrame(rf_model.predict_proba(X_test.to_numpy()))
        y_result  = pd.DataFrame(y_result_prob.idxmax(axis = 1))
        y_result.index = X_test.index
        y_result.columns = [modelos_especiais[k]+ str(fold_number)]
        resultados_fold = pd.concat([resultados_fold,y_result], axis = 1)
        #prob
        y_result_prob_maior =[0]*len(y_result_prob)
        y_result_prob.index = X_test.index
        for j in range(len(y_result_prob)):
            y_result_prob_maior[j] = max(y_result_prob.iloc[j])
        y_result_prob = pd.DataFrame(y_result_prob_maior)
        y_result_prob.columns = [modelos_especiais[k]+ str(fold_number)]
        resultados_fold_prob = pd.concat([resultados_fold_prob,y_result_prob], axis = 1)
# =============================================================================
    resultados_alg = []
    for l in range(len(resultados_fold)):
        #Colunas que acertaram
        columns = resultados_fold.iloc[l].where(resultados_fold.iloc[l].values == y_test["natureza_despesa_cod"].iloc[l]).dropna().index
        try:
            result_l = resultados_fold_prob[columns].iloc[l].idxmax(axis =1)
            resultados_alg.append(result_l)
        except:
            result_l = resultados_fold_prob.iloc[l].idxmax(axis =1)
            resultados_alg.append(result_l)
    
    resultados_alg = pd.DataFrame(resultados_alg)
    resultados_alg.index = X_test.index   
    resultados = pd.concat([resultados,resultados_alg],axis = 0)    
    fold_number += 1
resultados = pd.concat([resultados,label], axis = 1) 