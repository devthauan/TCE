import xgboost as xgb
from sklearn.metrics import f1_score
from tratamentos import salvar_dados
from tratamentos import pickles


def xboost(X_train, X_test, y_train, y_test,string,num_estagios,numero_classes,max_depth):
    if("Fold" in string):
        param = {'max_depth':max_depth,'objective':'multi:softprob',"eval_metric":"logloss", "num_class":numero_classes}
        num_round = num_estagios
        dtrain = xgb.DMatrix(X_train, y_train)
        bst = xgb.train(param, dtrain, num_round)
        pickles.criarModelo(bst,"oraculo/"+string) #SALVAR MODELO
        return 0
    else:
        param = {'max_depth':max_depth, 'objective': 'multi:softmax', "num_class":numero_classes}
        num_round = num_estagios
        dtrain = xgb.DMatrix(X_train, y_train)
        bst = xgb.train(param, dtrain, num_round)
    
    dtest = xgb.DMatrix(X_test)
    y_predito = bst.predict(dtest)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," xboost "+string)
    print("O f1Score micro do Xboost ", string ," com ",num_estagios," estagios e ",max_depth," profundidade é: ",micro)
    print("O f1Score macro do Xboost ", string ," com ",num_estagios," estagios e ",max_depth," profundidade é: ",macro)
