from sklearn.metrics import f1_score
from tratamentos import salvar_dados
from sklearn.ensemble import GradientBoostingClassifier

def xboost(X_train, X_test, y_train, y_test,string,num_estagios):
    clf = GradientBoostingClassifier(random_state=0, n_estimators = num_estagios)
    clf.fit(X_train, y_train.values.ravel())
    y_predito = clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," xboost "+string)
    print("O f1Score micro do GradientBoosting ", string ," com ",num_estagios," estagios é: ",micro)
    print("O f1Score macro do GradientBoosting ", string ," com ",num_estagios," estagios é: ",macro)