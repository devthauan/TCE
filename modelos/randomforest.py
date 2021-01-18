from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sys
sys.path.insert(1, '/projetoTCE/tratamentos')
from tratamentos import salvar_dados
from tratamentos import pickles

def randomForest(X_train, X_test, y_train, y_test,string):    
#    rnd_clf = RandomForestClassifier(n_jobs=-1,n_estimators=200,random_state=0,max_samples=int((len(X_train)+(len(X_test)))*0.3))
    rnd_clf = RandomForestClassifier(n_jobs=-1,n_estimators=200,random_state=0,max_samples=int((X_train.shape[0]+(X_test.shape[0]))*0.3))
#    rnd_clf = RandomForestClassifier(class_weight ="balanced",n_jobs=-1,n_estimators=200,random_state=0,max_samples=int((X_train.shape[0]+(X_test.shape[0]))*0.3))
#    rnd_clf = RandomForestClassifier(class_weight ="balanced_subsample",n_jobs=-1,n_estimators=200,random_state=0,max_samples=int((X_train.shape[0]+(X_test.shape[0]))*0.3))
    rnd_clf.fit(X_train, y_train.values.ravel())
    if(string == "prob"):
        y_predito_prob = rnd_clf.predict_proba(X_test)
        return y_predito_prob
    if("Fold" in string):
        pickles.criarModelo(rnd_clf,"oraculo/"+string) #SALVAR MODELO
        return 0
    y_predito = rnd_clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," randomForest "+string)
    print("O f1Score micro do random Forest ",string," é: ",micro)
    print("O f1Score macro do random Forest ",string," é: ",macro)
#    return y_predito
        