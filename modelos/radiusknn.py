from sklearn.neighbors import RadiusNeighborsClassifier
from tratamentos import salvar_dados
from sklearn.metrics import f1_score


def radius(X_train, X_test, y_train, y_test,string,valor):
    if(string == "prob"):
        clf = RadiusNeighborsClassifier(radius=valor,weights='distance',n_jobs = -1)
        clf.fit(X_train, y_train.values.ravel())
        return clf.predict_proba(X_test)
    clf = RadiusNeighborsClassifier(radius=valor,weights='distance',n_jobs = -1)
    clf.fit(X_train, y_train.values.ravel())
    #pickles.criarModelo(clf,"Rocchio "+string)
    y_predito = clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," Knn "+string)
    print("O f1Score micro do RadiusKnn ", string ," com ",valor," de raio é: ",micro)
    print("O f1Score macro do RadiusKnn ", string ," com ",valor," de raio é: ",macro)
#    return y_predito
