from sklearn.neighbors import KNeighborsClassifier
from tratamentos import salvar_dados
from sklearn.metrics import f1_score


def knn(X_train, X_test, y_train, y_test,string,num_neighbors):
    if(string == "prob"):
        clf = KNeighborsClassifier(n_neighbors=num_neighbors,n_jobs = -1)
        clf.fit(X_train, y_train.values.ravel())
        return clf.predict_proba(X_test)
    clf = KNeighborsClassifier(n_neighbors=num_neighbors,n_jobs = -1)
    clf.fit(X_train, y_train.values.ravel())
    #pickles.criarModelo(clf,"Rocchio "+string)
    y_predito = clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," Knn "+string)
    print("O f1Score micro do Knn ", string ," com ",num_neighbors," vizinhos é: ",micro)
    print("O f1Score macro do Knn ", string ," com ",num_neighbors," vizinhos é: ",macro)
#    return y_predito