from sklearn.neighbors import KNeighborsClassifier
from tratamentos import salvar_dados
from sklearn.metrics import f1_score
from tratamentos import pickles
import pandas as pd
from sklearn import preprocessing


def knn(X_train, X_test, y_train, y_test,string,num_neighbors):
    if(string == "prob"):
        clf = KNeighborsClassifier(n_neighbors=num_neighbors,n_jobs = -1)
        clf.fit(X_train, y_train.values.ravel())
#        distancia = pd.DataFrame(clf.kneighbors(X_test,return_distance= True)[0])
#        min_max_scaler = preprocessing.MinMaxScaler()
        return clf.predict(X_test)#,min_max_scaler.fit_transform(distancia)
    clf = KNeighborsClassifier(n_neighbors=num_neighbors,n_jobs = -1)
    clf.fit(X_train, y_train.values.ravel())
    if("Fold" in string):
        pickles.criarModelo(clf,"oraculo/"+string) #SALVAR MODELO
        return 0
    y_predito = clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    print("O f1Score micro do Knn ", string ," com ",num_neighbors," vizinhos é: ",micro)
    print("O f1Score macro do Knn ", string ," com ",num_neighbors," vizinhos é: ",macro)
