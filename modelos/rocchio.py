from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score
from modelos import analise_cluster
import pandas as pd
import sys
sys.path.insert(1, '/projetoTCE/tratamentos')
from tratamentos import salvar_dados
from tratamentos import pickles

def rocchio(X_train, X_test, y_train, y_test,string):
    clf = NearestCentroid()
    clf.fit(X_train, y_train.values.ravel())
    #pickles.criarModelo(clf,"Rocchio "+string)
    if("Fold" in string):
        pickles.criarModelo(clf,"oraculo/"+string) #SALVAR MODELO
    y_predito = clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," Rocchio "+string)
    print("O f1Score micro do Rocchio ", string ," é: ",micro)
    print("O f1Score macro do Rocchio ", string ," é: ",macro)
    
def cluster(X_train, X_test, y_train, y_test):
    clf = NearestCentroid()
    clf.fit(X_train, y_train.values.ravel())
    y_predito = clf.predict(X_test)
    analise_cluster.analisecluster(y_test,y_predito)