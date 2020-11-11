from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from modelos import analise_cluster
import pandas as pd
import sys
sys.path.insert(1, '/projetoTCE/tratamentos')
from tratamentos import salvar_dados
from tratamentos import pickles

def rocchio2(X_train, X_test, y_train, y_test,string):
    clf = NearestCentroid()
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    nca_pipe = Pipeline([('nca', nca), ('clf', clf)])
    nca_pipe.fit(X_train, y_train.values.ravel())
    clf=nca_pipe
    clf.fit(X_train, y_train.values.ravel())
    #pickles.criarModelo(clf,"Rocchio "+string)
    y_predito = clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," Rocchio "+string)
    print("O f1Score micro do Rocchio2.0 ", string ," é: ",micro)
    print("O f1Score macro do Rocchio2.0 ", string ," é: ",macro)
