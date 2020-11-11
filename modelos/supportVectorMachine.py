from sklearn.svm import SVC
from sklearn.metrics import f1_score
import sys
import numpy as np
sys.path.insert(1, '/projetoTCE/tratamentos')
from tratamentos import salvar_dados
from tratamentos import pickles
from scipy.sparse import csr_matrix

def svc(X_train, X_test, y_train, y_test,string,valor_c):
    if(string == "prob"):
#        clf = SVC(kernel="linear",C= valor_c,probability = True,random_state=0)
        clf = SVC(class_weight = "balanced",kernel="linear",C= valor_c,probability = True,random_state=0)
        clf.fit(X_train, y_train.values.ravel())
        return clf.predict_proba(X_test)
    if(string == "prob_sparse"):
        clf = SVC(kernel="linear",C= valor_c,probability = True,random_state=0)
        clf.fit(csr_matrix(X_train.values.astype(np.float64)), y_train.values.ravel())
        return clf.predict_proba(X_test)
    clf =SVC(kernel="linear",C= valor_c,random_state=0)
    if(string == "COM TFIDF"):
        clf.fit(csr_matrix(X_train.values.astype(np.float64)), y_train.values.ravel())
    #    pickles.criarModelo(clf,"SVC com C: "+str(valor_c)+" "+string)
        y_predito = clf.predict(csr_matrix(X_test.values.astype(np.float64)))
    else:
        clf.fit(X_train, y_train.values.ravel())
        y_predito = clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual,"SVC com parametro C = "+str(valor_c)+" e com "+string)
    print("O f1Score micro do SVC ",string," com parametro C = ",valor_c,"é: ",micro)
    print("O f1Score macro do SVC ",string," com parametro C = ",valor_c,"é: ",macro)
#    return y_predito