from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import sys
sys.path.insert(1, '/projetoTCE/tratamentos')
from tratamentos import salvar_dados
from tratamentos import pickles

def sgd(X_train, X_test, y_train, y_test,string,max_interations):   
    rnd_clf = SGDClassifier(loss="log", penalty="l2", max_iter = max_interations,random_state = 0)
    rnd_clf.fit(X_train, y_train.values.ravel())
    if(string == "prob"):
        y_predito_prob = rnd_clf.predict_proba(X_test)
        return y_predito_prob
    #pickles.criarModelo(rnd_clf,"SGD "+string)
    if("Fold" in string):
        pickles.criarModelo(rnd_clf,"oraculo/"+string) #SALVAR MODELO
        return 0
    y_predito = rnd_clf.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    #f1_individual = f1_score(y_test,y_predito,average=None)    
    #salvar_dados.salvar(y_test,y_predito,micro, macro, f1_individual," SGD "+string)
    print("O f1Score micro do Stocastico gradiente descendente com ",string," é: ",micro)
    print("O f1Score macro do Stocastico gradiente descendente com ",string," é: ",macro)        