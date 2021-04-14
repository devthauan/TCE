import numpy as np
from sklearn.svm import SVC
from tratamentos import pickles
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

data = csr_matrix(pickles.carregaPickle("data"))
label = pickles.carregaPickle("label")
kf = KFold(n_splits=3, random_state=10, shuffle=True)

vetor_micro = []
vetor_macro = []
for train_index, test_index in kf.split(data):
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label.iloc[train_index], label.iloc[test_index]
    modelo = SVC(kernel="linear",C= 10,random_state=0)
    modelo.fit(X_train, y_train.values.ravel())
    y_predito = modelo.predict(X_test)
    micro = f1_score(y_test,y_predito,average='micro')
    macro = f1_score(y_test,y_predito,average='macro')
    vetor_micro.append(micro)
    vetor_macro.append(macro)
print("Micros: ",vetor_micro)
print("Macros: ",vetor_macro)
