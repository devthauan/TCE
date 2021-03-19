import numpy as np
import pandas as pd
from scipy import sparse
from modelos import randomforest
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from preparacaoDados import tratamentoDados
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data, label = tratamentoDados("OHE")
tfidf = tratamentoDados("tfidf") 
dados = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
print(dados.shape)
del data,tfidf

X_train, X_test, y_train, y_test = train_test_split(dados, label,test_size=0.3,stratify = label,random_state =5)

# =============================================================================
# # Primeira floresta
# =============================================================================
rnd_clf = RandomForestClassifier(n_jobs=-1,n_estimators=200,
                                 random_state=0,max_samples=int((X_train.shape[0]+(X_test.shape[0]))*0.3))
rnd_clf.fit(X_train, y_train.values.ravel())
y_predito_prob = rnd_clf.predict_proba(X_test)

# =============================================================================
# # Segunda floresta
# =============================================================================
texto = tratamentoDados("texto")
cv = CountVectorizer(binary = True)
data_cv = cv.fit_transform(texto)
tfidf = pd.DataFrame.sparse.from_spmatrix(data_cv, columns = cv.get_feature_names())
# Pegando 30% dos dados para calcular profundidade
X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.7,stratify = label,random_state =5)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.3,random_state =5)

resultados = []
profundidade = list(range(1,20,1))
for prof in profundidade:
    rnd_clf = RandomForestClassifier(class_weight ="balanced_subsample",n_jobs=-1,n_estimators=200,
                                 random_state=0,max_samples=int((X_train.shape[0]+(X_test.shape[0]))*0.3),
                                 max_depth = prof)
    rnd_clf.fit(X_train, y_train.values.ravel())
    resultados.append(rnd_clf.score(X_test,y_test))
    print("Score: ",rnd_clf.score(X_test,y_test)," Profundidade: ",prof)


X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.3,stratify = label,random_state =5)
rnd_clf = RandomForestClassifier(class_weight ="balanced_subsample",n_jobs=-1,n_estimators=200,
                                 random_state=0,max_samples=int((X_train.shape[0]+(X_test.shape[0]))*0.3),
                                 max_depth = resultados.index(max(resultados))+1)
rnd_clf.fit(X_train, y_train.values.ravel())
y_predito_prob_2 = rnd_clf.predict_proba(X_test)
y_predito_final = pd.DataFrame((y_predito_prob + y_predito_prob_2)/2)
for i in range(len(y_predito_final)):
    y_predito_final.iloc[i] = rnd_clf.classes_[np.where(y_predito_final.iloc[i] ==y_predito_final.iloc[i].max())[0][0]]
y_predito_final = y_predito_final[0]
micro = f1_score(y_test,y_predito_final,average='micro')
macro = f1_score(y_test,y_predito_final,average='macro')
print("O f1Score micro do random Forest 2.0 é: ",micro)
print("O f1Score macro do random Forest 2.0 é: ",macro)
