from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados
from bert_sklearn import BertClassifier
from sklearn.metrics import f1_score


data, label = tratamentoDados("sem OHE")
tfidf = tratamentoDados("tfidf")
#X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.3,stratify = label,random_state =5)
X_train, X_test, y_train, y_test = train_test_split(tfidf, label,test_size=0.3,random_state =5)
del data, tfidf
# Definindo modelo
model = BertClassifier()# text/text pair classification

# Treino
model.fit(X_train, y_train.values.ravel())

# Predicoes
y_predito = model.predict(X_test)

# Resultados
micro = f1_score(y_test,y_predito,average='micro')
macro = f1_score(y_test,y_predito,average='macro')
print("O f1Score micro do Bert é: ",micro)
print("O f1Score macro do Bert é: ",macro)
