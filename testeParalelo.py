from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados
from sklearn.metrics import f1_score
from thundersvm import SVC

valor_c = "padrao"
data, label = tratamentoDados("OHE")
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label)

clf = SVC()
clf.fit(X_train.to_numpy(), y_train.values.ravel())
y_predito = clf.predict(X_test.to_numpy())

micro = f1_score(y_test,y_predito,average='micro')
macro = f1_score(y_test,y_predito,average='macro')
print("O f1Score micro do SVC com parametro C = ",valor_c,"é: ",micro)
print("O f1Score macro do SVC com parametro C = ",valor_c,"é: ",macro)