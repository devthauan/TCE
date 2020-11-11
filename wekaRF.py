import arff
import pandas as pd
import weka.core.jvm as jvm
from sklearn.metrics import f1_score
import weka.classifiers as Classifier
from weka.core.converters import Loader
from preparacaoDados import tratamentoDados
from sklearn.model_selection import train_test_split

# iniciando a maquina virtual
jvm.start()

data, label = tratamentoDados("sem OHE")
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3,stratify = label)
X_train.to_csv('pickles/X_train.csv',index=False)
y_train.to_csv('pickles/y_train.csv',index=False)
X_test.to_csv('pickles/X_test.csv',index=False)
y_test.to_csv('pickles/y_test.csv',index=False)
# Transformando os dados para arff
arff.dump('pickles/X_train.arff', X_train.values, relation='relation', names=X_train.columns)
arff.dump('pickles/y_train.arff', y_train.astype("int").values, relation='relation', names=y_train.columns)


# Lendo os dados transformados
loader = Loader(classname="weka.core.converters.ArffLoader")
X_train = loader.load_file("pickles/X_train.arff")
y_train = loader.load_file("pickles/y_train.arff")

# Criando o classificador
rf = Classifier.Classifier(classname="weka.classifiers.trees.RandomForest")

# Fit
rf.build_classifier(X_train)


import weka.core.version as weka_version
weka_version.weka_version()

for index, inst in enumerate(X_test):
    pred = Classifier.Classifier.classify_instance(inst)
    dist = Classifier.Classifier.distribution_for_instance(inst)
    print(str(index+1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))



micro = f1_score(y_test,y_predito,average='micro')
macro = f1_score(y_test,y_predito,average='macro')
print("O f1Score micro do Knn ", string ," com ",num_neighbors," vizinhos é: ",micro)
print("O f1Score macro do Knn ", string ," com ",num_neighbors," vizinhos é: ",macro)

jvm.stop()