import pandas as pd
from embedder import preprocessing
from embedder.regression import Embedder
from embedder.assessment import visualize
from sklearn.model_selection import train_test_split
from preparacaoDados import tratamentoDados

train, label = tratamentoDados("sem OHE")


cat_sz = preprocessing.categorize(train)
emb_sz = preprocessing.pick_emb_dim(cat_sz)
X_encoded, encoders = preprocessing.encode_categorical(train)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, label)

embedder = Embedder(emb_sz)
embedder.fit(X_train[:10000], y_train[:10000], epochs=1)

preds = embedder.predict(X_train[10000:11000])

embeddings = embedder.get_embeddings()

embedded = embedder.transform(X_train[:10000])