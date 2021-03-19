import pickle
import pandas as pd
from scipy import sparse
from datetime import date
from sklearn.svm import SVC
from conexaoDados import dados
from tratamentos import pickles
from tratarDados import tratarDados
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from preparacaoDados import tratamentoDados
# 1 para treinar e 0 caso contrario
TREINAR_MODELO = 1
data_atual = date.today().strftime('%d/%m/%Y')

if(TREINAR_MODELO):
    tratamentoDados('tfidf')
    tratamentoDados('OHE')
    data = pickles.carregaPickle("data")
    label = pickles.carregaPickle("label")
    tfidf = pickles.carregaPickle("tfidf")
    aux = sparse.hstack((csr_matrix(data),csr_matrix(tfidf) ))
    aux = pd.DataFrame.sparse.from_spmatrix(aux)
    data = csr_matrix(aux)
    del aux, tfidf
    print(data.shape)
    #Treinar modelo
    modelo = SVC(kernel="linear",C= 10,random_state=0)
    modelo.fit(data, label.values.ravel())
    with open('pickles/modelos_tratamentos/modelo_SVM.pk', 'wb') as fin:
        pickle.dump(modelo, fin)
else:
    with open('pickles/modelos_tratamentos/modelo_SVM.pk', 'rb') as pickle_file:
        modelo = pickle.load(pickle_file)

dados_hoje, label_hoje = tratarDados(dados())
identificador_empenho = pickles.carregaPickle("modelos_tratamentos/identificador_empenho")
y_predito = modelo.predict(dados_hoje)
micro = f1_score(label_hoje,y_predito,average='micro')
macro = f1_score(label_hoje,y_predito,average='macro')
string = ""
valor_c = 10
print("O f1Score micro do SVC ",string," com parametro C = ",valor_c,"é: ",micro)
print("O f1Score macro do SVC ",string," com parametro C = ",valor_c,"é: ",macro)
resultado = pd.concat([pd.DataFrame(identificador_empenho),pd.DataFrame(label_hoje)],axis = 1)
resultado = pd.concat([resultado,pd.DataFrame(y_predito)],axis = 1)
resultado['data_predicao'] = data_atual
colunas = ['empenho_sequencial_empenho','natureza_real','natureza_predita','data_predicao']
resultado.columns = colunas
# =============================================================================
# LOG
# =============================================================================
f = open('log.txt','a+')
f.write(data_atual+'\n')
f.write("Quantidade de documentos preditos: "+str(label_hoje.shape[0]))
f.flush()
f.close()