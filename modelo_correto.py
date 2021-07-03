# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from tratamentos import pickles
from preparacaoDados import filtro
from scipy.sparse import csr_matrix
from tratarDados import tratarDados
from sklearn.metrics import f1_score
from tratamentos import tratar_texto
from preparacaoDados import tratamentoDados
from sklearn.ensemble import RandomForestClassifier


#dados_tce = pd.read_csv("arquivos/dadosTCE.csv",low_memory=False)
#dados_juliete = pd.read_excel("dados_analisados.xlsx")
## retirando os dados analisados do conjunto principal
#indexes = [0]*dados_juliete.shape[0]
#for i in range(dados_juliete.shape[0]):
#    indexes[i] = dados_tce['Empenho (Sequencial Empenho)(EOF)'][dados_tce['Empenho (Sequencial Empenho)(EOF)'] == dados_juliete['Empenho (Sequencial Empenho)(EOF)'].iloc[i]].index[0]
#dados_tce.drop(indexes,inplace = True)
#dados_tce.reset_index(drop = True, inplace = True)
#del indexes
#pickles.criarPickle(dados_tce, "dados_tce_limpos")


dados_tce_originais = pickles.carregarPickle("dados_tce_limpos")[:1000] # 265246
dados_juliete_originais = pd.read_excel("dados_analisados.xlsx")
dados_juliete_originais.insert (25, "Empenho (Sequencial Empenho)(EOF).1", dados_juliete_originais["Empenho (Sequencial Empenho)(EOF)"])


# deixando apenas os dados corretos
dados_juliete_originais = dados_juliete_originais[dados_juliete_originais["ANÁLISE"] == "OK"]
dados_juliete_originais.drop("ANÁLISE",axis = 1, inplace = True)
dados_juliete_originais.reset_index(drop = True, inplace = True)
# =============================================================================
# CO-TRAINING
# =============================================================================
# inicializando o vetor de documentos a serem adicionados a cada ciclo
documentos_mais_90_certeza = []
for co in range(5):
    # faz uma copia dos dados antes do tratamento
    dados_juliete = dados_juliete_originais.copy()
    dados_tce = dados_tce_originais.copy()
    # pega os empenhos pelo identificador
    indexes_novos_dados = [0]*len(documentos_mais_90_certeza)
    for j in range(len(documentos_mais_90_certeza)):
        indexes_novos_dados[j] = dados_tce[dados_tce["Empenho (Sequencial Empenho)(EOF)"] == documentos_mais_90_certeza[j] ].index[0]
    documentos_mais_90_certeza = dados_tce.iloc[indexes_novos_dados]
    # adiciona os novos empenhos ao conjunto de dados principal
    dados_juliete_originais = pd.concat([dados_juliete, documentos_mais_90_certeza], axis = 0)
    dados_juliete_originais.reset_index(drop = True, inplace = True)
    dados_juliete = dados_juliete_originais.copy()
    # retirando os documentos ja selecionados do teste
    dados_tce_originais.drop(indexes_novos_dados,axis = 0, inplace = True)
    dados_tce_originais.reset_index(drop = True, inplace = True)
    dados_tce = dados_tce_originais.copy()
    #
    tratamentoDados(dados_juliete.copy(),"tfidf")
    tratamentoDados(dados_juliete.copy(),"OHE")
    dados_juliete = pickles.carregarPickle("data")
    label_juliete = pickles.carregarPickle("label")
    tfidf_juliete = pickles.carregarPickle("tfidf")
    dados_juliete = sparse.hstack((csr_matrix(dados_juliete),csr_matrix(tfidf_juliete) ))
    del tfidf_juliete
    print(dados_juliete.shape)
    # retirando as classes que nao estao no treino
    naturezas_tce = pd.DataFrame(dados_tce['Natureza Despesa (Cod)(EOF)'])
    naturezas_juliete = list(label_juliete['natureza_despesa_cod'].value_counts().index)
    fora_do_modelo = []
    for i in range(len(naturezas_tce)):
        if(naturezas_tce.iloc[i][0] not in naturezas_juliete):
            fora_do_modelo.append(i)
    dados_tce.drop(fora_do_modelo,inplace = True)
    dados_tce.reset_index(inplace = True,drop=True)
    del naturezas_tce,fora_do_modelo, naturezas_juliete
    # tratando os dados de teste
    dados_teste, label_teste = tratarDados(dados_tce)
    print("treino ",dados_juliete.shape)
    print("teste ",dados_teste.shape)
    # =============================================================================
    # RF
    # =============================================================================
    modelo = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state = 10)
    modelo.fit(dados_juliete, label_juliete.values.ravel())
    y_predito = modelo.predict(dados_teste)
    micro = f1_score(label_teste,y_predito,average='micro')
    macro = f1_score(label_teste,y_predito,average='macro')
    print("O f1Score micro do RandomForest com ",200,"arvores é: ",micro)
    print("O f1Score macro do RandomForest com ",200,"arvores é: ",macro)
    
    # pegando o indentificador dos empenhos de teste
    identificador_empenho = pickles.carregarPickle("modelos_tratamentos/identificador_empenho")
    # pegando a predicao com as probabilidades
    y_predito_prob = pd.DataFrame(modelo.predict_proba(dados_teste))
    # selecionando os empenhos com certeza acima ou igual a 90%
    documentos_mais_90_certeza = []
    for i in range(len(y_predito_prob)):
        if(max(y_predito_prob.iloc[i])>=0.9):
            documentos_mais_90_certeza.append(identificador_empenho.iloc[i][0])
