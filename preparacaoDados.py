### Bibliotecas python ###
import pandas as pd
from sklearn import preprocessing
### Meus pacotes ###
from tratamentos import tratar_label
from tratamentos import one_hot_encoding
from tratamentos import tratar_texto
### Meus pacotes ###

def tratamentoDados(escolha):
    # Nome do arquivo csv a ler
    nomeArquivoDadosBrutos = 'dadosTCE.csv';
    # Carrega os dados na variavel 'data' utilizando o Pandas
    data = pd.read_csv(nomeArquivoDadosBrutos, encoding = "utf-8",low_memory = False)
    del nomeArquivoDadosBrutos
#    data = data.iloc[0:1000]
    # Trata o nome das colunas para trabalhar melhor com os dados
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
    data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
    # Deletando empenhos sem relevancia devido ao saldo zerado
#    index = data["valor_saldo_do_empenho"].where(data["valor_saldo_do_empenho"] == 0).dropna().index
#    data.drop(index,inplace = True)
#    data.reset_index(drop=True, inplace=True)
    # Deleta colunas que atraves de analise foram identificadas como nao uteis
    data = data.drop(['empenho_sequencial_empenho.1','classificacao_orcamentaria_descricao',
                      'natureza_despesa_nome','valor_estorno_anulacao_empenho',
                      'valor_anulacao_cancelamento_empenho','fonte_recurso_cod','elemento_despesa'], axis='columns')
    # Funcao que gera o rotulo e retorna as linhas com as naturezas de despesa que so aparecem em 1 empenho
    label,linhas_label_unica = tratar_label.tratarLabel(data)
    label.reset_index(drop=True, inplace=True)
    # Excluindo as naturezas de despesas que so tem 1 empenho
    data = data.drop(data.index[linhas_label_unica])
    data.reset_index(drop=True, inplace=True)
    del linhas_label_unica
    #Excluindo empenhos irrelevantes devido nao estarem mais em vigencia
#    sem_relevancia = pd.read_excel("analise/Naturezas de despesa com vigência encerrada.xlsx")
#    sem_relevancia = sem_relevancia['Nat. Despesa']
#    sem_relevancia = [sem_relevancia.iloc[i].split('.')[3:][0]+sem_relevancia.iloc[i].split('.')[3:][1] for i in range(len(sem_relevancia))]
#    sem_relevancia = pd.DataFrame(sem_relevancia)
#    excluir = []
#    for i in range(len(sem_relevancia[0].value_counts())):
#        excluir.append( label.where(label[0] == int(sem_relevancia[0].value_counts().index[i])).dropna().index )
#        
#    excluir = [item for sublist in excluir for item in sublist]
#    label.drop(excluir,inplace =True)
#    data.drop(excluir,inplace = True)
#    data.reset_index(drop=True, inplace=True)
#    label.reset_index(drop=True, inplace=True)
# =============================================================================
#     classes_100_menos = []
#    for i in range(label[0].value_counts().count()):
#        if(label[0].value_counts(ascending = True).values[i]<100):
#            classes_100_menos.append(label[0].value_counts(ascending = True).index[i])
#    indexes = []
#    # Percorre as labels pegando index onde elas aparacem
#    for j in range(len(classes_100_menos)):
#        aux = label.where(label[0]== classes_100_menos[j]).dropna().index
#        # Adiciona os indexes da classe j no vetor
#        for k in range(len(aux)):
#            indexes.append(aux[k])
#    data.drop(indexes, inplace = True)
#    label.drop(indexes, inplace = True)
#    data.reset_index(drop=True, inplace=True)
#    label.reset_index(drop=True, inplace=True)
# =============================================================================
    
    if(escolha == "tfidf"):
        # Funcao que limpa o texto retira stopwords acentos pontuacao etc.
        textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
        # Função que gera o TF-IDF do texto tratado
        tfidf = tratar_texto.calculaTFIDF(textoTratado)
        del textoTratado
        return tfidf
    if(escolha == "texto"):
         # Funcao que limpa o texto retira stopwords acentos pontuacao etc.
        textoTratado = tratar_texto.cleanTextData(data["empenho_historico"])
        return textoTratado
    # Tratamento dos dados ########################################################
    # Codigo que gera o meta atributo "pessoa_juridica" onde 1 representa que a pessoa e juridica e 0 caso seja fisica
    identificacao_pessoa = [0] * data.shape[0]
    for i in range(data.shape[0]):
      if(data['beneficiario_cpf'].iloc[i] == "-"):
        identificacao_pessoa[i] = 1
      else: identificacao_pessoa[i]=0
    data['pessoa_juridica'] = identificacao_pessoa
    del identificacao_pessoa
    data['pessoa_juridica'] = data['pessoa_juridica'].astype("int8")
    data = data.drop(["beneficiario_cpf/cnpj","beneficiario_cpf","beneficiario_cnpj","beneficiario_nome"], axis='columns')
    
    # Codigo que gera o meta atributo "orgao_sucedido" onde 1 representa que o orgao tem um novo orgao sucessor e 0 caso contrario
    orgao_sucedido = [0] * data.shape[0]
    for i in range(data.shape[0]):
      if(data['orgao'].iloc[i] != data['orgao_sucessor_atual'].iloc[i]):
        orgao_sucedido[i] = 1
      else:
        orgao_sucedido[i] = 0
    data['orgao_sucedido'] = orgao_sucedido
    del orgao_sucedido
    data['orgao_sucedido'] = data['orgao_sucedido'].astype("int8")
    data = data.drop(["orgao"], axis='columns')
    
    # Codigo que retira o codigo de programa (retirando 10 valores)
    nome = [0] * data.shape[0]
    for i in range(len(data['programa'])):
      nome[i] = (data['programa'].iloc[i][7:])
    data['programa'] = nome
    del nome
    
    # Codigo que retira o codigo de acao (retirando 82 valores)
    nome = [0] * data.shape[0]
    for i in range(len(data['acao'])):
      nome[i] = (data['acao'].iloc[i][7:])
    data['acao'] = nome
    del nome
    
    # Codigo que concatena acao e programa
    acao_programa = [0] * data.shape[0]
    for i in range(data.shape[0]):
      acao_programa[i] = (data['acao'].iloc[i] + " & " + data['programa'].iloc[i])
    data['acao_programa'] = acao_programa
    del acao_programa
    data = data.drop(["acao","programa"],axis = 1)
    
    # Codigo que concatena os 3 primeiros codigos de empenho sequencial
    resumo_classificacao = [0] * data.shape[0]
    for i in range(data.shape[0]):
        resumo_classificacao[i] = int(data['empenho_sequencial_empenho'].iloc[i][:13].replace(".", ""))
    data['resumo_classificacao_orcamentaria'] = resumo_classificacao
    del resumo_classificacao
    data = data.drop('empenho_sequencial_empenho',axis =1)
    
    # Codigo que mostra a quantidade de empenhos por processo
    quantidade_empenhos_processo = data['empenho_numero_do_processo'].value_counts()
    quantidade_empenhos_processo = quantidade_empenhos_processo.to_dict()
    empenhos_processo = [0]* data.shape[0]
    for i in range(data.shape[0]):
        empenhos_processo[i] = quantidade_empenhos_processo[data['empenho_numero_do_processo'].iloc[i]]
    data['empenhos_por_processo'] = empenhos_processo
    del empenhos_processo
    data = data.drop('empenho_numero_do_processo',axis = 1)
    del quantidade_empenhos_processo

    # Tratamento dos dados ########################################################
    # Normalizando colunas numéricas
    min_max_scaler = preprocessing.MinMaxScaler()
    colunas = data.columns
    for col in colunas:
        if(data[col].dtype != "O"):
            data[col] = pd.DataFrame(min_max_scaler.fit_transform(data[col].values.reshape(-1,1)))
    # Normalizando colunas numéricas
    # Excluindo as colunas que ja foram tratadas
    data = data.drop(['empenho_historico','natureza_despesa_cod'], axis='columns')
    if(escolha == "sem OHE"):
        return data, label
    elif(escolha == "OHE"):
        # Aplicando a estrategia One Hot Encoding
        data = one_hot_encoding.oneHotEncoding(data)
        return data, label
    else: return None
###########################DADOS TRTADOS######################################
  

  
