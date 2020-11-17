### Bibliotecas python ###
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
### Meus pacotes ###
from tratamentos import tratar_label
from tratamentos import tratar_texto
from tratamentos import one_hot_encoding
### Meus pacotes ###

def tratamentoDados(escolha):
    # Nome do arquivo csv a ler
    nomeArquivoDadosBrutos = 'arquivos/dadosTCE.csv';
    # Carrega os dados na variavel 'data' utilizando o Pandas
    data = pd.read_csv(nomeArquivoDadosBrutos, encoding = "utf-8",low_memory = False)
    del nomeArquivoDadosBrutos
    # Trata o nome das colunas para trabalhar melhor com os dados
    data.columns = [c.lower().replace(' ', '_') for c in data.columns]
    data.columns = [tratar_texto.removerCaracteresEspeciais(c)for c in data.columns]
    data.columns = [tratar_texto.tratarnomecolunas(c)for c in data.columns]
    # Deletando empenhos sem relevancia devido ao saldo zerado
    index = data["valor_saldo_do_empenho"].where(data["valor_saldo_do_empenho"] == 0).dropna().index
    data.drop(index,inplace = True)
    data.reset_index(drop=True, inplace=True)
#    data = data[:5000] #limitando os dados para fazer testes
    # Deleta colunas que atraves de analise foram identificadas como nao uteis
    data = data.drop(['empenho_sequencial_empenho.1','classificacao_orcamentaria_descricao',
                      'natureza_despesa_nome','valor_estorno_anulacao_empenho',
                      'valor_anulacao_cancelamento_empenho','fonte_recurso_cod',
                      'elemento_despesa','grupo_despesa'], axis='columns')
    # Funcao que gera o rotulo e retorna as linhas com as naturezas de despesa que so aparecem em 1 empenho
    label,linhas_label_unica = tratar_label.tratarLabel(data)
    label = pd.DataFrame(label)
    # Excluindo as naturezas de despesas que so tem 1 empenho
    data = data.drop(linhas_label_unica)
    data.reset_index(drop=True, inplace=True)
    del linhas_label_unica
    # Excluindo empenhos irrelevantes devido nao estarem mais em vigencia
    sem_relevancia = pd.read_excel("analise/Naturezas de despesa com vigência encerrada.xlsx")
    sem_relevancia = sem_relevancia['Nat. Despesa']
    sem_relevancia = pd.DataFrame(sem_relevancia)
    excluir = []
    for i in range(len(sem_relevancia['Nat. Despesa'])):
        excluir.append( label.where( label['natureza_despesa_cod'] == sem_relevancia['Nat. Despesa'].iloc[i] ).dropna().index )
    excluir = [item for sublist in excluir for item in sublist]
    # Excluindo as naturezas que nao estao mais vigentes
    label.drop(excluir,inplace =True)
    label.reset_index(drop=True, inplace=True)
    data.drop(excluir,inplace = True)
    data.reset_index(drop=True, inplace=True)
    del excluir, sem_relevancia
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
# =============================================================================
#     Tratamento dos dados
# =============================================================================
    # Codigo que gera o meta atributo "pessoa_juridica" onde 1 representa que a pessoa e juridica e 0 caso seja fisica
    identificacao_pessoa = [0] * data.shape[0]
    for i in range(data.shape[0]):
      if(data['beneficiario_cpf'].iloc[i] == "-"):
        identificacao_pessoa[i] = 1
      else: identificacao_pessoa[i]=0
    data['pessoa_juridica'] = identificacao_pessoa
    del identificacao_pessoa
    data['pessoa_juridica'] = data['pessoa_juridica'].astype("int8")
#    data = data.drop(["beneficiario_cpf/cnpj","beneficiario_cpf","beneficiario_cnpj","beneficiario_nome"], axis='columns')
    data = data.drop(["beneficiario_cpf","beneficiario_cnpj","beneficiario_nome"], axis='columns')
    
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
    
    # Codigo que retira o codigo de acao (retirando 77 valores)
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
    
#    # Codigo que concatena os 3 primeiros codigos de empenho sequencial
#    resumo_classificacao = [0] * data.shape[0]
#    for i in range(data.shape[0]):
#        resumo_classificacao[i] = int(data['empenho_sequencial_empenho'].iloc[i][:13].replace(".", ""))
#    data['resumo_classificacao_orcamentaria'] = resumo_classificacao
#    del resumo_classificacao
    data = data.drop('empenho_sequencial_empenho',axis =1)
    
    # Codigo que mostra a quantidade de empenhos por processo
    quantidade_empenhos_processo = data['empenho_numero_do_processo'].value_counts()
    quantidade_empenhos_processo = quantidade_empenhos_processo.to_dict()
    empenhos_processo = [0]* data.shape[0]
    for i in range(data.shape[0]):
        empenhos_processo[i] = quantidade_empenhos_processo[data['empenho_numero_do_processo'].iloc[i]]
    data['empenhos_por_processo'] = empenhos_processo
    del empenhos_processo
    del quantidade_empenhos_processo
    data = data.drop('empenho_numero_do_processo',axis = 1)
# =============================================================================
#     Tratamento dos dados
# =============================================================================
    # Normalizando colunas numéricas
    min_max_scaler = preprocessing.MinMaxScaler()
    colunas = data.columns
    for col in colunas:
        if(data[col].dtype != "O"):
            data[col] = pd.DataFrame(min_max_scaler.fit_transform(data[col].values.reshape(-1,1)))
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
#  
#
##CPF 191.0 tem mais de um nome
## Código que mostra a inconsistência dos cpfs cnpjs
#data['empenho_sequencial_empenho'].where(data['beneficiario_cpf/cnpj']==191.0).dropna()
#data['beneficiario_nome'].where(data['beneficiario_cpf']=="191").dropna().value_counts()
#data['beneficiario_nome'].where(data['beneficiario_cnpj']=="00000000000191").dropna().value_counts()
##CARLOS PEREIRA DE SOUZA (cpf)    2 e BANCO DO BRASIL SA (cnpj)   200
#
#
#data['beneficiario_cpf/cnpj']
#
#
#
#
#
#tamanhos2 = []
#for i in range(len(data['beneficiario_cpf/cnpj'].value_counts())):
#    tamanhos2.append(len(str(data['beneficiario_cpf/cnpj'].value_counts().index[i])))
#    
#tamanhoscpf = []
#for i in range(len(data['beneficiario_cpf'].value_counts())):
#    tamanhoscpf.append(len(str(data['beneficiario_cpf'].value_counts().index[i])))
#    
#tamanhoscnpj = []
#for i in range(len(data['beneficiario_cnpj'].value_counts())):
#    tamanhoscnpj.append(len(str(data['beneficiario_cnpj'].value_counts().index[i])))
#
#
#pd.DataFrame(tamanhoscpf).value_counts()
#pd.DataFrame(tamanhoscnpj).value_counts()
#pd.DataFrame(tamanhos2).value_counts()
#
#data['beneficiario_cnpj'].value_counts()
#data['beneficiario_cpf'].value_counts()
#data['beneficiario_cpf/cnpj'].value_counts()
#data['beneficiario_nome'].where(data['beneficiario_cpf']=="1181045193").dropna().value_counts()
