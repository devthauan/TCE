import pandas as pd
# Separa os codigos em coluna dividindo-os pelo ponto e retorna um dataframe 
#  com essas colunas e os nomes passados por parametro
def separaColunaPorPonto(data, col, col_label):
    vector = []
    for i in range(data[col].shape[0]):
        vector.append(data[col].iloc[i].split("."))
    colunaSeparada = pd.DataFrame(data=vector , columns=col_label)
    return colunaSeparada