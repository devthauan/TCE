import pandas as pd
def tratarLabel(data):
    # Juntando os dois ultimos atributos da natureza de despesa pois precisam estar juntos
    result = [0]* data.shape[0]
    for i in range(data.shape[0]):
        result[i] = (data['natureza_despesa_cod'][i].split('.')[3] + data['natureza_despesa_cod'][i].split('.')[4])
    
    label = pd.DataFrame(result).astype("int32")
    #pegando as labels e a quantidade de documentos que elas aparecem
    quantidade_labels = (pd.DataFrame(label[0].value_counts()))
    #pegando o nome das labels só aparecem uma vez
    unico_valor_label = []
    for i in range(quantidade_labels.shape[0]):
      if(quantidade_labels.iloc[i].values == 1):
        unico_valor_label.append(quantidade_labels.iloc[i].name)
        
    #pegando as linhas das classes com só 1 documento
    rows_unica_label = []
    for i in range(label.shape[0]):
      if((label.iloc[i].values == unico_valor_label).any()):
        rows_unica_label.append(i)
    
    label = label.drop(label.index[rows_unica_label])
    return label, rows_unica_label