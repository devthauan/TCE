import pandas as pd
def tratarLabel(data):
    
    label = data['natureza_despesa_cod']
    #pegando as labels e a quantidade de documentos que elas aparecem
    quantidade_labels = (pd.DataFrame(label.value_counts()))
    #pegando o nome das labels só aparecem uma vez
    unico_valor_label = []
    for i in range(quantidade_labels.shape[0]):
      if(quantidade_labels.iloc[i].values == 1):
        unico_valor_label.append(quantidade_labels.iloc[i].name)
        
    #pegando as linhas das classes com só 1 documento
    rows_unica_label = []
    for i in range(label.shape[0]):
      if(label.iloc[i] in unico_valor_label):
        rows_unica_label.append(i)
    
    label = label.drop(label.index[rows_unica_label])
    return label, rows_unica_label