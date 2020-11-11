import pandas as pd
# Calcula a quantidade de naturezas que tem mais de 80% de empenhos alocados dentro de um cluster
def porcentagem_correta_cluster(label,labels_clusters):
    # Junta a label original com a predita pelo algoritmo
    label.reset_index(drop=True, inplace=True)
    labels_clusters.reset_index(drop=True, inplace=True)
    clusters = pd.concat([label,labels_clusters],axis=1)
    clusters.columns = ["label_original","labels_clusters"]
    # Vai em todas as labels e conta a porcentagens das labels originais que estão la dentro
    porcentagem_labels = []
    for lab in clusters['labels_clusters'].value_counts().index:
        #todas as labels originais colocadas dento deste cluster
        labels = clusters['label_original'].where(clusters['labels_clusters']==lab).dropna().value_counts()
        quantidade_labels_acima_80 = 0
        '''Para cada label original dentro do cluster verifica se a quantidade que esta no cluster
        é acima de 80% do total dessa classe'''
        for i in range(len(labels)):
            if((labels.values[i]/ clusters['label_original'].where(clusters['label_original'] == labels.index[i]).dropna().value_counts().values[0])*100 > 80 ):
                quantidade_labels_acima_80 += 1
        #adiciona para cada cluster o numero de clases cima de 80% presentes nele
        porcentagem_labels.append(quantidade_labels_acima_80)
    #retorna a quantidade de classes acima de 80% dos dados presentes em 1 cluster
    return sum(porcentagem_labels)

def analisecluster(y_test, y_predito):
    acerto = porcentagem_correta_cluster(pd.DataFrame(y_test),pd.DataFrame(y_predito))
    print("Esse método gerou ",acerto," de ",pd.DataFrame(y_test)[0].value_counts().count()," labels acima de 80% que é "+ str(acerto/pd.DataFrame(y_test)[0].value_counts().count())+"% de acerto." )
