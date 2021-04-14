import pandas as pd
from tratamentos import pickles


dados_dia_18_a_28 = pickles.carregaPickle("dados_test")
label_dia_18_a_28 = pd.DataFrame(dados_dia_18_a_28['Natureza Despesa (Cod)'])
del dados_dia_18_a_28
# Rotulos individuais dessa data
label_dia_18_a_28 = list(label_dia_18_a_28['Natureza Despesa (Cod)'].value_counts().index)
# =============================================================================
label_nova = pickles.carregaPickle("label")
label_nova = list(label_nova['natureza_despesa_cod'].value_counts().index)
classes = []# classes dos dias 18 a 28 nao presentes nos dados
for i in range(len(label_dia_18_a_28)):
    if(label_dia_18_a_28[i] not in label_nova):
        classes.append(i)
classes = pd.DataFrame(label_dia_18_a_28).iloc[classes]
# =============================================================================
label_nova_ntratada =  pickles.carregaPickle("df")
label_nova_ntratada = list(label_nova_ntratada['Natureza Despesa (Cod)'].value_counts().index)
classes_ntratadas = []# classes dos dias 18 a 28 nao presentes nos dados
for i in range(len(label_dia_18_a_28)):
    if(label_dia_18_a_28[i] not in label_nova_ntratada):
        classes_ntratadas.append(i)
classes_ntratadas = pd.DataFrame(label_dia_18_a_28).iloc[classes_ntratadas]
# =============================================================================
label_antiga = pd.read_csv("arquivos/dadosTCE.csv")
label_antiga = list(label_antiga['Natureza Despesa (Cod)(EOF)'].value_counts().index)
classes_antigas = []# classes dos dias 18 a 28 nao presentes nos dados
for i in range(len(label_dia_18_a_28)):
    if(label_dia_18_a_28[i] not in label_antiga):
        classes_antigas.append(i)
classes_antigas = pd.DataFrame(label_dia_18_a_28).iloc[classes_antigas]
# =============================================================================
# Isso mostra que existem novas classes
# =============================================================================

    