from preparacaoDados import tratamentoDados
from modelos import feature_importance
from matplotlib import pyplot as plt
import seaborn as sns

data, label = tratamentoDados("OHE")
print(data.shape)
# Pega os 15 atributos mais relevantes
resultado, colunasMaisImportantes = feature_importance.featureImportance(data,label,15,1)
resultado = data[colunasMaisImportantes]
print(resultado.shape)
plt.figure(figsize=(20,15))
corrMatrix = resultado.corr()
image = sns.heatmap(corrMatrix, annot=True,)
plot = image.get_figure()
plot.savefig("correlacao.png",bbox_inches='tight')


# =============================================================================
# Distribuicao do rotulo
# =============================================================================
plt.figure(figsize=(20,15))
plt.grid()
plt.title("Distribuição do Rótulo",fontsize= 18)
plt.xlabel("Naturezas de despesa",fontsize= 16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.yticks([0,100,1000,2000,3000,4000,5000,6000])
plt.ylabel("Quantidade de empenhos",fontsize= 16)
plt.bar(label['natureza_despesa_cod'].value_counts(ascending = True).index,label['natureza_despesa_cod'].value_counts(ascending = True).values)
plt.xticks([0,label['natureza_despesa_cod'].value_counts().count()-1],fontsize= 16)
plt.text(0,1,str(label['natureza_despesa_cod'].value_counts(ascending = True).values[0]),fontsize= 16, color='blue')
plt.text(label['natureza_despesa_cod'].value_counts().count()-10,label['natureza_despesa_cod'].value_counts(ascending = True).values[-1]+15,str(label['natureza_despesa_cod'].value_counts(ascending = True).values[-1]),fontsize= 16, color='blue')
plt.show()

contador = 0
for i in range(len(label['natureza_despesa_cod'].value_counts().values)):
    if(label['natureza_despesa_cod'].value_counts().values[i]<100):
        contador +=1
print("Existem ",contador," naturezas com menos de 1000 empenhos, equivalente a ",contador/label['natureza_despesa_cod'].value_counts().count()," dos dados.")