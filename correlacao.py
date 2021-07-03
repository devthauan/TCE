from preparacaoDados import tratamentoDados
from modelos import feature_importance
from matplotlib import pyplot as plt
from tratamentos import pickles
import seaborn as sns
import numpy as np
from pdf2image import convert_from_path

#tratamentoDados('OHE')
data = pickles.carregaPickle("data")[:10000]
label = pickles.carregaPickle("label")[:10000]
print(data.shape)
# Pega os 10 atributos mais relevantes
resultado, colunasMaisImportantes = feature_importance.featureImportance(data,label,10,0)
print("tamanho", len(colunasMaisImportantes))
resultado = data.iloc[:,colunasMaisImportantes]
print(resultado.shape)
plt.figure(figsize=(26,18))
matrix = np.triu(resultado.corr())
sns.set(font_scale=2.5)
image = sns.heatmap(resultado.corr(), annot=True, mask=matrix,annot_kws={"size": 28})
plot = image.get_figure()
plot.savefig("correlacao.pdf",bbox_inches='tight',transparent=True)
images = convert_from_path(
        "correlacao.pdf",
        output_folder="/home/devthauan/Documentos/ProjetoTCE",
        grayscale=True,
        fmt="jpeg",
        thread_count=4
    )

# =============================================================================
# Distribuicao do rotulo
# =============================================================================
 plt.figure(figsize=(20,15))
 plt.grid()
 plt.xlabel("Naturezas de despesa",fontsize= 30)
 plt.tick_params(axis='both', which='major', labelsize=30)
 plt.yticks([0,100,200,300,4000,5000,6000])
 plt.ylabel("Quantidade de empenhos",fontsize= 30)
 plt.plot(label['natureza_despesa_cod'].value_counts(ascending = True).index,label['natureza_despesa_cod'].value_counts(ascending = True).values,linewidth=10,color='black')
 plt.xticks([0,label['natureza_despesa_cod'].value_counts().count()-1],fontsize= 25)
 plt.text(0,200,str(label['natureza_despesa_cod'].value_counts(ascending = True).values[0]),fontsize= 25, color='black')
 plt.text(label['natureza_despesa_cod'].value_counts().count()-20,label['natureza_despesa_cod'].value_counts(ascending = True).values[-1]+75,str(label['natureza_despesa_cod'].value_counts(ascending = True).values[-1]),fontsize= 25, color='black')
 plt.savefig("distribuicao.png",bbox_inches='tight')
 plt.show()


contador = 0
for i in range(len(label['natureza_despesa_cod'].value_counts().values)):
    if(label['natureza_despesa_cod'].value_counts().values[i]<100):
        contador +=1
print("Existem ",contador," naturezas das "+str(label['natureza_despesa_cod'].value_counts().count())+" com menos de 1000 empenhos, equivalente a ",contador/label['natureza_despesa_cod'].value_counts().count()," dos dados.")

