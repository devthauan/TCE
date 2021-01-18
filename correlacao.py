from preparacaoDados import tratamentoDados
from modelos import feature_importance
from matplotlib import pyplot as plt
import seaborn as sns

data, label = tratamentoDados("OHE") 
print(data.shape)
resultado, colunasMaisImportantes = feature_importance.featureImportance(data,label,1,0.99815)
print(resultado.shape)
plt.figure(figsize=(20,15))
corrMatrix = resultado.corr()
image = sns.heatmap(corrMatrix, annot=True,)
plot = image.get_figure()
plot.savefig("correlacao.png",bbox_inches='tight')
