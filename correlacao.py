from matplotlib import pyplot as plt
from modelos import feature_importance
from preparacaoDados import tratamentoDados
import seaborn as sns
data, label = tratamentoDados("OHE") 
resultado, colunasMaisImportantes = feature_importance.featureImportance(data,label,1,0.647110)
plt.figure(figsize=(20,15))
corrMatrix = resultado.corr()
image = sns.heatmap(corrMatrix, annot=True,)
plot = image.get_figure()
plot.savefig("correlacao.png",bbox_inches='tight')
