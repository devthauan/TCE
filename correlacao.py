from matplotlib import pyplot as plt
from modelos import feature_importance
from preparacaoDados import tratamentoDados
import seaborn as sns
data, label = tratamentoDados("OHE") 
data_feature_importance, colunasMaisImportantes = feature_importance.featureImportance(data,label,30)
resultado = data.loc[:,colunasMaisImportantes]
plt.figure(figsize=(20,15))
corrMatrix = resultado.corr()
image = sns.heatmap(corrMatrix, annot=True,)
plot = image.get_figure()
plot.savefig("output.png",bbox_inches='tight')
