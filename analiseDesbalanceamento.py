#naturezas = [323,405,409,411,807,808,1706,3012,3013,3015,3016,3020,3021,3022,3030,3031,3036,3039,
#             3041,3042,3045,3046,3048,3050,3052,3053,3054,3055,3056,3058,3061,3203,3209,3210,3505,
#             3507,3514,3601,3603,3606,3607,3608,3612,3615,3620,3629,3637,3639,3646,3706,3927,3945,
#             3954,3958,3961,3967,3970,3976,4090,4115,4122,4125,4701,5111,5112,5201,5203,5204,5206,
#             5208,5212,5213,5214,5217,5218,5219,5220,5221,5223,5225,5234,5235,5241,5242,6109,9206,
#             9208,9232,9233,9234,9240,9241,9242,9247,9251,9259,9262,9272,9281,9290,9312]
#
#
#f = open('asd.xlsx','a+')
#f.write("Natureza"+" ; "+"Porcentagem"+'\n')
#for natureza in naturezas:
#    if(len(str(natureza))==3):
#        	result = (label[0].where(label[0]==int(natureza)).dropna().value_counts().values[0]/data['elemento_despesa'].where(data['elemento_despesa'].str.contains("0"+str(natureza)[:1])).dropna().value_counts().values[0])*100
#    else:
#        	result = (label[0].where(label[0]==int(natureza)).dropna().value_counts().values[0]/data['elemento_despesa'].where(data['elemento_despesa'].str.contains(str(natureza)[:2])).dropna().value_counts().values[0])*100
#    print(natureza,"  ", "%.2f" % round(result, 2),"%")
#    f.write(str(natureza)+" ; "+ "%.2f" % round(result, 2)+"%"+"\n")
#f.flush()
#f.close()    
   
#teste
#natureza = 3031
#label[0].where(label[0]==int(natureza)).dropna().value_counts().values[0]
#data['elemento_despesa'].where(data['elemento_despesa'].str.contains(str(natureza)[:2])).dropna().value_counts().values[0]

#
#(label[0].where(label[0]==int(natureza)).dropna().value_counts().values[0]/data['elemento_despesa'].where(data['elemento_despesa'].str.contains(str(natureza)[:2])).dropna().value_counts().values[0])*100
#
#data['empenho_historico'].value_counts()



from matplotlib import pyplot as plt
from preparacaoDados import tratamentoDados
from scipy.stats import pearsonr
import pandas as pd
import math
import numpy as np

data, label = tratamentoDados("OHE") 

#cria o vetor de correlaçao com a label
vetor_correlacoes = []
for col in data.columns:
    corr, _ = pearsonr(data[col], label[0])
    if not(math.isnan(corr)):
        vetor_correlacoes.append([corr,col])

#ordena pelo valor absoluto
sorted_vetor_correlacoes = pd.DataFrame(sorted([abs(x[0]), x[0], x[1]] for x in vetor_correlacoes))
sorted_vetor_correlacoes = sorted_vetor_correlacoes.drop(0,axis =1)
#pega as 30 mais e menos importantes
trinta_importantes = sorted_vetor_correlacoes[-30:]
trinta_menos_importantes = sorted_vetor_correlacoes[:30]

#plot dos 30 mais importantes 
vetor_importante_1=sorted(trinta_importantes[1])
vetor_importante_2=sorted(trinta_importantes[2])
plt.figure(figsize=(10,7))
plt.bar(vetor_importante_2,vetor_importante_1)
plt.xticks([])
plt.text(0,vetor_importante_1[0]+(-0.05) , str(vetor_importante_2[0]), color='black', fontweight='bold')
plt.text(len(vetor_importante_1),vetor_importante_1[-1] , str(vetor_importante_2[-1]), color='black', fontweight='bold')
plt.yticks(np.arange(-1,1.1,0.1))
plt.title("Correlação Atributo-Rótulo")
plt.xlabel("Atributos")
#plt.savefig("importante.png",bbox_inches='tight')
plt.show()

#plot dos 30 menos importantes 
vetor_menos_importante_1=sorted(trinta_menos_importantes[1])
vetor_menos_importante_2=sorted(trinta_menos_importantes[2])
plt.figure(figsize=(10,7))
plt.bar(vetor_menos_importante_2,vetor_menos_importante_1)
plt.xticks([])
plt.text(0,vetor_menos_importante_1[0]+(-0.05) , str(vetor_menos_importante_2[0]), color='black', fontweight='bold')
plt.text(len(vetor_menos_importante_1),vetor_menos_importante_1[-1] , str(vetor_menos_importante_2[-1]), color='black', fontweight='bold')
plt.yticks(np.arange(-1,1.1,0.1))
#plt.savefig("nao_importante.png",bbox_inches='tight')
plt.show()

contador=0
vetor_correlacoes2 = abs(pd.DataFrame(vetor_correlacoes).drop(1,axis=1))
for i in range(len(vetor_correlacoes2)):
    if(vetor_correlacoes2.iloc[i][0]>0.4):
        contador+=1
print("Existem ",contador," das ",len(vetor_correlacoes2)," Features com correlacao acima de 40%")