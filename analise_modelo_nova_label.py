# -*- coding: utf-8 -*-
import pandas as pd
# carregando os dados
correto = pd.read_excel("resultados/corretos.xlsx")
incorreto = pd.read_excel("resultados/incorretos.xlsx")
inconclusivo = pd.read_excel("resultados/inconclusivos.xlsx")
# calculando a quantidade total de empenhos por classe
tudo = pd.concat([correto,incorreto],axis = 0)
tudo = pd.concat([tudo,inconclusivo],axis = 0)
quantidade_classes_totais = dict(tudo["natureza_despesa_cod"].value_counts())
# =============================================================================
# calculando a porcentagem de corretos por classe
# =============================================================================
corretos = dict(correto["natureza_despesa_cod"].value_counts())
keys = list(corretos.keys())
porcentagem_corretos = {}
for i in range(len(keys)):
    porcentagem_corretos[keys[i]] = [ corretos[keys[i]] / quantidade_classes_totais[keys[i]], quantidade_classes_totais[keys[i]] ]
porcentagem_corretos = pd.DataFrame.from_dict(porcentagem_corretos, orient = "index").reset_index()
porcentagem_corretos.to_excel("porcentagem_corretos.xlsx")
# =============================================================================
# calculando a porcentagem de incorretos por classe
# =============================================================================
incorretos = dict(incorreto["natureza_despesa_cod"].value_counts())
keys = list(incorretos.keys())
porcentagem_incorretos = {}
for i in range(len(keys)):
    porcentagem_incorretos[keys[i]] = [incorretos[keys[i]] / quantidade_classes_totais[keys[i]], quantidade_classes_totais[keys[i]] ]
porcentagem_incorretos = pd.DataFrame.from_dict(porcentagem_incorretos, orient = "index").reset_index()
porcentagem_incorretos.to_excel("porcentagem_incorretos.xlsx")
# =============================================================================
# calculando a porcentagem de inconclusivos por classe
# =============================================================================
inconclusivos = dict(inconclusivo["natureza_despesa_cod"].value_counts())
keys = list(inconclusivos.keys())
porcentagem_inconclusivos= {}
for i in range(len(keys)):
    porcentagem_inconclusivos[keys[i]] = [inconclusivos[keys[i]] / quantidade_classes_totais[keys[i]],quantidade_classes_totais[keys[i]] ]
porcentagem_inconclusivos = pd.DataFrame.from_dict(porcentagem_inconclusivos, orient = "index").reset_index()
porcentagem_inconclusivos.to_excel("porcentagem_inconclusivos.xlsx")
#
#correto["natureza_despesa_cod"].value_counts()
#incorreto["natureza_despesa_cod"].value_counts()
#inconclusivo["natureza_despesa_cod"].value_counts()