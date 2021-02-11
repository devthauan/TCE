from preparacaoDados import tratamentoDados
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tratamentos import tratar_texto
from wordcloud import WordCloud

data, label = tratamentoDados("OHE")
textoTratado = tratamentoDados("texto")
data.reset_index(drop=True, inplace=True)

for natureza in data['natureza_despesa_nome'].value_counts().index[43:]:
    wordcloud = data['natureza_despesa_nome'].where(data['natureza_despesa_nome'] == natureza)
    wordcloud.dropna(inplace=True)
    naturezas = pd.DataFrame(textoTratado).iloc[wordcloud.index]
    
    wc_string = naturezas[0].str.cat(sep=' ')
    wc = WordCloud(width=1600, height=800,background_color="white", max_words=2000).generate(wc_string)
    plt.ioff()
    fig = plt.figure(figsize=(16,10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title('Natureza de despesa:'+natureza, fontsize=18)
    plt.savefig("wordclouds/"+tratar_texto.remove_punctuation(natureza)+".png",dpi=300)
    plt.close(fig)
    plt.show()

