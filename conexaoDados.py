from tratamentos import pickles
def dados():
    data = pickles.carregaPickle("df")
    return data[800:1200]