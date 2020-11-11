from sklearn.decomposition import TruncatedSVD
def svd(data,numero_componentes):
    svd_model = TruncatedSVD(n_components=numero_componentes, n_iter=2, random_state=42)
    data = svd_model.fit_transform(data)
    return data