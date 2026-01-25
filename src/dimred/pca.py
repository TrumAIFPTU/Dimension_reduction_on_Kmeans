from sklearn.decomposition import PCA

def make_pca(n_components: int, random_state: int = 42):
    return PCA(n_components=n_components, random_state=random_state)
