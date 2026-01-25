from sklearn.cluster import KMeans

def make_kmeans(n_clusters: int, random_state: int = 42):
    return KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
