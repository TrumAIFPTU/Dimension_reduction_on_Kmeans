import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def load_blobs_linear(
    n_samples: int,
    n_features: int,
    n_clusters: int,
    random_state: int = 42,
    cluster_std: float = 1.0
):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, y.astype(int)
