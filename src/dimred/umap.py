import umap

def make_umap(n_components, n_neighbors=15, min_dist=0.1, metric="euclidean",
              random_state=None, n_jobs=-1):
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,  # None => cho phép parallel
        n_jobs=n_jobs
    )
