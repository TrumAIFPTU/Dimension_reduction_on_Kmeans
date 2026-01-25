from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class SweepConfig:
    dataset: str              # "blobs_linear" | "mnist" | "fashion_mnist" | "swiss_roll"
    dimred: str               # "pca" | "umap"
    n_samples: int
    dims: List[int]
    n_clusters: int
    seeds: List[int]

    # UMAP params
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # blobs params
    blobs_n_features: int = 50
    blobs_cluster_std: float = 1.0

def get_default_configs():
    seeds = [42, 52, 62]

    # Dims cho dataset nhiều chiều (MNIST/Fashion/Blobs)
    dims_high = [2, 5, 10, 20, 50]

    # Swiss roll gốc chỉ có 3 feature -> PCA tối đa 3 (UMAP cũng nên để 2/3 để hợp lý)
    dims_swiss = [2, 3]

    cfgs = [
        # 1) Tuyến tính mạnh: blobs_linear (m=50)
        SweepConfig(dataset="blobs_linear", dimred="pca",  n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),
        SweepConfig(dataset="blobs_linear", dimred="umap", n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),

        # 2) Phi tuyến nhẹ: MNIST
        SweepConfig(dataset="mnist", dimred="pca",  n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),
        SweepConfig(dataset="mnist", dimred="umap", n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),

        # 3) Phi tuyến vừa: Fashion-MNIST
        SweepConfig(dataset="fashion_mnist", dimred="pca",  n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),
        SweepConfig(dataset="fashion_mnist", dimred="umap", n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),

        # 4) Phi tuyến mạnh: Swiss roll
        SweepConfig(dataset="swiss_roll", dimred="pca",  n_samples=8000, dims=dims_swiss, n_clusters=10, seeds=seeds),
        SweepConfig(dataset="swiss_roll", dimred="umap", n_samples=8000, dims=dims_swiss, n_clusters=10, seeds=seeds),
    ]
    return cfgs
