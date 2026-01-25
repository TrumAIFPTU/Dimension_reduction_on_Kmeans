from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class SweepConfig:
    dataset: str              # "mnist" | "swiss_roll"
    dimred: str               # "pca" | "umap"
    n_samples: int
    dims: List[int]           # list chiều sau giảm
    n_clusters: int
    seeds: List[int]

    # UMAP params
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
 
def get_default_configs():
    mnist_dims_pca  = [20, 10, 5, 2]
    mnist_dims_umap = [20, 10, 5, 2] 

    swiss_dims = [3, 2]

    seeds = [42, 52, 62]  # 3 seeds (bạn tăng lên 5-10 nếu muốn)

    cfgs = [
        SweepConfig(dataset="mnist", dimred="pca",  n_samples=10000, dims=mnist_dims_pca,  n_clusters=10, seeds=seeds),
        SweepConfig(dataset="mnist", dimred="umap", n_samples=5000,  dims=mnist_dims_umap, n_clusters=10, seeds=seeds),

        SweepConfig(dataset="swiss_roll", dimred="pca",  n_samples=8000, dims=swiss_dims, n_clusters=10, seeds=seeds),
        SweepConfig(dataset="swiss_roll", dimred="umap", n_samples=8000, dims=swiss_dims,   n_clusters=10, seeds=seeds),
    ]
    return cfgs
