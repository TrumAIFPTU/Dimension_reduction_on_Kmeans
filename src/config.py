from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class SweepConfig:
    dataset: str              # "mnist" | "fashion_mnist"
    dimred: str               # "pca" | "umap"
    n_samples: int
    dims: List[int]
    n_clusters: int
    seeds: List[int]

    # Image params
    kernel_mode: str="sharp"
    hog_orientations: int=9
    hog_pixels_per_cell: int=4
    hog_cells_per_block: int=2

    # UMAP params
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"


def get_default_configs():
    seeds = [42, 52, 62]
    dims_high = [2, 5, 10, 20, 50]

    cfgs = [
        #MNIST
        SweepConfig(dataset="mnist", dimred="pca",  n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),
        SweepConfig(dataset="mnist", dimred="umap", n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),
        #Fashion-MNIST
        SweepConfig(dataset="fashion_mnist", dimred="pca",  n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),
        SweepConfig(dataset="fashion_mnist", dimred="umap", n_samples=10000, dims=dims_high, n_clusters=10, seeds=seeds),
    ]
    return cfgs
