from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_all_subplots(df_sum: pd.DataFrame, out_path: Path):
    metrics = ["silhouette", "ari", "nmi", "time_total_sec"]
    datasets = sorted(df_sum["dataset"].unique().tolist())
    dimreds = sorted(df_sum["dimred"].unique().tolist())

    nrows = len(metrics)
    ncols = len(datasets)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 3.4 * nrows),
        sharex=False,
        sharey=False
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for r, metric in enumerate(metrics):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            for dimred in dimreds:
                sub = df_sum[(df_sum["dataset"] == dataset) & (df_sum["dimred"] == dimred)].sort_values("d")
                if sub.empty:
                    continue
                ax.plot(sub["d"], sub[f"{metric}_mean"], marker="o", label=dimred.upper())
            ax.set_title(f"{dataset} | {metric}")
            ax.set_xlabel("dimension")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend()

    fig.suptitle("PCA and UMAP on KMeans algorithms: Metrics and Reduced Dimension d", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_cluster_scatter_2datasets(
    out_path: Path,
    load_dataset_fn,
    make_pca_fn,
    make_umap_fn,
    make_kmeans_fn,
    datasets=("mnist", "fashion_mnist"),
    n_samples_map=None,
    n_clusters_map=None,
    seed: int = 42,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    hog_orientations: int = 9,
    hog_pixels_per_cell: int = 4,
    hog_cells_per_block: int = 2,
):
    """
    X đã là ma trận đặc trưng (HOG) => GT được vẽ bằng cách tô màu y_true trên PCA2 để trực quan.
    """
    if n_samples_map is None:
        n_samples_map = {"mnist": 4000, "fashion_mnist": 4000}
    if n_clusters_map is None:
        n_clusters_map = {"mnist": 10, "fashion_mnist": 10}

    cols = ["GT", "PCA+KMeans", "UMAP+KMeans"]
    nrows = len(datasets)
    ncols = len(cols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.2 * nrows))

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for r, ds in enumerate(datasets):
        n_samples = int(n_samples_map.get(ds, 4000))
        n_clusters = int(n_clusters_map.get(ds, 10))

        X, y_true = load_dataset_fn(
            dataset=ds,
            n_samples=n_samples,
            n_clusters=n_clusters,
            seed=seed,
            hog_orientations=hog_orientations,
            hog_pixels_per_cell=hog_pixels_per_cell,
            hog_cells_per_block=hog_cells_per_block,
        )

        pca2 = make_pca_fn(2, seed).fit_transform(X)
        umap2 = make_umap_fn(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=None,
            n_jobs=-1
        ).fit_transform(X)

        y_pred_pca = make_kmeans_fn(n_clusters, seed).fit_predict(pca2)
        y_pred_umap = make_kmeans_fn(n_clusters, seed).fit_predict(umap2)

        ax = axes[r][0]
        ax.scatter(pca2[:, 0], pca2[:, 1], s=6, c=y_true, alpha=0.85)
        ax.set_title(f"{ds} | GT (PCA2)")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        ax = axes[r][1]
        ax.scatter(pca2[:, 0], pca2[:, 1], s=6, c=y_pred_pca, alpha=0.85)
        ax.set_title(f"{ds} | PCA -> KMeans (k={n_clusters})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        ax = axes[r][2]
        ax.scatter(umap2[:, 0], umap2[:, 1], s=6, c=y_pred_umap, alpha=0.85)
        ax.set_title(f"{ds} | UMAP -> KMeans (k={n_clusters})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.15)

    fig.suptitle("Ground Truth (PCA2) vs PCA/UMAP + KMeans (2D scatter)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=250)
    plt.close(fig)