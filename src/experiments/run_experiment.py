from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.utils.seed import set_seed
from initialize.data import load_mnist_images
from initialize.data import load_fashion_mnist_images
from initialize.dimred import make_pca
from initialize.dimred import make_umap
from initialize.kmeans import make_kmeans

from src.image.preprocess import preprocess_and_sharpen
from src.image.features import extract_feature_matrix

from src.evaluation.metrics import compute_metrics
from src.evaluation.timer import timer


def _load_dataset(
    dataset: str,
    n_samples: int,
    n_clusters: int,
    seed: int,
    kernel_mode: str = "sharp",#fixed
    # image feature params
    hog_orientations: int = 9,
    hog_pixels_per_cell: int = 4,
    hog_cells_per_block: int = 2,
):
    if dataset == "mnist":
        X_img, y = load_mnist_images(n_samples, seed)
    elif dataset == "fashion_mnist":
        X_img, y = load_fashion_mnist_images(n_samples, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if kernel_mode.lower() != "sharp": #Optional decision for selection
        raise ValueError(
            "Hiện pipeline ảnh trong project này cố định kernel_mode='sharp', nếu muốn smooth thì tự viết ma trận<3 "
        )

    X01 = preprocess_and_sharpen(X_img)
    X = extract_feature_matrix(
        X01,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        hog_cells_per_block=hog_cells_per_block,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X, y.astype(int)


def _make_dimred(dimred: str, d: int, seed: int,
                 umap_n_neighbors: int, umap_min_dist: float, umap_metric: str):
    if dimred == "pca":
        return make_pca(d, seed)

    if dimred == "umap":
        return make_umap(
            n_components=d,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=None,  #Run parralell for avoiding seed warning
            n_jobs=-1
        )

    raise ValueError(f"Unknown dimred: {dimred}")

"""
def _plot_metric_vs_d(df_sum: pd.DataFrame, dataset: str, dimred: str, metric: str, out_path: Path):
    sub = df_sum[(df_sum["dataset"] == dataset) & (df_sum["dimred"] == dimred)].sort_values("d")
    plt.figure()
    plt.plot(sub["d"], sub[f"{metric}_mean"], marker="o")
    plt.xlabel("Reduced dimension d")
    plt.ylabel(metric)
    plt.title(f"{metric}: {dataset.upper()} and {dimred.upper()} on each reduced dimension d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    #Nếu vẫn muốn xuất từng plot riêng thì bỏ comment đoạn này
"""
# ================================================================================================================


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

    # normalize axes to 2D array
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
    datasets=("mnist", "fashion_mnist"),
    n_samples_map=None,
    n_clusters_map=None,
    seed: int = 42,
    # UMAP params
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    # image feature params
    hog_orientations: int = 9,
    hog_pixels_per_cell: int = 4,
    hog_cells_per_block: int = 2,
):
    """
    Vẽ scatter 3 cột cho mỗi dataset:
    [Ground Truth (PCA 2D) | PCA->KMeans | UMAP->KMeans]
    X đã được chuẩn hóa thành ma trận đặc trưng không còn là pixel thô nên có thể
    trực quan hóa scatter 2D đc
    """
    if n_samples_map is None:
        n_samples_map = {"mnist": 4000, "fashion_mnist": 4000}
    if n_clusters_map is None:
        n_clusters_map = {"mnist": 10, "fashion_mnist": 10}

    cols = ["GT", "PCA+KMeans", "UMAP+KMeans"]
    nrows = len(datasets)
    ncols = len(cols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.2 * nrows))

    # normalize axes to 2D
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for r, ds in enumerate(datasets):
        n_samples = int(n_samples_map.get(ds, 4000))
        n_clusters = int(n_clusters_map.get(ds, 10))

        # Load data
        X, y_true = _load_dataset(
            dataset=ds,
            n_samples=n_samples,
            n_clusters=n_clusters,
            seed=seed,
            hog_orientations=hog_orientations,
            hog_pixels_per_cell=hog_pixels_per_cell,
            hog_cells_per_block=hog_cells_per_block,
        )

        # 2D embeddings (để GT cũng vẽ trên cùng “mặt phẳng” so sánh được)
        pca2 = make_pca(2, seed).fit_transform(X)
        umap2 = make_umap(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=None,
            n_jobs=-1
        ).fit_transform(X)

        # KMeans trên từng embedding
        y_pred_pca = make_kmeans(n_clusters, seed).fit_predict(pca2)
        y_pred_umap = make_kmeans(n_clusters, seed).fit_predict(umap2)

        # --- Col 1: Ground Truth (PCA2) ---
        ax = axes[r][0]
        ax.scatter(pca2[:, 0], pca2[:, 1], s=6, c=y_true, alpha=0.85)
        ax.set_title(f"{ds} | GT (PCA2)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        # --- Col 2: PCA + KMeans ---
        ax = axes[r][1]
        ax.scatter(pca2[:, 0], pca2[:, 1], s=6, c=y_pred_pca, alpha=0.85)
        ax.set_title(f"{ds} | PCA -> KMeans (k={n_clusters})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        # --- Col 3: UMAP + KMeans ---
        ax = axes[r][2]
        ax.scatter(umap2[:, 0], umap2[:, 1], s=6, c=y_pred_umap, alpha=0.85)
        ax.set_title(f"{ds} | UMAP -> KMeans (k={n_clusters})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

    fig.suptitle("Ground Truth (PCA2) vs PCA/UMAP + KMeans (2D scatter)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def run_one(
    dataset: str,
    dimred: str,
    n_samples: int,
    d: int,
    n_clusters: int,
    seed: int,
    # image pipeline
    kernel_mode: str,
    hog_orientations: int,
    hog_pixels_per_cell: int,
    hog_cells_per_block: int,
    # UMAP params
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
):
    set_seed(seed)

    X, y_true = _load_dataset(
        dataset=dataset,
        n_samples=n_samples,
        n_clusters=n_clusters,
        seed=seed,
        kernel_mode=kernel_mode,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        hog_cells_per_block=hog_cells_per_block,
    )

    dr = _make_dimred(dimred, d, seed, umap_n_neighbors, umap_min_dist, umap_metric)
    km = make_kmeans(n_clusters, seed)

    with timer() as t_dim:
        X_emb = dr.fit_transform(X)
    time_dimred = t_dim()

    with timer() as t_km:
        y_pred = km.fit_predict(X_emb)
    time_kmeans = t_km()

    total_time = time_dimred + time_kmeans
    m = compute_metrics(X_emb, y_true, y_pred)

    return {
        "dataset": dataset,
        "dimred": dimred,
        "n_samples": n_samples,
        "d": d,
        "n_clusters": n_clusters,
        "seed": seed,
        "silhouette": m["silhouette"],
        "ari": m["ari"],
        "nmi": m["nmi"],
        "time_dimred_sec": time_dimred,
        "time_kmeans_sec": time_kmeans,
        "time_total_sec": total_time,
    }


def run_all_sweeps(cfgs, out_dir: Path):
    rows = []
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    for cfg in cfgs:
        # Lọc dims hợp lệ cho PCA: d <= min(n_samples, n_features)
        X0, _ = _load_dataset(
            dataset=cfg.dataset,
            n_samples=cfg.n_samples,
            n_clusters=cfg.n_clusters,
            seed=cfg.seeds[0],
            kernel_mode=cfg.kernel_mode,
            hog_orientations=cfg.hog_orientations,
            hog_pixels_per_cell=cfg.hog_pixels_per_cell,
            hog_cells_per_block=cfg.hog_cells_per_block,
        )
        max_d = min(X0.shape[0], X0.shape[1])
        dims = [d for d in cfg.dims if d <= max_d]

        for d in dims:
            for seed in cfg.seeds:
                rows.append(
                    run_one(
                        dataset=cfg.dataset,
                        dimred=cfg.dimred,
                        n_samples=cfg.n_samples,
                        d=d,
                        n_clusters=cfg.n_clusters,
                        seed=seed,
                        kernel_mode=cfg.kernel_mode,
                        hog_orientations=cfg.hog_orientations,
                        hog_pixels_per_cell=cfg.hog_pixels_per_cell,
                        hog_cells_per_block=cfg.hog_cells_per_block,
                        umap_n_neighbors=cfg.umap_n_neighbors,
                        umap_min_dist=cfg.umap_min_dist,
                        umap_metric=cfg.umap_metric,
                    )
                )

    df_detail = pd.DataFrame(rows)
    df_detail.to_csv(out_dir / "results_detail.csv", index=False)

    agg_cols = ["silhouette", "ari", "nmi", "time_total_sec", "time_dimred_sec", "time_kmeans_sec"]
    df_sum = (
        df_detail
        .groupby(["dataset", "dimred", "d", "n_samples", "n_clusters"], as_index=False)[agg_cols]
        .agg(["mean", "std"])
    )

    df_sum.columns = [f"{a}_{b}" if b else a for (a, b) in df_sum.columns.to_flat_index()]
    df_sum = df_sum.rename(columns={
        "dataset_": "dataset",
        "dimred_": "dimred",
        "d_": "d",
        "n_samples_": "n_samples",
        "n_clusters_": "n_clusters"
    })
    df_sum.to_csv(out_dir / "results_summary.csv", index=False)

    # Plot tổng hợp metrics vs d (1 file)
    plot_all_subplots(df_sum, out_dir / "figures" / "ALL_PLOTS_SUBPLOTS.png")

    # Plot scatter phân cụm (MNIST & Fashion-MNIST)
    plot_cluster_scatter_2datasets(out_dir / "figures" / "CLUSTERS_2DATASETS_PCA_UMAP.png")

    return df_detail, df_sum