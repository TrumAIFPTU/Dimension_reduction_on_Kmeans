from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.datasets.mnist import load_mnist
from src.datasets.fashion_mnist import load_fashion_mnist
from src.datasets.swiss_roll import load_swiss_roll
from src.datasets.blobs_linear import load_blobs_linear

from src.dimred.pca import make_pca
from src.dimred.umap import make_umap
from src.clustering.kmeans import make_kmeans
from src.evaluation.metrics import compute_metrics
from src.evaluation.timer import timer


def _load_dataset(dataset: str, n_samples: int, n_clusters: int, seed: int,
                  blobs_n_features: int = 50, blobs_cluster_std: float = 1.0):
    if dataset == "mnist":
        return load_mnist(n_samples, seed)

    if dataset == "fashion_mnist":
        return load_fashion_mnist(n_samples, seed)

    if dataset == "swiss_roll":
        # y_true được tạo bằng binning (0..K-1)
        return load_swiss_roll(n_samples, n_clusters, seed)

    if dataset == "blobs_linear":
        return load_blobs_linear(
            n_samples=n_samples,
            n_features=blobs_n_features,
            n_clusters=n_clusters,
            random_state=seed,
            cluster_std=blobs_cluster_std
        )

    raise ValueError(f"Unknown dataset: {dataset}")


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
            random_state=None,  # để chạy parallel, tránh warning do seed
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
    Nếu vẫn muốn xuất từng plot riêng thì bỏ comment đoạn này

"""
# ================================================================================================================


def plot_all_subplots(df_sum: pd.DataFrame, out_path: Path):
    """
    1 figure lớn:
    - Cột: mỗi dataset
    - Hàng: mỗi metric
    - Trong mỗi ô: các đường theo dimred (PCA/UMAP) theo d
    """
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
            display_name = dataset
            if dataset == "blobs_linear":
                display_name = "linear"
            elif dataset == "fashion_mnist":
                display_name = "nlinearmed"
            elif dataset == "mnist":
                display_name = "nlinearlow"
            elif dataset == "swiss_roll":
                display_name = "nlinearhigh"

            ax.set_title(f"{display_name} | {metric}")
            ax.set_xlabel("dimension")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

            if r == 0 and c == 0:
                ax.legend()

    fig.suptitle("PCA and UMAP on KMeans algorithms: Metrics and Reduced Dimension d", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_cluster_scatter_4datasets(
    out_path: Path,
    datasets=("blobs_linear", "mnist", "fashion_mnist", "swiss_roll"),
    n_samples_map=None,
    n_clusters_map=None,
    seed: int = 42,
    # UMAP params
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    # Blobs params
    blobs_n_features: int = 50,
    blobs_cluster_std: float = 1.0,
):
    """
    Vẽ scatter phân cụm KMeans sau PCA/UMAP cho 4 dataset.
    Layout: rows = datasets, cols = [PCA, UMAP]
    Mỗi ô: màu theo nhãn KMeans (y_pred).
    """
    if n_samples_map is None:
        n_samples_map = {
            "blobs_linear": 4000,
            "mnist": 4000,
            "fashion_mnist": 4000,
            "swiss_roll": 4000,
        }
    if n_clusters_map is None:
        n_clusters_map = {
            "blobs_linear": 10,
            "mnist": 10,
            "fashion_mnist": 10,
            "swiss_roll": 10,
        }

    methods = ["pca", "umap"]
    nrows = len(datasets)
    ncols = len(methods)

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

        X, _y_true = _load_dataset(
            dataset=ds,
            n_samples=n_samples,
            n_clusters=n_clusters,
            seed=seed,
            blobs_n_features=blobs_n_features,
            blobs_cluster_std=blobs_cluster_std
        )

        for c, dimred in enumerate(methods):
            ax = axes[r][c]

            # reduce to 2D for plotting
            if dimred == "pca":
                dr = make_pca(2, seed)
            else:
                dr = make_umap(
                    n_components=2,
                    n_neighbors=umap_n_neighbors,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                    random_state=None,
                    n_jobs=-1
                )

            X_2d = dr.fit_transform(X)

            # cluster on 2D embedding (giảm chiều rồi mới KMeans)
            km = make_kmeans(n_clusters, seed)
            y_pred = km.fit_predict(X_2d)

            ax.scatter(X_2d[:, 0], X_2d[:, 1], s=6, c=y_pred, alpha=0.85)
            ax.set_title(f"{ds} | {dimred.upper()} -> KMeans (k={n_clusters})")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.15)

    fig.suptitle("Cluster visualization after PCA/UMAP (2D) + KMeans", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def run_one(dataset: str, dimred: str, n_samples: int, d: int, n_clusters: int,
            seed: int,
            umap_n_neighbors: int, umap_min_dist: float, umap_metric: str,
            blobs_n_features: int, blobs_cluster_std: float):
    set_seed(seed)

    X, y_true = _load_dataset(
        dataset=dataset,
        n_samples=n_samples,
        n_clusters=n_clusters,
        seed=seed,
        blobs_n_features=blobs_n_features,
        blobs_cluster_std=blobs_cluster_std
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
            blobs_n_features=cfg.blobs_n_features,
            blobs_cluster_std=cfg.blobs_cluster_std
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
                        umap_n_neighbors=cfg.umap_n_neighbors,
                        umap_min_dist=cfg.umap_min_dist,
                        umap_metric=cfg.umap_metric,
                        blobs_n_features=cfg.blobs_n_features,
                        blobs_cluster_std=cfg.blobs_cluster_std
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

    #Plot tổng hợp metrics vs d (1 file)
    plot_all_subplots(df_sum, out_dir / "figures" / "ALL_PLOTS_SUBPLOTS.png")

    #Plot scatter phân cụm sau PCA/UMAP cho 4 dataset (1 file)
    plot_cluster_scatter_4datasets(out_dir / "figures" / "CLUSTERS_4DATASETS_PCA_UMAP.png")

    """
    for (dataset, dimred) in df_sum[["dataset", "dimred"]].drop_duplicates().itertuples(index=False):
        for metric in ["silhouette", "ari", "nmi", "time_total_sec"]:
            _plot_metric_vs_d(
                df_sum,
                dataset=dataset,
                dimred=dimred,
                metric=metric,
                out_path=out_dir / "figures" / f"{dataset}_{dimred}_{metric}_vs_d.png"
            )
    Nếu vẫn muốn xuất từng plot riêng thì bỏ comment đoạn này
    """

    return df_detail, df_sum