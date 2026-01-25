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
            random_state=None,
            n_jobs=-1
        )
    #print("UMAP params:", model.get_params().get("random_state"), model.get_params().get("n_jobs")) -> để xem lỗi ở umap nếu n_jobs k đạt -1 
    raise ValueError(f"Unknown dimred: {dimred}")


def _plot_metric_vs_d(df_sum: pd.DataFrame, dataset: str, dimred: str, metric: str, out_path: Path):
    sub = df_sum[(df_sum["dataset"] == dataset) & (df_sum["dimred"] == dimred)].sort_values("d")
    plt.figure()
    plt.plot(sub["d"], sub[f"{metric}_mean"], marker="o")
    plt.xlabel("Reduced dimension d")
    plt.ylabel(metric)
    plt.title(f"{dataset.upper()} + {dimred.upper()} : {metric} vs d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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

    for (dataset, dimred) in df_sum[["dataset", "dimred"]].drop_duplicates().itertuples(index=False):
        for metric in ["silhouette", "ari", "nmi", "time_total_sec"]:
            _plot_metric_vs_d(
                df_sum,
                dataset=dataset,
                dimred=dimred,
                metric=metric,
                out_path=out_dir / "figures" / f"{dataset}_{dimred}_{metric}_vs_d.png"
            )

    return df_detail, df_sum
