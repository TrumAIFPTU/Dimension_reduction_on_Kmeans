from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.datasets.mnist import load_mnist
from src.datasets.swiss_roll import load_swiss_roll
from src.dimred.pca import make_pca
from src.dimred.umap import make_umap   # ✅ đổi tsne -> umap
from src.clustering.kmeans import make_kmeans
from src.evaluation.metrics import compute_metrics
from src.evaluation.timer import timer
from src.utils.io import save_results_csv


def _load_dataset(dataset: str, n_samples: int, n_clusters: int, seed: int):
    if dataset == "mnist":
        return load_mnist(n_samples, seed)
    if dataset == "swiss_roll":
        return load_swiss_roll(n_samples, n_clusters, seed)
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
            seed: int, umap_n_neighbors: int, umap_min_dist: float, umap_metric: str):
    set_seed(seed)
    X, y_true = _load_dataset(dataset, n_samples, n_clusters, seed)

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

    # 1) Detail runs
    for cfg in cfgs:
        # Lọc dims hợp lệ theo dữ liệu (quan trọng cho PCA: d <= n_features)
        X0, _ = _load_dataset(cfg.dataset, cfg.n_samples, cfg.n_clusters, cfg.seeds[0])
        max_d = min(X0.shape[0], X0.shape[1])
        dims = [d for d in cfg.dims if d <= max_d]

        #UMAP không cần giới hạn d<4 như t-SNE nên bỏ phần đó

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
                    )
                )

    df_detail = pd.DataFrame(rows)
    df_detail.to_csv(out_dir / "results_detail.csv", index=False)

    # 2) Summary mean/std over seeds
    agg_cols = ["silhouette", "ari", "nmi", "time_total_sec", "time_dimred_sec", "time_kmeans_sec"]
    df_sum = (
        df_detail
        .groupby(["dataset", "dimred", "d", "n_samples", "n_clusters"], as_index=False)[agg_cols]
        .agg(["mean", "std"])
    )

    # flatten columns
    df_sum.columns = [
        f"{a}_{b}" if b else a
        for (a, b) in df_sum.columns.to_flat_index()
    ]
    df_sum = df_sum.rename(columns={
        "dataset_": "dataset",
        "dimred_": "dimred",
        "d_": "d",
        "n_samples_": "n_samples",
        "n_clusters_": "n_clusters"
    })
    df_sum.to_csv(out_dir / "results_summary.csv", index=False)

    # 3) Plot metrics vs d
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
