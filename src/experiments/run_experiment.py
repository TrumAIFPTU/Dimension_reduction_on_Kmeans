from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.seed import set_seed
from src.initialize.data import load_mnist_images
from src.initialize.data import load_fashion_mnist_images
from src.initialize.dimred import make_pca
from src.initialize.dimred import make_umap
from src.initialize.kmeans import make_kmeans

from src.image.preprocess import preprocess_and_sharpen
from src.image.features import extract_feature_matrix

from src.evaluation.metrics import compute_metrics
from src.evaluation.timer import timer




def _load_dataset(
    dataset: str,
    n_samples: int,
    n_clusters: int,
    seed: int,
    pipeline: str = "sharpen_hog",
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

    if pipeline == "raw":
        X01 = X_img.astype(np.float32) / 255.0
        X = X01.reshape(X01.shape[0], -1)  # (N, 784)
        X = StandardScaler().fit_transform(X).astype(np.float32)
        return X, y.astype(int)
    elif pipeline == "sharpen_hog":
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

def run_one(
    dataset: str,
    dimred: str,
    n_samples: int,
    d: int,
    n_clusters: int,
    seed: int,
    pipeline: str,
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
    #load datasets
    X, y_true = _load_dataset(
        dataset=dataset,
        n_samples=n_samples,
        n_clusters=n_clusters,
        seed=seed,
        pipeline=pipeline,
        kernel_mode=kernel_mode,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        hog_cells_per_block=hog_cells_per_block,
    )

    dr = _make_dimred(dimred, d, seed, umap_n_neighbors, umap_min_dist, umap_metric)
    km = make_kmeans(n_clusters, seed)
    #Time + metrics
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
        "pipeline": pipeline,
        "silhouette": m["silhouette"],
        "ari": m["ari"],
        "nmi": m["nmi"],
        "time_dimred_sec": time_dimred,
        "time_kmeans_sec": time_kmeans,
        "time_total_sec": total_time,
    }


def run_all_sweeps(cfgs, out_dir: Path):
    from src.plot.visualize import plot_all_subplots
    from src.plot.visualize import plot_cluster_scatter_2datasets
    rows = []
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    for cfg in cfgs:
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
        max_d = min(X0.shape[0], X0.shape[1])# Lọc dims hợp lệ cho PCA: d <= min(n_samples, n_features)
        dims = [d for d in cfg.dims if d <= max_d]

        for pipeline in ["raw", "sharpen_hog"]:
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
                            pipeline=pipeline,
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
        .groupby(["dataset","pipeline" , "dimred", "d", "n_samples", "n_clusters"], as_index=False)[agg_cols]
        .agg(["mean", "std"])
    )

    df_sum.columns = [f"{a}_{b}" if b else a for (a, b) in df_sum.columns.to_flat_index()]
    df_sum = df_sum.rename(columns={
        "dataset_": "dataset",
        "pipeline_": "pipeline",
        "dimred_": "dimred",
        "d_": "d",
        "n_samples_": "n_samples",
        "n_clusters_": "n_clusters"
    })
    df_sum.to_csv(out_dir / "results_summary.csv", index=False)

    # Plot tổng hợp metrics vs d (2 file)
    plot_all_subplots(
        df_sum[df_sum["pipeline"] == "raw"].copy(),
        out_dir / "figures" / "LINES_raw.png"
    )
    plot_all_subplots(
        df_sum[df_sum["pipeline"] == "sharpen_hog"].copy(),
        out_dir / "figures" / "LINES_sharpen_hog.png"
    )


    # Plot scatter phân cụm (MNIST & Fashion-MNIST)
    # --- Scatter không kernel & features extraction--
    plot_cluster_scatter_2datasets(
        out_dir / "figures" / "SCATTER_raw.png",
        load_dataset_fn=lambda **kw: _load_dataset(**kw, pipeline="raw"),
        make_pca_fn=make_pca,
        make_umap_fn=make_umap,
        make_kmeans_fn=make_kmeans,
    )

    # --- Scatter với kernel(sharpen) & features extraction(hog) ---
    plot_cluster_scatter_2datasets(
        out_dir / "figures" / "SCATTER_sharpen_hog.png",
        load_dataset_fn=lambda **kw: _load_dataset(**kw, pipeline="sharpen_hog"),
        make_pca_fn=make_pca,
        make_umap_fn=make_umap,
        make_kmeans_fn=make_kmeans,
    )


    return df_detail, df_sum