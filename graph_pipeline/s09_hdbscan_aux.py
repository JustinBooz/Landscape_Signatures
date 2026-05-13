"""
Step 09: Auxiliary HDBSCAN (point-wise, on 1M sample)
======================================================
Runs HDBSCAN on a 1M random sample of PCA embeddings (full-data HDBSCAN
materialises an MST that's typically tens of GB at 7.2M points). Sample labels
are lifted to the full 7.2M dataset via 1-NN in the same PCA space so the
resulting parquet still has one row per image. Comparison with Leiden uses the
lifted labels but the actual HDBSCAN structure is computed on the sample.

`min_cluster_size` / `min_samples` from `config.HDBSCAN_SETTINGS` are scaled
down by `n / sample_size` so the cluster-size semantics stay roughly stable.

Outputs:
  - hdbscan_pointwise_labels.parquet
  - hdbscan_noise_summary.csv
  - hdbscan_vs_leiden_contingency.csv
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import hdbscan
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from leiden_parallel import REF_SEED  # noqa: F401  (kept for backward-compat callers)

logger = config.setup_logging("s09_hdbscan_aux")


def _scale_min_cluster_size(mcs, sample_size, full_size):
    """Scale `mcs` from full-data semantics down to sample semantics, with a floor."""
    scaled = max(50, int(round(mcs * sample_size / full_size)))
    return scaled


def _lift_labels_via_1nn(pca_data_full, sample_idx, sample_labels):
    """Assign each non-sample point the label of its nearest sample point.
    Noise (-1) propagates: a non-sample point whose nearest sample is noise is
    itself noise.
    """
    n_full, d = pca_data_full.shape
    sample_vecs = np.ascontiguousarray(pca_data_full[sample_idx], dtype=np.float32)

    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    index.add(sample_vecs)

    full_labels = np.empty(n_full, dtype=np.int32)
    batch = 200_000
    for start in range(0, n_full, batch):
        end = min(start + batch, n_full)
        chunk = np.ascontiguousarray(pca_data_full[start:end], dtype=np.float32)
        _, I = index.search(chunk, 1)
        full_labels[start:end] = sample_labels[I.ravel()]
    return full_labels


def run():
    logger.info("=" * 60)
    logger.info("STEP 9: Auxiliary HDBSCAN (point-wise, on 1M sample)")
    logger.info("=" * 60)

    t0 = time.time()

    labels_path = config.hdbscan_pointwise_labels_path()
    if os.path.exists(labels_path):
        logger.info(f"[CHECKPOINT] HDBSCAN labels already exist: {labels_path}")
        return

    pca_dim = config.get_selected_pca_dim()

    logger.info(f"Loading PCA-{pca_dim} embeddings...")
    pca_data = config.load_memmap(
        config.pca_path(pca_dim),
        config.pca_shape_path(pca_dim),
        dtype='float32', mode='r'
    )
    n_full = pca_data.shape[0]
    sample_size = min(config.HDBSCAN_SAMPLE_SIZE, n_full)
    logger.info(f"  Full: {pca_data.shape}, sample: {sample_size:,}")

    manifest_df = pd.read_parquet(config.manifest_path(), columns=['image_id'])

    # Reproducible random sample, sorted for memmap-friendly access
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_idx = np.sort(rng.choice(n_full, sample_size, replace=False))
    sample_data = np.ascontiguousarray(pca_data[sample_idx], dtype=np.float32)
    logger.info(f"  Sample matrix loaded: {sample_data.shape} "
                f"({sample_data.nbytes / 1e9:.1f} GB)")

    all_full_labels = {}
    noise_records = []

    for params in config.HDBSCAN_SETTINGS:
        mcs_full = params['min_cluster_size']
        ms_full = params['min_samples']
        mcs = _scale_min_cluster_size(mcs_full, sample_size, n_full)
        ms = _scale_min_cluster_size(ms_full, sample_size, n_full)
        col_name = f"hdbscan_mcs{mcs_full}_ms{ms_full}"
        sample_ckpt = os.path.join(config.OUTPUT_DIR,
                                    f"hdbscan_{col_name}_sample_labels.npy")
        full_ckpt = os.path.join(config.OUTPUT_DIR,
                                  f"hdbscan_{col_name}_full_labels.npy")

        if os.path.exists(full_ckpt):
            logger.info(f"  [CHECKPOINT] {col_name} full labels exist")
            full_labels = np.load(full_ckpt)
        else:
            if os.path.exists(sample_ckpt):
                logger.info(f"  [CHECKPOINT] {col_name} sample labels exist; lifting")
                sample_labels = np.load(sample_ckpt)
            else:
                logger.info(f"\n  HDBSCAN on sample: full(mcs={mcs_full}, ms={ms_full}) "
                            f"→ sample(mcs={mcs}, ms={ms})")
                t1 = time.time()
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric='euclidean',
                    algorithm='boruvka_kdtree',
                    approx_min_span_tree=True,
                    core_dist_n_jobs=-1,
                )
                sample_labels = clusterer.fit_predict(sample_data).astype(np.int32)
                logger.info(f"    Sample HDBSCAN done in {(time.time()-t1)/60:.1f} min")
                np.save(sample_ckpt, sample_labels)
                del clusterer
                gc.collect()

            logger.info(f"  Lifting {sample_size:,} sample labels to {n_full:,} "
                        f"points via 1-NN...")
            t2 = time.time()
            full_labels = _lift_labels_via_1nn(pca_data, sample_idx, sample_labels)
            logger.info(f"    Lifted in {(time.time()-t2)/60:.1f} min")
            np.save(full_ckpt, full_labels)

        unique_labels = np.unique(full_labels)
        n_clusters = len(unique_labels) - (1 if (unique_labels == -1).any() else 0)
        n_noise = int((full_labels == -1).sum())
        pct_noise = n_noise / n_full * 100
        logger.info(f"    {col_name}: {n_clusters} clusters, "
                    f"{n_noise:,} noise ({pct_noise:.1f}%)")

        all_full_labels[col_name] = full_labels
        noise_records.append({
            'setting': col_name,
            'min_cluster_size': mcs_full,
            'min_samples': ms_full,
            'sample_min_cluster_size': mcs,
            'sample_min_samples': ms,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'pct_noise': pct_noise,
        })

    del sample_data
    gc.collect()

    df = manifest_df.copy()
    for col_name, labels in all_full_labels.items():
        df[col_name] = labels
    df.to_parquet(labels_path, index=False)
    logger.info(f"\nHDBSCAN labels saved: {labels_path}")

    noise_df = pd.DataFrame(noise_records)
    noise_df.to_csv(config.hdbscan_noise_summary_path(), index=False)
    logger.info(f"Noise summary saved: {config.hdbscan_noise_summary_path()}")

    # Contingency with Leiden (uses the lifted full labels)
    logger.info("\n--- HDBSCAN vs Leiden Contingency ---")
    pca_dim_sel = config.get_selected_pca_dim()
    k = config.KNN_DEFAULT_K
    gt = config.GRAPH_DEFAULT_TYPE

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    contingency_records = []
    for res in config.LEIDEN_RESOLUTIONS:
        leiden_path = config.leiden_labels_path(pca_dim_sel, k, gt, res, config.REF_SEED)
        if not os.path.exists(leiden_path):
            continue
        leiden_labels = np.load(leiden_path)

        for col_name, hdb_labels in all_full_labels.items():
            mask = hdb_labels >= 0
            if mask.sum() < 100:
                continue
            ari = adjusted_rand_score(leiden_labels[mask], hdb_labels[mask])
            nmi = normalized_mutual_info_score(leiden_labels[mask], hdb_labels[mask])
            contingency_records.append({
                'hdbscan_setting': col_name,
                'leiden_resolution': res,
                'ari': ari,
                'nmi': nmi,
                'n_compared': int(mask.sum()),
            })
            logger.info(f"  {col_name} vs leiden_res{res}: "
                        f"ARI={ari:.4f}, NMI={nmi:.4f}")

    if contingency_records:
        cont_df = pd.DataFrame(contingency_records)
        cont_df.to_csv(config.hdbscan_vs_leiden_path(), index=False)
        logger.info(f"Contingency saved: {config.hdbscan_vs_leiden_path()}")

    elapsed = time.time() - t0
    logger.info(f"\nAuxiliary HDBSCAN complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    run()
