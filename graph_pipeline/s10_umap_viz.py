"""
Step 10: UMAP Visualization
==============================
Generates UMAP coordinates for stratified samples (250K, 500K).
Colors are Leiden labels discovered from kNN graph — NOT re-clustered.

Outputs:
  - umap_sample_250k.parquet
  - umap_sample_500k.parquet
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from leiden_parallel import REF_SEED

logger = config.setup_logging("s10_umap_viz")


def _stratified_sample(labels, coords, n_sample, min_per_cluster=50):
    """Create a stratified sample preserving cluster and geographic distribution."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    n_total = len(labels)

    if n_sample >= n_total:
        return np.arange(n_total)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Allocate proportionally, with minimum per cluster
    cluster_sizes = {l: (labels == l).sum() for l in unique_labels}
    total_min = min(min_per_cluster * n_clusters, n_sample // 2)

    # Proportional allocation for remaining budget
    remaining_budget = n_sample - total_min
    total_count = sum(cluster_sizes.values())

    sampled_indices = []

    for label in unique_labels:
        members = np.where(labels == label)[0]
        n_members = len(members)

        # Minimum allocation
        n_min = min(min_per_cluster, n_members)
        # Proportional allocation
        n_prop = int(remaining_budget * n_members / total_count)
        n_alloc = min(n_min + n_prop, n_members)

        if n_alloc >= n_members:
            sampled_indices.extend(members.tolist())
        else:
            picked = rng.choice(members, n_alloc, replace=False)
            sampled_indices.extend(picked.tolist())

    # Deduplicate
    sampled_arr = np.unique(np.asarray(sampled_indices, dtype=np.int64))
    if len(sampled_arr) > n_sample:
        sampled_arr = rng.choice(sampled_arr, n_sample, replace=False)
    elif len(sampled_arr) < n_sample:
        # Pick remaining from the complement via a boolean mask (O(N) memory, no big sets)
        used = np.zeros(n_total, dtype=bool)
        used[sampled_arr] = True
        remaining = np.flatnonzero(~used)
        n_extra = min(n_sample - len(sampled_arr), len(remaining))
        if n_extra > 0:
            extra = rng.choice(remaining, n_extra, replace=False)
            sampled_arr = np.concatenate([sampled_arr, extra])

    sampled_arr.sort()
    return sampled_arr


def _run_umap_for_sample(pca_data, sample_idx, umap_params, pca_dim):
    """Run UMAP on a sample of PCA embeddings."""
    import umap

    data = pca_data[sample_idx].astype(np.float32).copy()
    n = len(data)

    logger.info(f"  UMAP: {n:,} points, {pca_dim}D → 2D")
    logger.info(f"    n_neighbors={umap_params['n_neighbors']}, "
                f"min_dist={umap_params['min_dist']}")
    t0 = time.time()

    reducer = umap.UMAP(
        n_components=config.UMAP_N_COMPONENTS,
        n_neighbors=umap_params['n_neighbors'],
        min_dist=umap_params['min_dist'],
        metric=config.UMAP_METRIC,
        random_state=config.RANDOM_STATE,
        low_memory=True,
        n_jobs=-1,
        verbose=True,
    )
    coords_2d = reducer.fit_transform(data)

    elapsed = time.time() - t0
    logger.info(f"    UMAP done in {elapsed / 60:.1f} min")

    del data, reducer
    gc.collect()

    return coords_2d.astype(np.float32)


def run():
    logger.info("=" * 60)
    logger.info("STEP 10: UMAP Visualization")
    logger.info("=" * 60)

    t0 = time.time()

    pca_dim = config.get_selected_pca_dim()

    # Load PCA data
    logger.info(f"Loading PCA-{pca_dim} embeddings...")
    pca_data = config.load_memmap(
        config.pca_path(pca_dim),
        config.pca_shape_path(pca_dim),
        dtype='float32', mode='r'
    )

    # Load manifest
    manifest_df = pd.read_parquet(config.manifest_path())

    # Load Leiden labels (reference resolution + reference seed for sampling)
    k = config.KNN_DEFAULT_K
    gt = config.GRAPH_DEFAULT_TYPE
    primary_leiden_path = config.leiden_labels_path(pca_dim, k, gt, 1.0, REF_SEED)
    if os.path.exists(primary_leiden_path):
        primary_labels = np.load(primary_leiden_path)
    else:
        logger.warning("No Leiden labels found for stratification, "
                        "using random sampling")
        primary_labels = np.zeros(len(manifest_df), dtype=np.int32)

    coords = manifest_df[['norm_easting', 'norm_northing']].values

    # Use first UMAP setting as default
    umap_params = config.UMAP_SETTINGS[0]

    for n_sample in config.UMAP_SAMPLE_SIZES:
        output_path = config.umap_sample_path(n_sample)
        if os.path.exists(output_path):
            logger.info(f"[CHECKPOINT] UMAP sample {n_sample // 1000}k exists")
            continue

        logger.info(f"\n--- UMAP {n_sample // 1000}k sample ---")

        # Stratified sampling
        sample_idx = _stratified_sample(primary_labels, coords, n_sample)
        logger.info(f"  Sampled {len(sample_idx):,} indices")

        # Run UMAP
        umap_2d = _run_umap_for_sample(pca_data, sample_idx, umap_params,
                                         pca_dim)

        # Build output dataframe
        df = manifest_df.iloc[sample_idx][['image_id', 'norm_easting',
                                            'norm_northing']].copy()
        df = df.reset_index(drop=True)
        df['umap_x'] = umap_2d[:, 0]
        df['umap_y'] = umap_2d[:, 1]
        df['sample_index'] = sample_idx

        # Add Leiden labels for available resolutions
        for res in config.LEIDEN_RESOLUTIONS:
            label_path = config.leiden_labels_path(pca_dim, k, gt, res, REF_SEED)
            if os.path.exists(label_path):
                all_labels = np.load(label_path)
                res_str = f"{res:.2f}".replace(".", "")
                df[f'leiden_res{res_str}'] = all_labels[sample_idx]

        df.to_parquet(output_path, index=False)
        logger.info(f"  Saved: {output_path}")

        del umap_2d, sample_idx
        gc.collect()

    elapsed = time.time() - t0
    logger.info(f"\nUMAP visualization complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    run()
