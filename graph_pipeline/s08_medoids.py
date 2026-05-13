"""
Step 08: Medoid & Representative Image Extraction
====================================================
For each Leiden cluster at each resolution, computes medoid,
nearest members, random samples, and spatial samples.

Outputs:
  - cluster_medoids_res{res}.parquet (per resolution)
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

logger = config.setup_logging("s08_medoids")


def _compute_medoid(pca_data, member_indices, max_sample=None):
    """Find the medoid (member with highest mean cosine similarity to others).
    For large clusters, subsample to max_sample members.
    """
    if max_sample and len(member_indices) > max_sample:
        rng = np.random.RandomState(config.RANDOM_STATE)
        sample = rng.choice(member_indices, max_sample, replace=False)
    else:
        sample = member_indices

    vectors = pca_data[sample].astype(np.float32)

    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vectors = vectors / norms

    # Compute pairwise similarities in chunks to avoid memory explosion
    n = len(sample)
    mean_sims = np.zeros(n, dtype=np.float32)
    chunk_size = 2000

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = vectors[start:end]  # [chunk, d]
        sims = chunk @ vectors.T  # [chunk, n]
        # Zero out self-similarity (diagonal) without a Python loop
        rows_local = np.arange(end - start)
        cols_global = np.arange(start, end)
        sims[rows_local, cols_global] = 0
        mean_sims[start:end] = sims.sum(axis=1) / max(n - 1, 1)

    medoid_local_idx = np.argmax(mean_sims)
    medoid_global_idx = sample[medoid_local_idx]

    # Intra-cluster similarity stats
    all_sims = mean_sims  # mean similarity per member
    sim_stats = {
        'intra_mean_sim': float(all_sims.mean()),
        'intra_p10_sim': float(np.percentile(all_sims, 10)),
        'intra_p25_sim': float(np.percentile(all_sims, 25)),
        'intra_p50_sim': float(np.median(all_sims)),
        'intra_p75_sim': float(np.percentile(all_sims, 75)),
        'intra_p90_sim': float(np.percentile(all_sims, 90)),
    }

    return medoid_global_idx, sim_stats, vectors, sample


def _get_nearest_to_medoid(pca_data, medoid_idx, member_indices, top_n=20):
    """Find top_n members nearest to medoid in PCA space."""
    medoid_vec = pca_data[medoid_idx].astype(np.float32)
    medoid_vec = medoid_vec / max(np.linalg.norm(medoid_vec), 1e-12)

    member_vecs = pca_data[member_indices].astype(np.float32)
    norms = np.linalg.norm(member_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    member_vecs = member_vecs / norms

    sims = member_vecs @ medoid_vec
    # argpartition is O(N) vs argsort's O(N log N); reorder partition by score
    if len(sims) > top_n:
        part = np.argpartition(-sims, top_n)[:top_n]
        top_local = part[np.argsort(-sims[part])]
    else:
        top_local = np.argsort(-sims)
    return member_indices[top_local].tolist()


def _spatial_sample(coords, member_indices, n_sample=100):
    """Grid-based spatial sampling to get geographically distributed members."""
    if len(member_indices) <= n_sample:
        return member_indices.tolist()

    member_coords = coords[member_indices]
    e_min, e_max = member_coords[:, 0].min(), member_coords[:, 0].max()
    n_min, n_max = member_coords[:, 1].min(), member_coords[:, 1].max()

    # Create grid
    grid_side = int(np.sqrt(n_sample)) + 1
    e_bins = np.linspace(e_min, e_max + 1e-10, grid_side + 1)
    n_bins = np.linspace(n_min, n_max + 1e-10, grid_side + 1)

    sampled = []
    rng = np.random.RandomState(config.RANDOM_STATE)

    e_idx = np.digitize(member_coords[:, 0], e_bins) - 1
    n_idx = np.digitize(member_coords[:, 1], n_bins) - 1

    for ei in range(grid_side):
        for ni in range(grid_side):
            mask = (e_idx == ei) & (n_idx == ni)
            cell_members = np.where(mask)[0]
            if len(cell_members) > 0:
                pick = rng.choice(cell_members)
                sampled.append(member_indices[pick])

    # If we have fewer than n_sample, pad with random
    if len(sampled) < n_sample:
        remaining = set(member_indices.tolist()) - set(sampled)
        remaining = list(remaining)
        if remaining:
            extra = rng.choice(remaining,
                                min(n_sample - len(sampled), len(remaining)),
                                replace=False)
            sampled.extend(extra.tolist())

    return sampled[:n_sample]


def _process_resolution(labels, pca_data, manifest_df, resolution):
    """Process all clusters for a given resolution."""
    output_path = config.medoids_path(resolution)
    if os.path.exists(output_path):
        logger.info(f"  [CHECKPOINT] Medoids for res={resolution} exist")
        return

    coords = manifest_df[['norm_easting', 'norm_northing']].values
    image_ids = manifest_df['image_id'].values

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    logger.info(f"  Resolution {resolution}: {n_clusters} clusters")

    records = []
    rng = np.random.RandomState(config.RANDOM_STATE)

    for ci, cluster_id in enumerate(unique_labels):
        member_mask = labels == cluster_id
        member_indices = np.where(member_mask)[0]
        cluster_size = len(member_indices)

        # Medoid computation
        medoid_idx, sim_stats, _, _ = _compute_medoid(
            pca_data, member_indices,
            max_sample=config.MEDOID_MAX_SAMPLE
        )

        # Nearest to medoid
        top_nearest = _get_nearest_to_medoid(
            pca_data, medoid_idx, member_indices,
            top_n=config.MEDOID_TOP_NEAREST
        )

        # Random sample
        if cluster_size <= config.MEDOID_RANDOM_SAMPLE:
            random_ids = member_indices.tolist()
        else:
            random_ids = rng.choice(member_indices,
                                     config.MEDOID_RANDOM_SAMPLE,
                                     replace=False).tolist()

        # Spatial sample
        spatial_ids = _spatial_sample(coords, member_indices,
                                       n_sample=config.MEDOID_SPATIAL_SAMPLE)

        record = {
            'resolution': resolution,
            'cluster_id': int(cluster_id),
            'cluster_size': cluster_size,
            'medoid_image_id': image_ids[medoid_idx],
            'medoid_global_index': int(medoid_idx),
            'top_nearest_image_ids': ','.join(image_ids[i] for i in top_nearest),
            'random_sample_image_ids': ','.join(image_ids[i] for i in random_ids[:100]),
            'spatial_sample_image_ids': ','.join(image_ids[i] for i in spatial_ids[:100]),
        }
        record.update(sim_stats)
        records.append(record)

        if (ci + 1) % 100 == 0:
            logger.info(f"    {ci + 1}/{n_clusters} clusters processed")

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    logger.info(f"    Saved: {output_path}")


def run():
    logger.info("=" * 60)
    logger.info("STEP 8: Medoid & Representative Image Extraction")
    logger.info("=" * 60)

    t0 = time.time()

    pca_dim = config.get_selected_pca_dim()
    k = config.KNN_DEFAULT_K
    graph_type = config.GRAPH_DEFAULT_TYPE

    # Load PCA data
    logger.info(f"Loading PCA-{pca_dim} embeddings...")
    pca_data = config.load_memmap(
        config.pca_path(pca_dim),
        config.pca_shape_path(pca_dim),
        dtype='float32', mode='r'
    )
    logger.info(f"  Shape: {pca_data.shape}")

    # Load manifest
    manifest_df = pd.read_parquet(config.manifest_path())

    # Process each resolution using the reference seed
    seed = REF_SEED
    for res in config.LEIDEN_RESOLUTIONS:
        logger.info(f"\n--- Resolution {res} ---")
        label_path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
        if not os.path.exists(label_path):
            logger.warning(f"  Labels not found for res={res}, seed={seed}")
            continue

        labels = np.load(label_path)
        _process_resolution(labels, pca_data, manifest_df, res)

    elapsed = time.time() - t0
    logger.info(f"\nMedoid extraction complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    run()
