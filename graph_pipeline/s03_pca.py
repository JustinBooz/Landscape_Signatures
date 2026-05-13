"""
Step 03: PCA Compression & Neighbor Preservation Diagnostics
==============================================================
Runs IncrementalPCA at 128/256/512 dimensions on normalized embeddings.
Then runs neighbor preservation diagnostics to select the optimal dimension.

Outputs:
  - pca_{128,256,512}.mmap + shape files
  - pca_variance_report.csv
  - pca_neighbor_preservation.csv
  - pca_selected_dim.txt
"""

import os
import sys
import json
import time
import pickle
import gc
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s03_pca")


def _load_normed_embeddings():
    """Load the normalized embedding memmap."""
    return config.load_memmap(
        config.normed_embeddings_path(),
        config.normed_embeddings_shape_path(),
        dtype='float16', mode='r'
    )


def _run_pca_for_dim(dim, emb_mmap):
    """Fit IncrementalPCA and transform all data for a given dimension."""
    pca_mmap_path = config.pca_path(dim)
    pca_shape_path = config.pca_shape_path(dim)
    pca_model_file = config.pca_model_path(dim)

    # Check if already done
    if os.path.exists(pca_mmap_path) and os.path.exists(pca_shape_path):
        with open(pca_shape_path, 'r') as f:
            meta = json.load(f)
        if meta.get('complete', False):
            logger.info(f"  [CHECKPOINT] PCA-{dim} already complete")
            return meta.get('explained_variance_ratio_sum', None)

    n = emb_mmap.shape[0]
    batch_size = config.PCA_BATCH_SIZE
    logger.info(f"  PCA-{dim}: fitting on {n:,} × {config.EMBEDDING_DIM}D")
    t0 = time.time()

    # ---- Phase 1: Fit ----
    n_seen = 0
    if os.path.exists(pca_model_file):
        with open(pca_model_file, 'rb') as f:
            ipca = pickle.load(f)
        n_seen = getattr(ipca, 'n_samples_seen_', 0)
        
        if n_seen >= n:
            logger.info(f"    PCA-{dim} fit already complete ({n_seen:,} samples)")
        else:
            logger.info(f"    Resuming PCA-{dim} fit from {n_seen:,}/{n:,} samples")
    else:
        ipca = IncrementalPCA(n_components=dim)

    if n_seen < n:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            
            # Skip batches already seen in a previous partial run
            if end <= n_seen:
                continue
                
            batch = emb_mmap[start:end].astype(np.float32)
            ipca.partial_fit(batch)
            
            if (start // batch_size) % 20 == 0:
                logger.info(f"    PCA-{dim} fit: {end:,}/{n:,}")
                # Save intermediate checkpoint
                with open(pca_model_file + ".tmp", 'wb') as f:
                    pickle.dump(ipca, f)
                os.replace(pca_model_file + ".tmp", pca_model_file)

        explained = ipca.explained_variance_ratio_.sum()
        logger.info(f"    PCA-{dim} fit done: {explained:.4f} variance explained "
                     f"({time.time() - t0:.0f}s)")

        # Save model
        with open(pca_model_file, 'wb') as f:
            pickle.dump(ipca, f)
        logger.info(f"    PCA model saved: {pca_model_file}")

    explained = ipca.explained_variance_ratio_.sum()

    # ---- Phase 2: Transform ----
    shape_out = (n, dim)
    logger.info(f"    PCA-{dim} transform: {n:,} → {shape_out}")

    pca_mmap = np.memmap(pca_mmap_path, dtype='float32', mode='w+',
                         shape=shape_out)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = emb_mmap[start:end].astype(np.float32)
        pca_mmap[start:end] = ipca.transform(batch)
        if (start // batch_size) % 20 == 0:
            logger.info(f"    PCA-{dim} transform: {end:,}/{n:,}")

    pca_mmap.flush()
    del pca_mmap

    # Save shape metadata
    with open(pca_shape_path, 'w') as f:
        json.dump({
            'shape': list(shape_out),
            'dtype': 'float32',
            'complete': True,
            'explained_variance_ratio_sum': float(explained),
        }, f)

    elapsed = time.time() - t0
    logger.info(f"    PCA-{dim} complete in {elapsed / 60:.1f} min "
                f"(explained={explained:.4f})")

    return explained


def _run_neighbor_preservation(emb_mmap):
    """Compare kNN overlap between original space and PCA compressions."""
    output_path = config.pca_neighbor_preservation_path()
    if os.path.exists(output_path):
        logger.info(f"[CHECKPOINT] Neighbor preservation already exists")
        return pd.read_csv(output_path)

    import faiss

    n_sample = min(config.PCA_NEIGHBOR_SAMPLE_SIZE, emb_mmap.shape[0])
    k_eval = 100  # compute neighbors up to k=100

    logger.info(f"Neighbor preservation diagnostics: sampling {n_sample:,}")
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(emb_mmap.shape[0], n_sample, replace=False)
    sample_idx.sort()

    # Get original 4096D sample vectors
    logger.info("  Loading original 4096D sample...")
    orig_sample = emb_mmap[sample_idx].astype(np.float32)

    # Exact kNN in original space
    logger.info("  Computing exact kNN in 4096D space...")
    index_orig = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index_orig.add(orig_sample)
    _, orig_neighbors = index_orig.search(orig_sample, k_eval + 1)
    # Remove self (first column)
    orig_neighbors = orig_neighbors[:, 1:]  # [n_sample, k_eval]
    del index_orig
    gc.collect()

    results = []
    for dim in config.PCA_DIMS:
        pca_shape_path = config.pca_shape_path(dim)
        if not os.path.exists(pca_shape_path):
            logger.warning(f"  PCA-{dim} not yet computed, skipping")
            continue

        # Load PCA embeddings for sample
        pca_mmap = config.load_memmap(
            config.pca_path(dim), pca_shape_path,
            dtype='float32', mode='r'
        )
        pca_sample = pca_mmap[sample_idx].copy()

        # Exact kNN in PCA space
        logger.info(f"  Computing exact kNN in PCA-{dim} space...")
        index_pca = faiss.IndexFlatIP(dim)
        index_pca.add(pca_sample)
        _, pca_neighbors = index_pca.search(pca_sample, k_eval + 1)
        pca_neighbors = pca_neighbors[:, 1:]  # remove self
        del index_pca, pca_sample

        # Compute overlap at various k
        for k in [10, 50, 100]:
            orig_set = orig_neighbors[:, :k]
            pca_set = pca_neighbors[:, :k]
            overlaps = []
            for i in range(n_sample):
                ov = len(set(orig_set[i]) & set(pca_set[i]))
                overlaps.append(ov / k)
            mean_overlap = np.mean(overlaps)
            median_overlap = np.median(overlaps)
            results.append({
                'pca_dim': dim,
                'k': k,
                'mean_overlap': mean_overlap,
                'median_overlap': median_overlap,
                'std_overlap': np.std(overlaps),
                'min_overlap': np.min(overlaps),
                'p10_overlap': np.percentile(overlaps, 10),
            })
            logger.info(f"    PCA-{dim} overlap@{k}: mean={mean_overlap:.4f} "
                        f"median={median_overlap:.4f}")

        del pca_neighbors
        gc.collect()

    del orig_sample, orig_neighbors
    gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path}")
    return df


def _select_pca_dim(preservation_df):
    """Select PCA dimension based on neighbor preservation."""
    output_path = config.pca_selected_dim_path()

    # Check PCA-128 recall@50
    row_128_50 = preservation_df[
        (preservation_df['pca_dim'] == 128) &
        (preservation_df['k'] == 50)
    ]

    if len(row_128_50) > 0:
        recall_128 = row_128_50.iloc[0]['mean_overlap']
        if recall_128 >= config.PCA_RECALL_THRESHOLD:
            selected = 128
            logger.info(f"PCA-128 recall@50 = {recall_128:.4f} >= "
                        f"{config.PCA_RECALL_THRESHOLD} → selecting PCA-128")
        else:
            selected = 256
            logger.info(f"PCA-128 recall@50 = {recall_128:.4f} < "
                        f"{config.PCA_RECALL_THRESHOLD} → selecting PCA-256")
    else:
        selected = config.PCA_DEFAULT_DIM
        logger.info(f"No PCA-128 data found → defaulting to PCA-{selected}")

    with open(output_path, 'w') as f:
        f.write(str(selected))
    logger.info(f"Selected PCA dimension: {selected} → {output_path}")
    return selected


def run():
    logger.info("=" * 60)
    logger.info("STEP 3: PCA Compression & Neighbor Preservation")
    logger.info("=" * 60)

    t0 = time.time()

    # Load normalized embeddings
    emb_mmap = _load_normed_embeddings()
    logger.info(f"Loaded normalized embeddings: {emb_mmap.shape}")

    # ---- Phase 1: Run PCA for each dimension ----
    variance_records = []
    for dim in config.PCA_DIMS:
        logger.info(f"\n--- PCA-{dim} ---")
        explained = _run_pca_for_dim(dim, emb_mmap)
        if explained is not None:
            variance_records.append({
                'pca_dim': dim,
                'explained_variance_ratio_sum': explained
            })

    # Save variance report
    if variance_records:
        var_df = pd.DataFrame(variance_records)
        var_df.to_csv(config.pca_variance_report_path(), index=False)
        logger.info(f"Variance report saved: {config.pca_variance_report_path()}")

    gc.collect()

    # ---- Phase 2: Neighbor preservation ----
    logger.info("\n--- Neighbor Preservation Diagnostics ---")
    pres_df = _run_neighbor_preservation(emb_mmap)

    # ---- Phase 3: Select PCA dimension ----
    selected = _select_pca_dim(pres_df)

    elapsed = time.time() - t0
    logger.info(f"\nPCA step complete in {elapsed / 60:.1f} min")
    logger.info(f"Selected dimension: {selected}")

    return selected


if __name__ == "__main__":
    run()
