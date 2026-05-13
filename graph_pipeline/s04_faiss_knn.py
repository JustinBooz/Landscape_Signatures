"""
Step 04: FAISS kNN Search + Recall Diagnostics
================================================
Builds FAISS IVF index on PCA embeddings and searches for k-nearest
neighbors at multiple k values. Uses GPU acceleration.

Outputs:
  - neighbors_k{50,100,200}_pca{dim}.npy
  - distances_k{50,100,200}_pca{dim}.npy
  - faiss_recall_diagnostics.csv
"""

import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s04_faiss_knn")


def _build_index(data, pca_dim):
    """Build a FAISS IVFFlat index with inner product (cosine on L2-normed)."""
    n, d = data.shape
    logger.info(f"Building FAISS IVFFlat index: {n:,} × {d}D, "
                f"nlist={config.FAISS_NLIST}")

    # Use GPU if available
    ngpus = faiss.get_num_gpus()
    logger.info(f"  FAISS GPUs available: {ngpus}")

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, config.FAISS_NLIST,
                                faiss.METRIC_INNER_PRODUCT)

    if ngpus > 0:
        logger.info("  Training on GPU...")
        gpu_res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        # Train on a subsample if data is huge
        train_size = min(n, config.FAISS_NLIST * 256)
        rng = np.random.RandomState(config.RANDOM_STATE)
        train_idx = rng.choice(n, train_size, replace=False)
        train_data = data[train_idx].copy()
        gpu_index.train(train_data)

        # Copy trained index back to CPU, add all data
        index = faiss.index_gpu_to_cpu(gpu_index)
        del gpu_index, gpu_res
        gc.collect()
    else:
        train_size = min(n, config.FAISS_NLIST * 256)
        rng = np.random.RandomState(config.RANDOM_STATE)
        train_idx = rng.choice(n, train_size, replace=False)
        train_data = data[train_idx].copy()
        index.train(train_data)

    # Add data in batches
    logger.info(f"  Adding {n:,} vectors to index...")
    add_batch = 500_000
    for start in range(0, n, add_batch):
        end = min(start + add_batch, n)
        index.add(data[start:end].copy())
        if start > 0:
            logger.info(f"    Added {end:,}/{n:,}")

    index.nprobe = config.FAISS_NPROBE
    logger.info(f"  Index ready: ntotal={index.ntotal:,}, nprobe={index.nprobe}")
    return index


def _search_knn(index, data, k, pca_dim):
    """Search for k nearest neighbors. Returns (distances, neighbors)."""
    neighbors_path = config.knn_neighbors_path(k, pca_dim)
    distances_path = config.knn_distances_path(k, pca_dim)

    if os.path.exists(neighbors_path) and os.path.exists(distances_path):
        logger.info(f"  [CHECKPOINT] k={k} already exists")
        return np.load(distances_path), np.load(neighbors_path)

    n = data.shape[0]
    logger.info(f"  Searching k={k} for {n:,} vectors...")
    t0 = time.time()

    # Search on CPU with multi-threading (GPU OOM for full-scale search)
    faiss.omp_set_num_threads(24)
    search_batch = 50_000
    all_distances = np.empty((n, k), dtype=np.float32)
    all_neighbors = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, search_batch):
        end = min(start + search_batch, n)
        batch = data[start:end].copy()
        D, I = index.search(batch, k)
        all_distances[start:end] = D
        all_neighbors[start:end] = I
        if (start // search_batch) % 20 == 0:
            elapsed_so_far = time.time() - t0
            rate = end / elapsed_so_far if elapsed_so_far > 0 else 0
            logger.info(f"    k={k} search: {end:,}/{n:,} "
                        f"({elapsed_so_far:.0f}s, {rate:.0f} vec/s)")

    elapsed = time.time() - t0
    logger.info(f"    k={k} search done in {elapsed:.0f}s")

    # Save atomically
    tmp_n = neighbors_path + ".tmp.npy"
    tmp_d = distances_path + ".tmp.npy"
    np.save(tmp_n, all_neighbors)
    os.rename(tmp_n, neighbors_path)
    np.save(tmp_d, all_distances)
    os.rename(tmp_d, distances_path)

    logger.info(f"    Saved: {neighbors_path}, {distances_path}")

    return all_distances, all_neighbors


def _recall_diagnostics(data, pca_dim):
    """Compare FAISS approximate neighbors against exact neighbors on a sample."""
    output_path = config.faiss_recall_path()
    if os.path.exists(output_path):
        logger.info(f"[CHECKPOINT] FAISS recall diagnostics already exist")
        return pd.read_csv(output_path)

    n_sample = min(config.FAISS_RECALL_SAMPLE_SIZE, data.shape[0])
    k_max = max(config.KNN_K_VALUES)
    d = data.shape[1]

    logger.info(f"FAISS recall diagnostics: {n_sample:,} samples, k_max={k_max}")
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(data.shape[0], n_sample, replace=False)
    sample_data = data[sample_idx].copy()

    # Exact neighbors on sample
    logger.info("  Computing exact neighbors on sample...")
    exact_index = faiss.IndexFlatIP(d)
    exact_index.add(sample_data)
    _, exact_neighbors = exact_index.search(sample_data, k_max + 1)
    exact_neighbors = exact_neighbors[:, 1:]  # remove self
    del exact_index

    # Recall test: build an IVF on the sample and compare against exact (on the
    # sample). NOTE: this is a sample-on-sample proxy and does not measure the
    # production index's recall against full-scale data — treat as indicative.
    results = []
    logger.info("  Building FAISS IVF on sample for recall test...")
    quantizer = faiss.IndexFlatIP(d)
    nlist = min(256, n_sample // 40)
    ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist,
                                    faiss.METRIC_INNER_PRODUCT)

    ngpus = faiss.get_num_gpus()
    if ngpus > 0:
        gpu_res = faiss.StandardGpuResources()
        gpu_ivf = faiss.index_cpu_to_gpu(gpu_res, 0, ivf_index)
        gpu_ivf.train(sample_data)
        ivf_index = faiss.index_gpu_to_cpu(gpu_ivf)
        del gpu_ivf, gpu_res
    else:
        ivf_index.train(sample_data)

    ivf_index.add(sample_data)
    ivf_index.nprobe = config.FAISS_NPROBE

    _, approx_neighbors = ivf_index.search(sample_data, k_max + 1)
    approx_neighbors = approx_neighbors[:, 1:]  # remove self
    del ivf_index

    for k in [10, 50, 100]:
        exact_k = exact_neighbors[:, :k]
        approx_k = approx_neighbors[:, :k]
        recalls = []
        for i in range(n_sample):
            ov = len(set(exact_k[i]) & set(approx_k[i]))
            recalls.append(ov / k)
        mean_recall = np.mean(recalls)
        results.append({
            'pca_dim': pca_dim,
            'k': k,
            'nlist': nlist,
            'nprobe': config.FAISS_NPROBE,
            'mean_recall': mean_recall,
            'median_recall': np.median(recalls),
            'min_recall': np.min(recalls),
            'p10_recall': np.percentile(recalls, 10),
        })
        logger.info(f"    FAISS recall@{k}: mean={mean_recall:.4f}")

    del sample_data, exact_neighbors, approx_neighbors
    gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path}")
    return df


def run():
    logger.info("=" * 60)
    logger.info("STEP 4: FAISS kNN Search")
    logger.info("=" * 60)

    t0 = time.time()

    pca_dim = config.get_selected_pca_dim()
    logger.info(f"Using PCA-{pca_dim}")

    # Load PCA embeddings
    pca_mmap = config.load_memmap(
        config.pca_path(pca_dim),
        config.pca_shape_path(pca_dim),
        dtype='float32', mode='r'
    )
    logger.info(f"Loaded PCA embeddings: {pca_mmap.shape}")

    # We need contiguous float32 data for FAISS
    # For 6.9M × 256 float32 = ~7 GB — fits in RAM
    logger.info("Loading PCA data into contiguous array...")
    data = np.array(pca_mmap, dtype=np.float32)
    logger.info(f"  Data shape: {data.shape}, size: {data.nbytes / 1e9:.1f} GB")

    # L2-normalize the PCA embeddings for inner product = cosine similarity
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    data = data / norms
    del norms

    # Build index
    index = _build_index(data, pca_dim)

    # Search for each k
    for k in config.KNN_K_VALUES:
        logger.info(f"\n--- kNN k={k} ---")
        _search_knn(index, data, k, pca_dim)

    del index
    gc.collect()

    # Recall diagnostics
    logger.info("\n--- FAISS Recall Diagnostics ---")
    _recall_diagnostics(data, pca_dim)

    del data
    gc.collect()

    elapsed = time.time() - t0
    logger.info(f"\nFAISS kNN step complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    run()
