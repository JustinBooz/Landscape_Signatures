"""
Step 04b: Microclustering
================================
Performs FAISS spherical k-means on PCA-256 embeddings to compute 50,000 microclusters.

Outputs:
  - microcluster_centroids.npy
  - microcluster_assignments.parquet
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s04b_microclusters")

def run():
    logger.info("=" * 60)
    logger.info("STEP 4b: Microclustering")
    logger.info("=" * 60)

    t0 = time.time()
    
    centroids_path = config.microcluster_centroids_path()
    assignments_path = config.microcluster_assignments_path()

    if os.path.exists(centroids_path) and os.path.exists(assignments_path):
        logger.info("[CHECKPOINT] Microclusters already computed.")
        return

    pca_dim = config.get_selected_pca_dim()
    logger.info(f"Using PCA-{pca_dim} embeddings for microclustering.")

    pca_mmap = config.load_memmap(
        config.pca_path(pca_dim),
        config.pca_shape_path(pca_dim),
        dtype='float32', mode='r'
    )
    n = pca_mmap.shape[0]
    
    # K-means clustering on embeddings
    k = getattr(config, 'MICROCLUSTER_K', 50000)
    logger.info(f"Loading {n:,} embeddings into RAM for k-means (k={k})...")
    data = np.array(pca_mmap, dtype=np.float32)
    
    # Spherical k-means requires L2-normalized data
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    data = data / norms

    logger.info("Running FAISS spherical K-means...")
    kmeans = faiss.Kmeans(d=pca_dim, k=k, spherical=True, niter=20, verbose=True, nredo=1)
    
    # Use GPU if available
    ngpus = faiss.get_num_gpus()
    if ngpus > 0:
        logger.info(f"Using {ngpus} GPUs for k-means.")
        kmeans.gpu = True
    else:
        logger.info("Using CPU for k-means.")
        
    kmeans.train(data)

    centroids = kmeans.centroids
    logger.info(f"Found {len(centroids)} centroids. Computing final assignments...")

    # Compute assignments (closest centroid)
    # Using the inner product index since spherical means normalized
    index = faiss.IndexFlatIP(pca_dim)
    if ngpus > 0:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        
    index.add(centroids)
    
    batch_size = 500000
    all_I = np.empty((n,), dtype=np.int64)
    all_D = np.empty((n,), dtype=np.float32)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = data[start:end]
        D, I = index.search(batch, 1)
        all_D[start:end] = D.flatten()
        all_I[start:end] = I.flatten()

    # Save outputs
    logger.info("Saving results...")
    np.save(centroids_path + ".tmp", centroids)
    os.rename(centroids_path + ".tmp", centroids_path)

    df = pd.DataFrame({
        'global_index': np.arange(n),
        'microcluster_id': all_I,
        'distance_to_centroid': 1.0 - all_D  # Convert cosine similarity to distance
    })
    
    df.to_parquet(assignments_path + ".tmp")
    os.rename(assignments_path + ".tmp", assignments_path)
    
    empty_clusters = k - len(np.unique(all_I))
    logger.info(f"Microclustering complete: {empty_clusters} empty clusters. Time: {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    run()
