"""
Step 09b: Auxiliary Comparators (Microcluster-based)
===================================================
Runs HDBSCAN and FINCH on the 50,000 microcluster centroids.
Gracefully skips if dependencies are missing.
Lifts the cluster labels from centroids back to the 7.2M images.

Outputs:
  - hdbscan_labels.parquet
  - finch_labels.parquet
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from leiden_parallel import REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED

logger = config.setup_logging("s09b_aux_comparators")

def lift_labels(micro_labels, assignments):
    """Lift 50k centroid labels back to 7.2M points."""
    return micro_labels[assignments]

def compute_agreement(ref_labels, comp_labels):
    mask = comp_labels >= 0  # ignore noise for ARI/NMI
    if mask.sum() == 0:
        return 0.0, 0.0
    ari = adjusted_rand_score(ref_labels[mask], comp_labels[mask])
    nmi = normalized_mutual_info_score(ref_labels[mask], comp_labels[mask])
    return ari, nmi

def run_hdbscan(centroids, assignments, ref_labels, out_path):
    try:
        import hdbscan
    except ImportError:
        logger.warning("HDBSCAN not installed. Skipping HDBSCAN comparator.")
        return

    if os.path.exists(out_path):
        logger.info("[CHECKPOINT] HDBSCAN comparator already computed.")
        return

    logger.info("Running HDBSCAN on microcluster centroids...")
    t0 = time.time()
    
    # Using typical parameters for 50k centroids
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10, metric='euclidean', core_dist_n_jobs=4)
    centroid_labels = clusterer.fit_predict(centroids)
    
    logger.info(f"HDBSCAN finished in {time.time()-t0:.1f}s")
    
    # Lift labels
    full_labels = lift_labels(centroid_labels, assignments)
    
    # Save
    df = pd.DataFrame({'hdbscan_label': full_labels})
    df.to_parquet(out_path, index=False)
    
    # Agreement
    ari, nmi = compute_agreement(ref_labels, full_labels)
    noise_pct = (full_labels == -1).mean() * 100
    n_clusters = len(set(full_labels)) - (1 if -1 in full_labels else 0)
    
    logger.info(f"HDBSCAN Summary: {n_clusters} clusters, {noise_pct:.1f}% noise. "
                f"Agreement with Reference: ARI={ari:.4f}, NMI={nmi:.4f}")


def run_finch(centroids, assignments, ref_labels, out_path):
    try:
        from finch import FINCH
    except ImportError:
        logger.warning("FINCH not installed. Skipping FINCH comparator.")
        return

    if os.path.exists(out_path):
        logger.info("[CHECKPOINT] FINCH comparator already computed.")
        return

    logger.info("Running FINCH on microcluster centroids...")
    t0 = time.time()
    
    c, num_clust, req_c = FINCH(centroids)
    
    logger.info(f"FINCH finished in {time.time()-t0:.1f}s. Hierarchy levels: {c.shape[1]}")
    
    df_dict = {}
    for level in range(c.shape[1]):
        centroid_labels = c[:, level]
        full_labels = lift_labels(centroid_labels, assignments)
        df_dict[f'finch_label_L{level+1}'] = full_labels
        
        ari, nmi = compute_agreement(ref_labels, full_labels)
        n_clusters = len(set(full_labels))
        logger.info(f"FINCH L{level+1}: {n_clusters} clusters. "
                    f"Agreement with Reference: ARI={ari:.4f}, NMI={nmi:.4f}")
        
    df = pd.DataFrame(df_dict)
    df.to_parquet(out_path, index=False)

def run():
    logger.info("=" * 60)
    logger.info("STEP 9b: Auxiliary Comparators (Microcluster-based)")
    logger.info("=" * 60)

    centroids_path = config.microcluster_centroids_path()
    assignments_path = config.microcluster_assignments_path()

    if not os.path.exists(centroids_path) or not os.path.exists(assignments_path):
        logger.error("Microcluster outputs not found. Run s04b_microclusters.py first.")
        return

    logger.info("Loading microclusters...")
    centroids = np.load(centroids_path)
    assignments_df = pd.read_parquet(assignments_path)
    assignments = assignments_df['microcluster_id'].values

    # Load reference labels for agreement metrics
    ref_labels_path = config.leiden_labels_path(REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED)
    if not os.path.exists(ref_labels_path):
        logger.error("Reference Leiden clustering not found. Cannot compute agreement.")
        return
    ref_labels = np.load(ref_labels_path)

    # Run HDBSCAN
    hdb_path = config.hdbscan_labels_path()
    run_hdbscan(centroids, assignments, ref_labels, hdb_path)

    # Run FINCH
    finch_path = config.finch_labels_path()
    run_finch(centroids, assignments, ref_labels, finch_path)

    logger.info("Auxiliary comparators complete.")

if __name__ == "__main__":
    run()
