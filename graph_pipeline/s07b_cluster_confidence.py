"""
Step 07b: Per-Cluster Confidence
================================
For each cluster in the reference clustering, computes the best-matching cluster 
Jaccard against every perturbation run. Aggregates into seed, k, graph, and resolution stability.

Outputs:
  - cluster_confidence.parquet
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from leiden_parallel import REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED

logger = config.setup_logging("s07b_cluster_confidence")

def get_labels(pca_dim, k, graph_type, res, seed):
    path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
    if not os.path.exists(path):
        return None
    return np.load(path)

def compute_best_jaccard(ref_labels, pert_labels, ref_clusters):
    """Compute best matching Jaccard for each cluster in ref_clusters against pert_labels."""
    # Build dataframe of assignments
    df = pd.DataFrame({'ref': ref_labels, 'pert': pert_labels})
    
    # Cross-tabulate intersection sizes
    intersection = pd.crosstab(df['ref'], df['pert'])
    
    ref_sizes = df.groupby('ref').size()
    pert_sizes = df.groupby('pert').size()
    
    jaccards = {}
    for c_ref in ref_clusters:
        if c_ref not in intersection.index:
            jaccards[c_ref] = 0.0
            continue
            
        # Get intersection row for this reference cluster
        inter_row = intersection.loc[c_ref]
        # Only check non-zero intersections
        non_zero = inter_row[inter_row > 0]
        
        if len(non_zero) == 0:
            jaccards[c_ref] = 0.0
            continue
            
        best_j = 0.0
        for c_pert, inter_size in non_zero.items():
            union_size = ref_sizes[c_ref] + pert_sizes[c_pert] - inter_size
            j_sim = inter_size / union_size
            if j_sim > best_j:
                best_j = j_sim
        
        jaccards[c_ref] = best_j
        
    return jaccards

def run():
    logger.info("=" * 60)
    logger.info("STEP 7b: Per-Cluster Confidence")
    logger.info("=" * 60)

    t0 = time.time()

    # Load reference labels
    ref_labels = get_labels(REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED)
    if ref_labels is None:
        logger.error("Reference clustering not found. Cannot compute confidence.")
        return

    ref_clusters = np.unique(ref_labels)
    n_clusters = len(ref_clusters)
    logger.info(f"Computing confidence for {n_clusters} reference clusters.")

    # Data structures to accumulate scores
    seed_scores = {c: [] for c in ref_clusters}
    k_scores = {c: [] for c in ref_clusters}
    graph_scores = {c: [] for c in ref_clusters}
    res_scores = {c: [] for c in ref_clusters}

    # 1. Seed stability
    for seed in [1, 2, 3, 4]:
        l2 = get_labels(REF_PCA, REF_K, REF_GRAPH, REF_RES, seed)
        if l2 is not None:
            j_dict = compute_best_jaccard(ref_labels, l2, ref_clusters)
            for c in ref_clusters:
                seed_scores[c].append(j_dict[c])

    # 2. k stability
    for k in [50, 200]:
        l2 = get_labels(REF_PCA, k, REF_GRAPH, REF_RES, REF_SEED)
        if l2 is not None:
            j_dict = compute_best_jaccard(ref_labels, l2, ref_clusters)
            for c in ref_clusters:
                k_scores[c].append(j_dict[c])

    # 3. graph stability
    for g in ['mutual', 'mutual_snn_jaccard']:
        l2 = get_labels(REF_PCA, REF_K, g, REF_RES, REF_SEED)
        if l2 is not None:
            j_dict = compute_best_jaccard(ref_labels, l2, ref_clusters)
            for c in ref_clusters:
                graph_scores[c].append(j_dict[c])

    # 4. resolution stability
    for res in [0.25, 0.5, 2.0, 4.0]:
        l2 = get_labels(REF_PCA, REF_K, REF_GRAPH, res, REF_SEED)
        if l2 is not None:
            j_dict = compute_best_jaccard(ref_labels, l2, ref_clusters)
            for c in ref_clusters:
                res_scores[c].append(j_dict[c])

    # Aggregate
    records = []
    for c in ref_clusters:
        s_seed = np.mean(seed_scores[c]) if seed_scores[c] else 0.0
        s_k = np.mean(k_scores[c]) if k_scores[c] else 0.0
        s_graph = np.mean(graph_scores[c]) if graph_scores[c] else 0.0
        s_res = np.mean(res_scores[c]) if res_scores[c] else 0.0
        
        # Weighted mean (giving equal weight to the 4 axes)
        overall = (s_seed + s_k + s_graph + s_res) / 4.0
        
        records.append({
            'cluster_id': c,
            'seed_stability': float(s_seed),
            'k_stability': float(s_k),
            'graph_stability': float(s_graph),
            'resolution_stability': float(s_res),
            'overall_cluster_confidence': float(overall)
        })

    df = pd.DataFrame(records)
    out_path = config.cluster_confidence_path()
    df.to_parquet(out_path, index=False)
    
    logger.info(f"Saved cluster confidence scores: {out_path}")
    logger.info(f"Time: {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    run()
