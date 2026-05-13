"""
Step 13: Calibrated Negative Controls
=====================================
Computes empirical p-values for spatial coherence and geodata purity 
by comparing true cluster scores against a null distribution generated 
by randomly permuting cluster labels (preserving size distribution).

Outputs:
  - negative_control_pvalues.parquet
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from leiden_parallel import REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED

logger = config.setup_logging("s13_negative_controls")

def get_labels(pca_dim, k, graph_type, res, seed):
    path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
    if not os.path.exists(path):
        return None
    return np.load(path)

SIZE_BINS = np.array([0, 10, 30, 100, 300, 1_000, 3_000, 10_000,
                      30_000, 100_000, 300_000, np.inf])


def _size_bin(n):
    """Return the index of the size bin a cluster of size n falls into."""
    return int(np.searchsorted(SIZE_BINS, n, side='right') - 1)


def compute_null_iteration(df, rng, all_clusters):
    """Run one null permutation. Returns dicts mapping size_bin -> [scores].

    Permuting labels preserves the per-cluster size distribution, so each
    null cluster of size N yields a score directly comparable to a true cluster
    of similar size (in the same bin). This avoids the original bug where a
    100-member true cluster was compared against null scores dominated by
    very-large pseudo-clusters.
    """
    shuffled_labels = rng.permutation(df['cluster_id'].values)
    df = df.assign(null_cluster_id=shuffled_labels)

    # Sample a subset of pseudo-clusters per iteration to save time
    sampled_clusters = rng.choice(all_clusters, size=min(100, len(all_clusters)),
                                  replace=False)
    sub = df[df['null_cluster_id'].isin(sampled_clusters)]
    groups = sub.groupby('null_cluster_id', sort=False)

    coh_by_bin = {}
    pur_by_bin = {}
    for c, c_df in groups:
        n_members = len(c_df)
        if n_members < 2:
            continue
        b = _size_bin(n_members)

        if n_members > 1000:
            sample_df = c_df.sample(n=1000, random_state=rng.integers(0, 2**31 - 1))
        else:
            sample_df = c_df

        sample_pts = sample_df[['lv95_easting', 'lv95_northing']].values
        coh = float(np.median(distance.pdist(sample_pts, 'euclidean')))

        cats = c_df['landcover_class'].astype(str) + " / " + c_df['nearest_building_category'].astype(str)
        val_counts = cats.value_counts()
        pur = val_counts.iloc[0] / len(cats) if len(val_counts) > 0 else 0.0

        coh_by_bin.setdefault(b, []).append(coh)
        pur_by_bin.setdefault(b, []).append(pur)

    return coh_by_bin, pur_by_bin

def run():
    logger.info("=" * 60)
    logger.info("STEP 13: Calibrated Negative Controls")
    logger.info("=" * 60)

    t0 = time.time()
    
    out_path = config.negative_control_pvalues_path()
    if os.path.exists(out_path):
        logger.info(f"[CHECKPOINT] Negative controls already computed: {out_path}")
        return

    ref_labels = get_labels(REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED)
    if ref_labels is None:
        logger.error("Reference clustering not found.")
        return

    logger.info("Loading data for null modeling...")
    manifest = pd.read_parquet(config.manifest_path(), columns=['lv95_easting', 'lv95_northing', 'global_index'])
    geodata = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "geodata_enriched.parquet"), 
                              columns=['global_index', 'landcover_class', 'nearest_building_category'])
    
    df = pd.merge(manifest, geodata, on='global_index', how='inner')
    df['cluster_id'] = ref_labels[df['global_index'].values]

    all_clusters = df['cluster_id'].unique()

    rng = np.random.default_rng(config.RANDOM_STATE)
    n_iterations = 100
    logger.info(f"Running {n_iterations} size-stratified null permutations...")

    null_coh_by_bin = {}
    null_pur_by_bin = {}

    for i in range(n_iterations):
        coh_b, pur_b = compute_null_iteration(df, rng, all_clusters)
        for b, vs in coh_b.items():
            null_coh_by_bin.setdefault(b, []).extend(vs)
        for b, vs in pur_b.items():
            null_pur_by_bin.setdefault(b, []).extend(vs)
        if (i+1) % 10 == 0:
            total_coh = sum(len(v) for v in null_coh_by_bin.values())
            logger.info(f"  {i+1}/{n_iterations} permutations complete "
                        f"({total_coh:,} null scores)")

    null_coh_by_bin = {b: np.asarray(v) for b, v in null_coh_by_bin.items()}
    null_pur_by_bin = {b: np.asarray(v) for b, v in null_pur_by_bin.items()}

    # Load true cluster scores (need cluster sizes too)
    val_path = config.cluster_spatial_validation_path()
    if not os.path.exists(val_path):
        logger.error("Spatial validation scores not found. Run s11_spatial.py first.")
        return

    true_df = pd.read_parquet(val_path)

    pvalues = []
    for _, row in true_df.iterrows():
        c = int(row['cluster_id'])
        true_coh = row['spatial_coherence_score']
        true_pur = row['geodata_purity_score']
        size = int(row['cluster_size']) if 'cluster_size' in row else int(np.sum(df['cluster_id'] == c))
        b = _size_bin(size)

        # Fall back to the global pool if a size-bin has too few samples.
        coh_null = null_coh_by_bin.get(b, np.array([]))
        if len(coh_null) < 20:
            coh_null = np.concatenate(list(null_coh_by_bin.values())) if null_coh_by_bin else np.array([])
        pur_null = null_pur_by_bin.get(b, np.array([]))
        if len(pur_null) < 20:
            pur_null = np.concatenate(list(null_pur_by_bin.values())) if null_pur_by_bin else np.array([])

        # Empirical p-values (with pseudo-count).
        # spatial coherence: smaller is better → p = P(null ≤ true)
        # geodata purity:    larger is better → p = P(null ≥ true)
        p_coh = (np.sum(coh_null <= true_coh) + 1) / (len(coh_null) + 1) if len(coh_null) else 1.0
        p_pur = (np.sum(pur_null >= true_pur) + 1) / (len(pur_null) + 1) if len(pur_null) else 1.0
        p_combined = max(p_coh, p_pur)

        pvalues.append({
            'cluster_id': c,
            'cluster_size': size,
            'size_bin': b,
            'pvalue_spatial_coherence': p_coh,
            'pvalue_geodata_purity': p_pur,
            'negative_control_pvalue': p_combined,
            'n_null_samples_in_bin': int(len(coh_null)),
        })

    res_df = pd.DataFrame(pvalues)
    res_df.to_parquet(out_path, index=False)
    logger.info(f"Saved negative control p-values to {out_path}")

    # Diagnostic: report null distribution percentiles per size bin
    logger.info("=" * 60)
    logger.info("NULL DISTRIBUTION PERCENTILES BY SIZE BIN:")
    for b in sorted(null_coh_by_bin):
        lo, hi = SIZE_BINS[b], SIZE_BINS[b + 1]
        coh_p1 = np.percentile(null_coh_by_bin[b], 1)
        pur_p99 = np.percentile(null_pur_by_bin.get(b, [0]), 99) if b in null_pur_by_bin else float('nan')
        logger.info(f"  size [{lo:.0f}, {hi:.0f}): coh_p1={coh_p1:.1f} m, "
                    f"pur_p99={pur_p99:.4f}, n={len(null_coh_by_bin[b])}")
    logger.info("=" * 60)

    logger.info(f"Time: {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    run()
