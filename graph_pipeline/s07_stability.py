"""
Step 07: Cluster Stability Diagnostics
=========================================
Computes ARI and NMI across the four fractional grid perturbation axes:
seed, k, graph, and resolution, always comparing against the reference configuration.

Outputs:
  - stability_summary.parquet
  - stability_summary.md
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

logger = config.setup_logging("s07_stability")

def get_labels(pca_dim, k, graph_type, res, seed):
    path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
    if not os.path.exists(path):
        return None
    return np.load(path)

def generate_markdown_report(df, ref_meta, ref_n, out_path):
    md = [
        "# Stability Diagnostics Summary",
        f"**Reference Configuration:** PCA={REF_PCA}, k={REF_K}, graph={REF_GRAPH}, res={REF_RES}",
        f"**Reference Clusters:** {ref_meta.get('num_clusters', 'N/A')}",
        f"**Reference Singletons:** {ref_meta.get('singleton_count', 'N/A')}",
        ""
    ]

    md.append("## Axis Stability (Mean against Reference)")
    for axis in df['axis'].unique():
        sub = df[df['axis'] == axis]
        md.append(f"### {axis.capitalize()} Perturbations")
        md.append(f"- **Mean ARI**: {sub['ari'].mean():.4f}")
        md.append(f"- **Mean NMI**: {sub['nmi'].mean():.4f}")
        for _, row in sub.iterrows():
            md.append(f"  - vs {row['param_val']}: ARI={row['ari']: .4f}, NMI={row['nmi']: .4f}")
        md.append("")

    md.append("## Structural Flags")
    warnings = []
    if ref_meta.get('singleton_count', 0) > 0.05 * ref_n:
        warnings.append("- **WARNING**: Reference configuration is overfragmented (>5% singletons).")
    if ref_meta.get('percent_in_top10_clusters', 0) > 30.0:
        warnings.append("- **WARNING**: Reference configuration is giant-cluster-dominated (>30% points in top 10).")

    if warnings:
        md.extend(warnings)
    else:
        md.append("- No structural warnings for reference configuration.")

    with open(out_path, 'w') as f:
        f.write("\n".join(md))

def run():
    logger.info("=" * 60)
    logger.info("STEP 7: Cluster Stability Diagnostics")
    logger.info("=" * 60)

    t0 = time.time()

    # Load reference labels
    ref_labels = get_labels(REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED)
    if ref_labels is None:
        logger.error("Reference clustering not found. Cannot compute stability.")
        return

    # Load reference metadata
    ref_meta_path = config.leiden_labels_path(REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED).replace(".npy", ".meta.json")
    import json
    ref_meta = {}
    if os.path.exists(ref_meta_path):
        with open(ref_meta_path, 'r') as f:
            ref_meta = json.load(f)

    records = []

    # 1. Seed stability (compare seed 1,2,3,4 to seed 0)
    for seed in [1, 2, 3, 4]:
        l2 = get_labels(REF_PCA, REF_K, REF_GRAPH, REF_RES, seed)
        if l2 is not None:
            records.append({
                'axis': 'seed', 'param_val': f'seed={seed}',
                'ari': adjusted_rand_score(ref_labels, l2),
                'nmi': normalized_mutual_info_score(ref_labels, l2)
            })

    # 2. k stability (compare k=50, 200 to k=100)
    for k in [50, 200]:
        l2 = get_labels(REF_PCA, k, REF_GRAPH, REF_RES, REF_SEED)
        if l2 is not None:
            records.append({
                'axis': 'k_value', 'param_val': f'k={k}',
                'ari': adjusted_rand_score(ref_labels, l2),
                'nmi': normalized_mutual_info_score(ref_labels, l2)
            })

    # 3. graph stability (compare mutual, mutual_snn_jaccard to snn_jaccard)
    for g in ['mutual', 'mutual_snn_jaccard']:
        l2 = get_labels(REF_PCA, REF_K, g, REF_RES, REF_SEED)
        if l2 is not None:
            records.append({
                'axis': 'graph', 'param_val': f'g={g}',
                'ari': adjusted_rand_score(ref_labels, l2),
                'nmi': normalized_mutual_info_score(ref_labels, l2)
            })

    # 4. resolution stability (compare 0.25, 0.5, 2.0, 4.0 to 1.0)
    for res in [0.25, 0.5, 2.0, 4.0]:
        l2 = get_labels(REF_PCA, REF_K, REF_GRAPH, res, REF_SEED)
        if l2 is not None:
            records.append({
                'axis': 'resolution', 'param_val': f'res={res}',
                'ari': adjusted_rand_score(ref_labels, l2),
                'nmi': normalized_mutual_info_score(ref_labels, l2)
            })

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(os.path.join(config.OUTPUT_DIR, "stability_summary.parquet"), index=False)
        generate_markdown_report(df, ref_meta, len(ref_labels),
                                  os.path.join(config.OUTPUT_DIR, "stability_summary.md"))
        logger.info("Saved stability_summary.parquet and stability_summary.md")
    else:
        logger.warning("No perturbation runs found.")

    logger.info(f"Stability diagnostics complete in {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    run()
