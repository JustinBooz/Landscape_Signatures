"""
Step 06: Leiden Community Detection
=====================================
Runs Leiden clustering at multiple resolutions and seeds on the
mutual kNN graph. Uses igraph + leidenalg.

Outputs:
  - leiden_pca{dim}_k{k}_{type}_res{res}_seed{seed}.npy (per-run labels)
  - cluster_labels_leiden.parquet (consolidated)
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from scipy import sparse
import igraph as ig
import leidenalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s06_leiden")


def _sparse_to_igraph(graph_sparse):
    """Convert scipy sparse CSR to igraph Graph with weights."""
    logger.info("  Converting sparse matrix to igraph...")
    t0 = time.time()

    # Get upper triangle to avoid duplicate edges
    upper = sparse.triu(graph_sparse, k=1, format='coo')
    n = graph_sparse.shape[0]

    # Build edge array in a single C call rather than a Python list of tuples
    edges = np.column_stack((upper.row, upper.col))
    g = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    g.es['weight'] = upper.data.tolist()

    elapsed = time.time() - t0
    logger.info(f"    igraph: {g.vcount():,} vertices, {g.ecount():,} edges "
                f"({elapsed:.0f}s)")
    return g


def _run_leiden_single(g, resolution, seed, pca_dim, k, graph_type):
    """Run a single Leiden partition and return labels."""
    label_path = config.leiden_labels_path(pca_dim, k, graph_type,
                                            resolution, seed)
    if os.path.exists(label_path):
        logger.info(f"    [CHECKPOINT] res={resolution}, seed={seed} exists")
        return np.load(label_path)

    logger.info(f"    Leiden: res={resolution}, seed={seed}")
    t0 = time.time()

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights='weight',
        seed=seed,
        n_iterations=-1,  # iterate until convergence
    )

    labels = np.array(partition.membership, dtype=np.int32)
    n_clusters = len(set(labels))
    modularity = partition.modularity

    elapsed = time.time() - t0
    logger.info(f"      → {n_clusters} communities, "
                f"modularity={modularity:.4f} ({elapsed:.0f}s)")

    # Size distribution summary
    sizes = np.bincount(labels)
    logger.info(f"      sizes: min={sizes.min()}, median={np.median(sizes):.0f}, "
                f"max={sizes.max()}, mean={sizes.mean():.0f}")

    np.save(label_path, labels)
    return labels


def run():
    logger.info("=" * 60)
    logger.info("STEP 6: Leiden Community Detection")
    logger.info("=" * 60)

    t0 = time.time()

    # Import fractional grid
    from leiden_parallel import JOBS

    # Load manifest for image_ids
    manifest = pd.read_parquet(config.manifest_path(),
                                columns=['image_id'])

    # Group jobs by (pca_dim, k, graph_type) so we load each graph at most once.
    from collections import defaultdict
    grouped_jobs = defaultdict(list)
    for pca_dim_j, k_j, graph_type_j, res, seed in JOBS:
        grouped_jobs[(pca_dim_j, k_j, graph_type_j)].append((res, seed))

    all_labels = {}
    meta_records = []

    for (pca_dim_j, k_j, graph_type_j), runs in grouped_jobs.items():
        # Figure out which runs in this group still need computation
        pending = []
        cached = []
        for res, seed in runs:
            label_path = config.leiden_labels_path(pca_dim_j, k_j, graph_type_j, res, seed)
            if os.path.exists(label_path):
                cached.append((res, seed, label_path))
            else:
                pending.append((res, seed, label_path))

        g = None
        if pending:
            graph_file = config.graph_path(graph_type_j, k_j, pca_dim_j)
            logger.info(f"Loading graph: {graph_file}")
            graph_sparse = sparse.load_npz(graph_file)
            g = _sparse_to_igraph(graph_sparse)
            del graph_sparse

        for res, seed, label_path in pending:
            labels = _run_leiden_single(g, res, seed, pca_dim_j, k_j, graph_type_j)
            res_str = f"{res:.2f}".replace(".", "")
            col_name = f"leiden_k{k_j}_g{graph_type_j}_res{res_str}_seed{seed}"
            all_labels[col_name] = labels

            meta_path = label_path.replace(".npy", ".meta.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    meta['config_id'] = col_name
                    meta_records.append(meta)

        if g is not None:
            del g
            gc.collect()

        for res, seed, label_path in cached:
            labels = np.load(label_path)
            res_str = f"{res:.2f}".replace(".", "")
            col_name = f"leiden_k{k_j}_g{graph_type_j}_res{res_str}_seed{seed}"
            all_labels[col_name] = labels

            meta_path = label_path.replace(".npy", ".meta.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    meta['config_id'] = col_name
                    meta_records.append(meta)

    # Build consolidated parquet
    consolidated_path = config.leiden_parquet_path()
    if not os.path.exists(consolidated_path):
        logger.info("\nBuilding consolidated Leiden parquet...")
        df = manifest.copy()
        for col_name, labels in all_labels.items():
            df[col_name] = labels

        df.to_parquet(consolidated_path + ".tmp", index=False)
        os.rename(consolidated_path + ".tmp", consolidated_path)
        logger.info(f"  Saved: {consolidated_path}")

        # Summary
        for col_name in all_labels:
            nc = df[col_name].nunique()
            logger.info(f"  {col_name}: {nc} communities")
    else:
        logger.info(f"[CHECKPOINT] Consolidated parquet exists: {consolidated_path}")
        
    # Save consolidated metadata
    if meta_records:
        meta_df = pd.DataFrame(meta_records)
        meta_out = config.leiden_parquet_path().replace(".parquet", "_metadata.csv")
        meta_df.to_csv(meta_out, index=False)
        logger.info(f"  Saved metadata: {meta_out}")

    elapsed = time.time() - t0
    logger.info(f"\nLeiden clustering complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    run()
