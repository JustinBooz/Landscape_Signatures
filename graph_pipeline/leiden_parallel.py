"""
Parallel Leiden Runner — Runs multiple resolution/seed combos simultaneously.
Each worker loads the graph independently and runs one Leiden call.
Skips existing checkpoints. Uses multiprocessing to saturate CPU cores.
Retries segfaulted runs with reduced n_iterations.
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("leiden_parallel")

# Reference configuration — single source of truth lives in config.py.
# Re-exported here so existing `from leiden_parallel import REF_*` imports keep
# working.
from config import REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED

# Fractional grid based on the reference configuration
JOBS = []

# 1. Seed perturbations (reference parameters, varying seed)
for seed in [REF_SEED, 1, 2, 3, 4]:
    JOBS.append((REF_PCA, REF_K, REF_GRAPH, REF_RES, seed))

# 2. k perturbations (reference seed, varying k)
for k in [50, 200]:
    JOBS.append((REF_PCA, k, REF_GRAPH, REF_RES, REF_SEED))

# 3. Graph perturbations (reference seed, varying graph)
for g in ['mutual', 'mutual_snn_jaccard']:
    JOBS.append((REF_PCA, REF_K, g, REF_RES, REF_SEED))

# 4. Resolution perturbations (reference seed, varying resolution)
for res in [0.25, 0.5, 2.0, 4.0]:
    JOBS.append((REF_PCA, REF_K, REF_GRAPH, res, REF_SEED))

# Note: 'mutual_knn' from instructions matches our 'mutual' string in config

# Concurrent worker count — sourced from config.LEIDEN_MAX_PARALLEL
MAX_PARALLEL = config.LEIDEN_MAX_PARALLEL


def _run_one_leiden(res, seed, graph_path, pca_dim, k, graph_type, n_iterations=-1):
    """Run a single Leiden community detection in its own process."""
    import igraph as ig
    import leidenalg

    out_path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
    label = f"res={res}, seed={seed}"

    # Check checkpoint
    if os.path.exists(out_path):
        print(f"  [{label}] CHECKPOINT exists, skipping", flush=True)
        return

    t0 = time.time()
    iter_str = f"n_iter={n_iterations}" if n_iterations > 0 else "n_iter=∞"
    print(f"  [{label}] Loading graph... ({iter_str})", flush=True)

    mat = sp.load_npz(graph_path)
    n = mat.shape[0]

    # Convert to igraph — build edge list via numpy (single C-speed copy)
    coo = sp.triu(mat, k=1).tocoo()
    edges = np.column_stack((coo.row, coo.col)).tolist()
    weights = coo.data.tolist()
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es['weight'] = weights
    del coo, edges, mat

    t_load = time.time() - t0
    print(f"  [{label}] Graph loaded ({t_load:.0f}s), running Leiden...", flush=True)

    t1 = time.time()
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=res,
        weights='weight',
        seed=seed,
        n_iterations=n_iterations,
    )

    labels = np.array(partition.membership, dtype=np.int32)
    mod = partition.modularity
    n_comm = len(set(labels))
    elapsed = time.time() - t1

    # Atomic save — labels
    tmp_path = out_path + ".tmp.npy"
    np.save(tmp_path, labels)
    os.rename(tmp_path, out_path)

    # Compute additional requested metadata
    sizes = np.bincount(labels)
    singleton_count = int(np.sum(sizes == 1))
    largest_cluster_size = int(sizes.max())
    sorted_sizes = np.sort(sizes)[::-1]
    top10_percent = float(sorted_sizes[:10].sum() / n * 100) if len(sizes) >= 10 else 100.0

    meta = {
        "pca_dim": pca_dim,
        "k": k,
        "graph_type": graph_type,
        "resolution": res,
        "seed": seed,
        "num_clusters": n_comm,
        "singleton_count": singleton_count,
        "largest_cluster_size": largest_cluster_size,
        "percent_in_top10_clusters": top10_percent,
        "modularity": float(mod),
        "runtime_seconds": float(elapsed),
        "graph_used": graph_path
    }

    # Save metadata sidecar
    meta_path = out_path.replace(".npy", ".meta.json")
    import json
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.rename(meta_path + ".tmp", meta_path)

    print(
        f"  [{label}] DONE: {n_comm} communities, mod={mod:.4f}, "
        f"max={largest_cluster_size}, singletons={singleton_count}, "
        f"top10%={top10_percent:.1f}%, time={elapsed:.0f}s",
        flush=True
    )


def main():
    # Filter to pending jobs
    pending = []
    for pca_dim, k, graph_type, res, seed in JOBS:
        out_path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
        if os.path.exists(out_path):
            logger.info(f"  [CHECKPOINT] k={k}, g={graph_type}, res={res}, seed={seed} exists")
        else:
            pending.append((pca_dim, k, graph_type, res, seed))

    if not pending:
        logger.info(f"All {len(JOBS)} Leiden runs complete!")
        return

    logger.info(f"Pending: {len(pending)} runs, max parallel: {MAX_PARALLEL}")
    for pca_dim, k, graph_type, res, seed in pending:
        logger.info(f"  → k={k}, g={graph_type}, res={res}, seed={seed}")

    # Run in batches of MAX_PARALLEL with retry
    i = 0
    while i < len(pending):
        batch = pending[i:i + MAX_PARALLEL]
        procs = []

        for pca_dim, k, graph_type, res, seed in batch:
            graph_path = config.graph_path(graph_type, k, pca_dim)
            if not os.path.exists(graph_path):
                logger.error(f"Missing graph for k={k}, g={graph_type}. Skipping.")
                continue

            p = Process(
                target=_run_one_leiden,
                args=(res, seed, graph_path, pca_dim, k, graph_type),
                name=f"leiden_k{k}_g{graph_type}_r{res}_s{seed}"
            )
            p.start()
            logger.info(f"  Started PID {p.pid}: k={k}, g={graph_type}, res={res}, seed={seed}")
            procs.append((p, pca_dim, k, graph_type, res, seed))

        # Wait for all in batch to complete
        failed_in_batch = []
        for p, pca_dim, k, graph_type, res, seed in procs:
            p.join()
            if p.exitcode != 0:
                logger.warning(f"  k={k}, g={graph_type}, res={res}, seed={seed} exited with code {p.exitcode}")
                failed_in_batch.append((pca_dim, k, graph_type, res, seed))
            else:
                logger.info(f"  k={k}, g={graph_type}, res={res}, seed={seed} completed successfully")

        # Retry failed runs with reduced iterations (one at a time)
        for pca_dim, k, graph_type, res, seed in failed_in_batch:
            graph_path = config.graph_path(graph_type, k, pca_dim)
            out_path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
            if os.path.exists(out_path):
                continue  # somehow succeeded
            logger.info(f"  Retrying k={k}, g={graph_type}, res={res}, seed={seed} with n_iterations=10")
            p = Process(
                target=_run_one_leiden,
                args=(res, seed, graph_path, pca_dim, k, graph_type, 10),
                name=f"leiden_retry_k{k}_g{graph_type}_r{res}_s{seed}"
            )
            p.start()
            p.join()
            if p.exitcode != 0:
                logger.error(
                    f"  k={k}, g={graph_type}, res={res}, seed={seed} FAILED on retry (exit={p.exitcode}). "
                    f"Skipping this run."
                )
            else:
                logger.info(f"  k={k}, g={graph_type}, res={res}, seed={seed} retry succeeded")

        i += len(batch)

    # Summary
    done = 0
    for pca_dim, k, graph_type, res, seed in JOBS:
        if os.path.exists(config.leiden_labels_path(pca_dim, k, graph_type, res, seed)):
            done += 1
    logger.info(f"\nLeiden complete: {done}/{len(JOBS)} runs saved")


if __name__ == "__main__":
    main()
