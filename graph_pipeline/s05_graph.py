"""
Step 05: Graph Construction
==============================
Constructs symmetric and mutual kNN graphs from FAISS neighbor arrays.
Stores as scipy sparse CSR matrices.

Outputs:
  - graph_symmetric_k{k}_pca{dim}.npz
  - graph_mutual_k{k}_pca{dim}.npz
  - graph_stats.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s05_graph")


def _snn_jaccard_edgewise(A, n, edge_mask=None, chunk_size=5_000):
    """Compute SNN-Jaccard weights on the edges of `edge_mask` (defaults to A)
    without materialising the full A·A. Processes A in row chunks; the
    `.multiply(edge_mask_chunk)` immediately prunes to existing edges so the
    intermediate per chunk has ~deg² · chunk_size nnz (a few GB at deg≈100,
    chunk=5k), not 7.2M × 7.2M.

    Returns a symmetric CSR matrix of Jaccard weights on the surviving edges.
    """
    A = A.tocsr()
    if edge_mask is None:
        edge_mask_csr = A
    else:
        edge_mask_csr = edge_mask.tocsr()

    A_T = A.T.tocsr()
    row_sums = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)

    rows_out = []
    cols_out = []
    vals_out = []

    t_start = time.time()
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        # |N(i) ∩ N(j)| = (A · A^T)[i, j] when A is binary
        chunk_inter = A[start:end].dot(A_T)
        # Restrict to existing edges immediately to bound memory
        chunk_inter = chunk_inter.multiply(edge_mask_csr[start:end]).tocoo()
        if chunk_inter.nnz == 0:
            continue
        local_rows = chunk_inter.row.astype(np.int64) + start
        local_cols = chunk_inter.col.astype(np.int64)
        inter_vals = chunk_inter.data.astype(np.float32)
        union_vals = (row_sums[local_rows] + row_sums[local_cols] - inter_vals).astype(np.float32)
        jaccard = inter_vals / np.maximum(union_vals, 1.0)

        rows_out.append(local_rows)
        cols_out.append(local_cols)
        vals_out.append(jaccard)

        if (ci + 1) % 20 == 0 or ci + 1 == n_chunks:
            so_far = sum(len(v) for v in vals_out)
            logger.info(f"    SNN chunk {ci+1}/{n_chunks} ({end:,}/{n:,} rows, "
                        f"{so_far:,} edges, {time.time()-t_start:.0f}s)")

    if not vals_out:
        return sparse.csr_matrix((n, n), dtype=np.float32)

    rows_all = np.concatenate(rows_out)
    cols_all = np.concatenate(cols_out)
    vals_all = np.concatenate(vals_out)
    return sparse.csr_matrix((vals_all, (rows_all, cols_all)), shape=(n, n))


def _build_graphs(neighbors, distances, n, k, pca_dim):
    """Build symmetric and mutual kNN graphs from neighbor arrays."""
    results = {}

    for graph_type in config.GRAPH_TYPES:
        graph_file = config.graph_path(graph_type, k, pca_dim)
        if os.path.exists(graph_file):
            logger.info(f"  [CHECKPOINT] {graph_type} graph already exists")
            results[graph_type] = sparse.load_npz(graph_file)
            continue

        logger.info(f"  Building {graph_type} kNN graph (k={k})...")
        t0 = time.time()

        # Vectorized edge list: drop invalid (j<0) and self-loops (j==i)
        i_idx = np.repeat(np.arange(n, dtype=np.int64), k)
        j_idx = neighbors.ravel().astype(np.int64, copy=False)
        w_flat = distances.ravel().astype(np.float32, copy=False)
        keep = (j_idx >= 0) & (j_idx != i_idx)

        rows = i_idx[keep]
        cols = j_idx[keep]
        weights = w_flat[keep]
        del i_idx, j_idx, w_flat, keep

        logger.info(f"    Raw directed edges: {len(rows):,}")

        # Build directed sparse matrix
        directed = sparse.csr_matrix(
            (weights, (rows, cols)), shape=(n, n)
        )

        if graph_type == "symmetric":
            # Keep edge if i→j OR j→i, weight = max(w_ij, w_ji)
            graph = directed.maximum(directed.T)
        elif graph_type == "mutual":
            # Keep edge only if i→j AND j→i, weight = avg(w_ij, w_ji)
            # mutual = element-wise minimum of presence, then average weights
            mask_ij = (directed > 0).astype(np.float32)
            mask_ji = (directed.T > 0).astype(np.float32)
            mutual_mask = mask_ij.multiply(mask_ji)
            # Average the weights
            graph = (directed + directed.T).multiply(mutual_mask) * 0.5
        elif graph_type == "snn_jaccard":
            # Build symmetric adjacency A (1 where i or j has the other as neighbor)
            A = (directed.maximum(directed.T) > 0).astype(np.int32).tocsr()
            graph = _snn_jaccard_edgewise(A, n)
            del A

        elif graph_type == "mutual_snn_jaccard":
            # SNN Jaccard restricted to mutual edges of the directed graph.
            # Compute SNN edge-wise on the mutual mask itself (cheaper and equivalent
            # since multiply(mutual_mask) would prune everything else anyway).
            mask_ij = (directed > 0)
            mutual_mask = mask_ij.multiply(directed.T > 0).astype(np.int32).tocsr()
            # Use the full symmetric adjacency for the neighborhood sets, but
            # only score the edges that survive the mutual filter.
            A = (directed.maximum(directed.T) > 0).astype(np.int32).tocsr()
            graph = _snn_jaccard_edgewise(A, n, edge_mask=mutual_mask)
            del A, mask_ij, mutual_mask

        # Ensure no self-loops
        graph.setdiag(0)
        graph.eliminate_zeros()

        elapsed = time.time() - t0
        logger.info(f"    {graph_type} graph: {graph.nnz:,} edges in {elapsed:.0f}s")

        # Save
        sparse.save_npz(graph_file, graph)
        logger.info(f"    Saved: {graph_file}")
        results[graph_type] = graph

        del rows, cols, weights, directed

    return results


def _compute_graph_stats(graphs, k, pca_dim):
    """Compute and save graph statistics."""
    output_path = config.graph_stats_path()
    if os.path.exists(output_path):
        logger.info(f"[CHECKPOINT] Graph stats already exist")
        return pd.read_csv(output_path)

    records = []
    for graph_type, graph in graphs.items():
        n = graph.shape[0]

        # Degree distribution (symmetric → degree = row sum of nonzeros)
        # For undirected graph stored as symmetric sparse, each edge is
        # stored twice. We use the upper triangle to count.
        degrees = np.array((graph > 0).sum(axis=1)).flatten()

        # Connected components
        n_components, component_labels = sparse.csgraph.connected_components(
            graph, directed=False
        )
        component_sizes = np.bincount(component_labels)
        giant_component_size = component_sizes.max()
        isolated = (degrees == 0).sum()

        # Edge weight quantiles
        edge_weights = graph.data
        if len(edge_weights) > 0:
            w_quantiles = np.percentile(edge_weights, [10, 25, 50, 75, 90])
        else:
            w_quantiles = [0] * 5

        record = {
            'graph_type': graph_type,
            'k': k,
            'pca_dim': pca_dim,
            'n_nodes': n,
            'n_edges': graph.nnz // 2,  # undirected
            'avg_degree': degrees.mean(),
            'median_degree': np.median(degrees),
            'min_degree': degrees.min(),
            'max_degree': degrees.max(),
            'p10_degree': np.percentile(degrees, 10),
            'p25_degree': np.percentile(degrees, 25),
            'p75_degree': np.percentile(degrees, 75),
            'p90_degree': np.percentile(degrees, 90),
            'n_connected_components': n_components,
            'giant_component_pct': giant_component_size / n * 100,
            'giant_component_size': int(giant_component_size),
            'n_isolated_nodes': int(isolated),
            'edge_weight_p10': w_quantiles[0],
            'edge_weight_p25': w_quantiles[1],
            'edge_weight_p50': w_quantiles[2],
            'edge_weight_p75': w_quantiles[3],
            'edge_weight_p90': w_quantiles[4],
        }
        records.append(record)

        logger.info(f"  {graph_type}: {record['n_edges']:,} edges, "
                     f"avg_deg={record['avg_degree']:.1f}, "
                     f"components={n_components}, "
                     f"giant={record['giant_component_pct']:.1f}%, "
                     f"isolated={isolated}")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved: {output_path}")
    return df


def run():
    logger.info("=" * 60)
    logger.info("STEP 5: Graph Construction")
    logger.info("=" * 60)

    t0 = time.time()

    pca_dim = config.get_selected_pca_dim()
    k = config.KNN_DEFAULT_K

    # Load kNN results
    neighbors_path = config.knn_neighbors_path(k, pca_dim)
    distances_path = config.knn_distances_path(k, pca_dim)

    logger.info(f"Loading kNN results: k={k}, PCA-{pca_dim}")
    neighbors = np.load(neighbors_path)
    distances = np.load(distances_path)
    n = neighbors.shape[0]
    logger.info(f"  {n:,} nodes, k={k}")

    # Build graphs
    graphs = _build_graphs(neighbors, distances, n, k, pca_dim)

    del neighbors, distances

    # Compute stats
    logger.info("\n--- Graph Statistics ---")
    _compute_graph_stats(graphs, k, pca_dim)

    elapsed = time.time() - t0
    logger.info(f"\nGraph construction complete in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    run()
