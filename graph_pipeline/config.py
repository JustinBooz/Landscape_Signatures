"""
Graph-Based Embedding Analysis Pipeline — Shared Configuration
===============================================================
Central configuration for all pipeline steps. Import this module
from any step script to get paths, parameters, and logging setup.
"""

import os
import sys
import logging

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PT_DIR = "/home/jubooz/landscape_signatures/training_data_7b"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Ensure output dirs exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding parameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 4096
EMBEDDING_DTYPE_STORAGE = "float16"   # for memmap storage
EMBEDDING_DTYPE_COMPUTE = "float32"   # for PCA/FAISS/indexing

# ---------------------------------------------------------------------------
# PCA parameters
# ---------------------------------------------------------------------------
PCA_DIMS = [128, 256, 512]
PCA_BATCH_SIZE = 40_000
PCA_DEFAULT_DIM = 256                 # default unless 128 proves sufficient
PCA_NEIGHBOR_SAMPLE_SIZE = 100_000
PCA_RECALL_THRESHOLD = 0.95           # if recall@50 for 128D >= this, use 128

# ---------------------------------------------------------------------------
# FAISS kNN parameters
# ---------------------------------------------------------------------------
KNN_K_VALUES = [50, 100, 200]
KNN_DEFAULT_K = 100
FAISS_NLIST = 4096                    # IVF cluster count
FAISS_NPROBE = 128                    # search-time probe count
FAISS_RECALL_SAMPLE_SIZE = 100_000

# ---------------------------------------------------------------------------
# Graph parameters
# ---------------------------------------------------------------------------
GRAPH_TYPES = ["symmetric", "mutual", "snn_jaccard", "mutual_snn_jaccard"]
GRAPH_DEFAULT_TYPE = "snn_jaccard"

# ---------------------------------------------------------------------------
# Microcluster parameters
# ---------------------------------------------------------------------------
MICROCLUSTER_K = 50_000


# ---------------------------------------------------------------------------
# Leiden parameters
# ---------------------------------------------------------------------------
LEIDEN_RESOLUTIONS = [0.25, 0.5, 1.0, 2.0, 4.0]

# Reference (canonical) Leiden configuration — single source of truth.
# All step files use these to load THE reference clustering.
REF_PCA = 256
REF_K = 100
REF_GRAPH = "snn_jaccard"
REF_RES = 1.0
REF_SEED = 0

# Number of concurrent Leiden workers in leiden_parallel.py.
# Each worker holds ~18 GB during the heaviest jobs (k=200 graph).
LEIDEN_MAX_PARALLEL = 3

# ---------------------------------------------------------------------------
# HDBSCAN auxiliary parameters
# ---------------------------------------------------------------------------
# s09 (point-wise) samples the embeddings before running HDBSCAN to bound
# memory; min_cluster_size is scaled to keep cluster-size meaning consistent.
HDBSCAN_SAMPLE_SIZE = 1_000_000
HDBSCAN_SETTINGS = [
    {"min_cluster_size": 5_000,  "min_samples": 50},
    {"min_cluster_size": 20_000, "min_samples": 100},
    {"min_cluster_size": 50_000, "min_samples": 250},
]

# ---------------------------------------------------------------------------
# UMAP visualization parameters
# ---------------------------------------------------------------------------
UMAP_SAMPLE_SIZES = [250_000, 500_000]
UMAP_SETTINGS = [
    {"n_neighbors": 100, "min_dist": 0.1},
    {"n_neighbors": 300, "min_dist": 0.2},
]
UMAP_METRIC = "cosine"
UMAP_N_COMPONENTS = 2

# ---------------------------------------------------------------------------
# Coordinate constants (EPSG:2056 / CH1903+ / LV95)
# ---------------------------------------------------------------------------
MIN_EASTING = 2_400_000.0
MAX_EASTING = 2_900_000.0
MIN_NORTHING = 1_000_000.0
MAX_NORTHING = 1_350_000.0

# ---------------------------------------------------------------------------
# Medoid parameters
# ---------------------------------------------------------------------------
MEDOID_MAX_SAMPLE = 5_000           # max members to use for medoid computation
MEDOID_TOP_NEAREST = 20
MEDOID_RANDOM_SAMPLE = 100
MEDOID_SPATIAL_SAMPLE = 100

# ---------------------------------------------------------------------------
# Random state
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Output file paths (functions to generate paths with parameters)
# ---------------------------------------------------------------------------
def manifest_path():
    return os.path.join(OUTPUT_DIR, "manifest.parquet")

def normed_embeddings_path():
    return os.path.join(OUTPUT_DIR, "embeddings_normed.mmap")

def normed_embeddings_shape_path():
    return os.path.join(OUTPUT_DIR, "embeddings_normed_shape.json")

def pca_path(dim):
    return os.path.join(OUTPUT_DIR, f"pca_{dim}.mmap")

def pca_shape_path(dim):
    return os.path.join(OUTPUT_DIR, f"pca_{dim}_shape.json")

def pca_model_path(dim):
    return os.path.join(OUTPUT_DIR, f"pca_{dim}_model.pkl")

def pca_variance_report_path():
    return os.path.join(OUTPUT_DIR, "pca_variance_report.csv")

def pca_neighbor_preservation_path():
    return os.path.join(OUTPUT_DIR, "pca_neighbor_preservation.csv")

def pca_selected_dim_path():
    return os.path.join(OUTPUT_DIR, "pca_selected_dim.txt")

def knn_neighbors_path(k, pca_dim=None):
    pca_dim = pca_dim or PCA_DEFAULT_DIM
    return os.path.join(OUTPUT_DIR, f"neighbors_k{k}_pca{pca_dim}.npy")

def knn_distances_path(k, pca_dim=None):
    pca_dim = pca_dim or PCA_DEFAULT_DIM
    return os.path.join(OUTPUT_DIR, f"distances_k{k}_pca{pca_dim}.npy")

def faiss_recall_path():
    return os.path.join(OUTPUT_DIR, "faiss_recall_diagnostics.csv")

def graph_path(graph_type, k, pca_dim=None):
    pca_dim = pca_dim or PCA_DEFAULT_DIM
    return os.path.join(OUTPUT_DIR, f"graph_{graph_type}_k{k}_pca{pca_dim}.npz")

def graph_stats_path():
    return os.path.join(OUTPUT_DIR, "graph_stats.csv")

def leiden_labels_path(pca_dim, k, graph_type, res, seed):
    res_str = f"{res:.2f}".replace(".", "")
    return os.path.join(OUTPUT_DIR,
        f"leiden_pca{pca_dim}_k{k}_{graph_type}_res{res_str}_seed{seed}.npy")

def leiden_parquet_path():
    return os.path.join(OUTPUT_DIR, "cluster_labels_leiden.parquet")

def stability_matrix_path():
    return os.path.join(OUTPUT_DIR, "leiden_stability_matrix.csv")

def cluster_size_dist_path():
    return os.path.join(OUTPUT_DIR, "cluster_size_distributions.csv")

def stable_cluster_summary_path():
    return os.path.join(OUTPUT_DIR, "stable_cluster_summary.csv")

def medoids_path(res):
    res_str = f"{res:.2f}".replace(".", "")
    return os.path.join(OUTPUT_DIR, f"cluster_medoids_res{res_str}.parquet")

def cluster_confidence_path():
    return os.path.join(OUTPUT_DIR, "cluster_confidence.parquet")

def microcluster_centroids_path():
    return os.path.join(OUTPUT_DIR, "microcluster_centroids.npy")

def microcluster_assignments_path():
    return os.path.join(OUTPUT_DIR, "microcluster_assignments.parquet")

def finch_labels_path():
    return os.path.join(OUTPUT_DIR, "finch_labels.parquet")

def hdbscan_labels_path():
    """Microcluster-based HDBSCAN labels (written by s09b)."""
    return os.path.join(OUTPUT_DIR, "hdbscan_labels.parquet")

def hdbscan_pointwise_labels_path():
    """Point-wise HDBSCAN labels on the 1M sample (written by s09)."""
    return os.path.join(OUTPUT_DIR, "hdbscan_pointwise_labels.parquet")

def hdbscan_noise_summary_path():
    return os.path.join(OUTPUT_DIR, "hdbscan_noise_summary.csv")

def hdbscan_vs_leiden_path():
    return os.path.join(OUTPUT_DIR, "hdbscan_vs_leiden_contingency.csv")

def umap_sample_path(n):
    return os.path.join(OUTPUT_DIR, f"umap_sample_{n // 1000}k.parquet")

def spatial_summary_path():
    return os.path.join(OUTPUT_DIR, "cluster_spatial_summary.parquet")

def cluster_spatial_validation_path():
    return os.path.join(OUTPUT_DIR, "cluster_spatial_validation.parquet")

def region_enrichment_path():
    return os.path.join(OUTPUT_DIR, "cluster_region_enrichment.csv")

def external_validation_path():
    return os.path.join(OUTPUT_DIR, "external_validation_summary.csv")

def negative_control_path():
    return os.path.join(OUTPUT_DIR, "negative_control_results.csv")

def negative_control_pvalues_path():
    return os.path.join(OUTPUT_DIR, "negative_control_pvalues.parquet")

def master_parquet_path():
    return os.path.join(OUTPUT_DIR, "landscape_embedding_analysis_master.parquet")

def cluster_signatures_summary_path():
    return os.path.join(OUTPUT_DIR, "cluster_signatures_summary.parquet")

def method_summary_path():
    return os.path.join(REPORTS_DIR, "README_method_summary.md")


# ---------------------------------------------------------------------------
# Utility: get selected PCA dimension (reads from file, defaults to 256)
# ---------------------------------------------------------------------------
def get_selected_pca_dim():
    """Read the selected PCA dimension from the selection file, or return default."""
    path = pca_selected_dim_path()
    if os.path.exists(path):
        with open(path, 'r') as f:
            return int(f.read().strip())
    return PCA_DEFAULT_DIM


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(name, log_file=None):
    """Configure logging for a pipeline step."""
    if log_file is None:
        log_file = os.path.join(BASE_DIR, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-import
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Utility: load memmap with shape metadata
# ---------------------------------------------------------------------------
def load_memmap(mmap_path, shape_path, dtype='float32', mode='r'):
    """Load a numpy memmap with shape read from a JSON sidecar."""
    import json
    import numpy as np
    with open(shape_path, 'r') as f:
        meta = json.load(f)
    shape = tuple(meta['shape'])
    return np.memmap(mmap_path, dtype=dtype, mode=mode, shape=shape)
