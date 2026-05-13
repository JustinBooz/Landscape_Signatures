"""
Zero-Shot HDBSCAN Clustering for DINOv3 ViT-7B16 Embeddings
=============================================================
Loads pre-extracted 4096D embeddings from .pt files, reduces
dimensionality via PCA → UMAP, then runs multi-resolution HDBSCAN
to discover hierarchical landscape typologies.

Pipeline: Load .pt → L2-normalize → PCA(128) → UMAP(2D) → HDBSCAN
Output:   zeroshot_7b_hdbscan_data.parquet + visualization PNGs

Hardware: 128 GB RAM system (no GPU needed for clustering)
"""

import os
import sys
import glob
import time
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('zeroshot_7b_clustering.log')
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PT_DIR       = "/home/jubooz/landscape_signatures/training_data_7b"
OUTPUT_DIR   = "/home/jubooz/landscape_signatures/clustering_7b_results"
PCA_DIMS     = 128       # Intermediate PCA reduction (4096 → 128)
UMAP_DIMS    = 2         # Final UMAP projection
UMAP_NEIGHBORS = 15      # n_neighbors for UMAP
UMAP_MIN_DIST  = 0.0     # min_dist (0.0 = tighter clusters)
UMAP_SUBSAMPLE = 500000  # Fit UMAP on subsample, then transform rest
RANDOM_STATE   = 42

# HDBSCAN multi-resolution scales
HDBSCAN_SCALES = {
    "micro":  {"label": "Micro-Typologies (Neighborhoods)",    "min_cluster_size": 30,   "min_samples": 10},
    "meso":   {"label": "Meso-Typologies (Cities & Cantons)",  "min_cluster_size": 200,  "min_samples": 50},
    "macro":  {"label": "Macro-Typologies (Bioregional)",      "min_cluster_size": 800,  "min_samples": 100},
}


# ---------------------------------------------------------------------------
# Step 1: Load all .pt files into contiguous arrays
# ---------------------------------------------------------------------------
def load_pt_embeddings(pt_dir: str, cache_path: str = None):
    """Load all .pt files into pre-allocated arrays (two-pass to avoid OOM).
    
    Pass 1: count total rows.
    Pass 2: fill pre-allocated float16 embeddings + float32 coords.
    
    Uses float16 embeddings (~26 GB for 3.2M×4096) instead of float32
    to stay within memory. PCA will upcast per-batch internally.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"[CACHE HIT] Loading from {cache_path}")
        data = np.load(cache_path)
        return data['embeddings'], data['coords']

    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files in {pt_dir}")

    # --- Pass 1: Count total rows ---
    logger.info("Pass 1: Counting total embeddings...")
    row_counts = []
    for pt_path in pt_files:
        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        row_counts.append(d['embeddings'].shape[0])
        del d
    total_rows = sum(row_counts)
    logger.info(f"  Total: {total_rows:,} embeddings across {len(pt_files)} files")

    # --- Pass 2: Pre-allocate and fill ---
    logger.info(f"Pass 2: Pre-allocating float16 array ({total_rows:,} × 4096)...")
    embeddings = np.empty((total_rows, 4096), dtype=np.float16)
    coords = np.empty((total_rows, 2), dtype=np.float32)

    offset = 0
    for i, pt_path in enumerate(pt_files):
        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        n = d['embeddings'].shape[0]
        # bfloat16 → float16 via float32 intermediate (per-file, small)
        embeddings[offset:offset+n] = d['embeddings'].float().numpy().astype(np.float16)
        coords[offset:offset+n] = d['coords'].numpy()
        offset += n
        del d

        if (i + 1) % 200 == 0:
            logger.info(f"  Filled {i+1}/{len(pt_files)} files ({offset:,} embeddings)")

    logger.info(f"Loaded: {embeddings.shape[0]:,} × {embeddings.shape[1]}D (float16)")
    logger.info(f"Coords: {coords.shape} (float32)")
    logger.info(f"Memory: embeddings={embeddings.nbytes/1e9:.1f} GB, "
                f"coords={coords.nbytes/1e6:.0f} MB")

    if cache_path:
        logger.info(f"Saving cache to {cache_path}...")
        np.savez(cache_path, embeddings=embeddings, coords=coords)
        logger.info(f"Cache saved ({os.path.getsize(cache_path) / 1e9:.1f} GB)")

    return embeddings, coords


def reduce_dimensions_pca(embeddings: np.ndarray, n_components: int = 128,
                          batch_size: int = 50000, cache_path: str = None):
    """PCA: fit on subsample, transform all. Fast for 3M+ scale.
    
    - Fits PCA on 500K random samples (converges well for 128 components)
    - Transforms full dataset in 50K batches
    - Handles float16 input via per-batch float32 cast
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"[CACHE HIT] Loading PCA result from {cache_path}")
        return np.load(cache_path)['pca_emb']

    n = len(embeddings)
    FIT_SUBSAMPLE = 500000
    fit_n = min(FIT_SUBSAMPLE, n)
    
    logger.info(f"PCA: fitting on {fit_n:,} subsample, then transforming {n:,} points")
    logger.info(f"  {embeddings.shape[1]}D → {n_components}D, batch_size={batch_size}")
    t0 = time.time()

    # Select random subsample for fitting
    rng = np.random.RandomState(RANDOM_STATE)
    fit_idx = rng.choice(n, size=fit_n, replace=False)
    
    ipca = IncrementalPCA(n_components=n_components)
    
    # Fit on subsample in batches
    for start in range(0, fit_n, batch_size):
        end = min(start + batch_size, fit_n)
        batch = embeddings[fit_idx[start:end]].astype(np.float32)
        batch = normalize(batch, norm='l2', axis=1)
        ipca.partial_fit(batch)
        logger.info(f"  PCA fit: {end:,}/{fit_n:,}")

    t_fit = time.time() - t0
    explained = ipca.explained_variance_ratio_.sum()
    logger.info(f"  PCA fit done: {explained:.1%} variance in {t_fit:.0f}s")

    # Transform ALL points in batches
    pca_result = np.empty((n, n_components), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = embeddings[start:end].astype(np.float32)
        batch = normalize(batch, norm='l2', axis=1)
        pca_result[start:end] = ipca.transform(batch)
        if (start // batch_size) % 10 == 0:
            logger.info(f"  PCA transform: {end:,}/{n:,}")

    elapsed = time.time() - t0
    logger.info(f"PCA complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if cache_path:
        np.savez(cache_path, pca_emb=pca_result)
        logger.info(f"PCA cache saved to {cache_path}")

    return pca_result


# ---------------------------------------------------------------------------
# Step 3: UMAP projection (subsample-fit, then batch-transform)
# ---------------------------------------------------------------------------
def run_umap(pca_embeddings: np.ndarray, cache_path: str = None):
    """UMAP 128D → 2D via subsample strategy for speed.
    
    1. Fit UMAP on UMAP_SUBSAMPLE random points (~20 min)
    2. Transform ALL points in batches via the fitted model (~10 min)
    Total: ~30 min instead of 3-6 hours for full fit_transform.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"[CACHE HIT] Loading UMAP result from {cache_path}")
        return np.load(cache_path)['umap_2d']

    import umap

    n = len(pca_embeddings)
    subsample_n = min(UMAP_SUBSAMPLE, n)
    
    # Select random subsample
    rng = np.random.RandomState(RANDOM_STATE)
    subsample_idx = rng.choice(n, size=subsample_n, replace=False)
    subsample_data = pca_embeddings[subsample_idx]

    logger.info(f"UMAP Phase 1: Fitting on {subsample_n:,} / {n:,} subsample...")
    t0 = time.time()

    reducer = umap.UMAP(
        n_components=UMAP_DIMS,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=RANDOM_STATE,
        low_memory=True,
        verbose=True,
    )
    reducer.fit(subsample_data)
    del subsample_data

    t_fit = time.time() - t0
    logger.info(f"UMAP fit complete in {t_fit:.0f}s ({t_fit/60:.1f} min)")

    # Transform ALL points in batches
    logger.info(f"UMAP Phase 2: Transforming all {n:,} points...")
    t1 = time.time()
    
    TRANSFORM_BATCH = 100000
    umap_2d = np.empty((n, UMAP_DIMS), dtype=np.float32)
    for start in range(0, n, TRANSFORM_BATCH):
        end = min(start + TRANSFORM_BATCH, n)
        umap_2d[start:end] = reducer.transform(pca_embeddings[start:end])
        logger.info(f"  Transformed {end:,}/{n:,}")

    t_transform = time.time() - t1
    total = time.time() - t0
    logger.info(f"UMAP transform complete in {t_transform:.0f}s ({t_transform/60:.1f} min)")
    logger.info(f"UMAP total: {total:.0f}s ({total/60:.1f} min)")

    if cache_path:
        np.savez(cache_path, umap_2d=umap_2d)
        logger.info(f"UMAP cache saved to {cache_path}")

    return umap_2d


# ---------------------------------------------------------------------------
# Step 4: Multi-resolution HDBSCAN clustering
# ---------------------------------------------------------------------------
def run_hdbscan_multiresolution(umap_2d: np.ndarray):
    """Run HDBSCAN at micro/meso/macro scales."""
    import hdbscan

    labels_dict = {}
    for scale_key, params in HDBSCAN_SCALES.items():
        logger.info(f"HDBSCAN [{scale_key}]: min_cluster_size={params['min_cluster_size']}, "
                     f"min_samples={params['min_samples']}")
        t0 = time.time()

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            gen_min_span_tree=True,
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(umap_2d)
        labels_dict[scale_key] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        elapsed = time.time() - t0
        logger.info(f"  → {n_clusters} clusters, {n_noise:,} noise points ({elapsed:.1f}s)")

    return labels_dict


# ---------------------------------------------------------------------------
# Step 5: Visualization
# ---------------------------------------------------------------------------
def plot_results(umap_2d: np.ndarray, labels_dict: dict, output_path: str):
    """Generate multi-resolution HDBSCAN visualization."""
    n = len(umap_2d)
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), facecolor='#0a0a0a')
    fig.suptitle(
        f"DINOv3 ViT-7B16 Zero-Shot Landscape Typologies — HDBSCAN (N={n:,})",
        fontsize=22, color='white', y=1.02, fontweight='bold'
    )

    for i, (scale_key, params) in enumerate(HDBSCAN_SCALES.items()):
        labels = labels_dict[scale_key]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Color mapping
        cmap = plt.get_cmap('turbo', max(1, n_clusters))
        colors = np.array([
            [0.12, 0.12, 0.12, 0.03] if l == -1 
            else list(cmap(l % cmap.N)[:3]) + [0.6]
            for l in labels
        ])

        # Sort so noise is drawn first (behind clusters)
        sort_idx = np.argsort(labels)
        axes[i].scatter(
            umap_2d[sort_idx, 0], umap_2d[sort_idx, 1],
            c=colors[sort_idx], s=0.8, edgecolors='none', rasterized=True
        )
        axes[i].set_title(
            f"{params['label']}\n{n_clusters} Signatures",
            fontsize=15, color='white', pad=12
        )
        axes[i].set_facecolor('#0a0a0a')
        axes[i].axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    logger.info(f"Saved visualization → {output_path}")
    plt.close(fig)


def _get_swiss_border_normalized():
    """Load Swiss border as normalized [-1,1] coordinates matching embedding space.
    
    Downloads Natural Earth 10m data on first call, caches as geojson.
    Returns list of (easting_array, northing_array) for each polygon ring.
    """
    import geopandas as gpd

    geojson_path = os.path.join(OUTPUT_DIR, "switzerland_border_2056.geojson")
    
    if os.path.exists(geojson_path):
        ch = gpd.read_file(geojson_path)
    else:
        logger.info("Downloading Swiss border from Natural Earth 10m...")
        url = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        world = gpd.read_file(url)
        ch = world[world['NAME'] == 'Switzerland'].to_crs('EPSG:2056')
        ch.to_file(geojson_path, driver='GeoJSON')
        logger.info(f"Cached border → {geojson_path}")

    # EPSG:2056 normalization bounds (must match extract_7b_embeddings.py)
    MIN_E, MAX_E = 2400000.0, 2900000.0
    MIN_N, MAX_N = 1000000.0, 1350000.0

    rings = []
    geom = ch.geometry.values[0]
    
    def _normalize_ring(ring_coords):
        coords = np.array(ring_coords)
        norm_e = 2.0 * (coords[:, 0] - MIN_E) / (MAX_E - MIN_E) - 1.0
        norm_n = 2.0 * (coords[:, 1] - MIN_N) / (MAX_N - MIN_N) - 1.0
        return norm_e, norm_n

    if geom.geom_type == 'Polygon':
        rings.append(_normalize_ring(geom.exterior.coords))
        for interior in geom.interiors:
            rings.append(_normalize_ring(interior.coords))
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            rings.append(_normalize_ring(poly.exterior.coords))
            for interior in poly.interiors:
                rings.append(_normalize_ring(interior.coords))

    return rings


def _norm_to_lv95(norm_val, is_easting=True):
    """Convert normalized [-1,1] value back to EPSG:2056 coordinate."""
    if is_easting:
        return (norm_val + 1.0) / 2.0 * (2900000.0 - 2400000.0) + 2400000.0
    else:
        return (norm_val + 1.0) / 2.0 * (1350000.0 - 1000000.0) + 1000000.0


def plot_spatial_map(coords: np.ndarray, labels: np.ndarray, 
                     scale_name: str, output_path: str):
    """Plot clusters in geographic space with Swiss border and LV95 grid."""
    from matplotlib.ticker import FuncFormatter

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cmap = plt.get_cmap('turbo', max(1, n_clusters))

    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='#0a0a0a')
    
    # --- Swiss border outline ---
    try:
        border_rings = _get_swiss_border_normalized()
        for ring_e, ring_n in border_rings:
            ax.plot(ring_e, ring_n, color='white', linewidth=1.2, alpha=0.85, zorder=10)
            # Subtle fill for land area
            ax.fill(ring_e, ring_n, color='#1a1a1a', alpha=0.3, zorder=1)
    except Exception as e:
        logger.warning(f"Could not load Swiss border: {e}")

    # --- LV95 Grid ---
    # Major grid lines at 100km intervals in EPSG:2056
    grid_eastings = [2500000, 2600000, 2700000, 2800000]
    grid_northings = [1100000, 1200000, 1300000]
    
    for ge in grid_eastings:
        norm_e = 2.0 * (ge - 2400000.0) / (2900000.0 - 2400000.0) - 1.0
        ax.axvline(norm_e, color='#333333', linewidth=0.5, alpha=0.6, zorder=2)
    
    for gn in grid_northings:
        norm_n = 2.0 * (gn - 1000000.0) / (1350000.0 - 1000000.0) - 1.0
        ax.axhline(norm_n, color='#333333', linewidth=0.5, alpha=0.6, zorder=2)

    # --- Data points ---
    # Noise
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                   c='#1a1a1a', s=0.3, alpha=0.05, edgecolors='none', rasterized=True, zorder=3)
    
    # Clusters
    cluster_mask = ~noise_mask
    if cluster_mask.any():
        cluster_colors = [cmap(l % cmap.N) for l in labels[cluster_mask]]
        ax.scatter(coords[cluster_mask, 0], coords[cluster_mask, 1],
                   c=cluster_colors, s=0.5, alpha=0.4, edgecolors='none', rasterized=True, zorder=4)

    # --- Axis formatting with LV95 tick labels ---
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: f"{_norm_to_lv95(val, True)/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: f"{_norm_to_lv95(val, False)/1e6:.2f}M"))

    # Set ticks at grid positions
    e_ticks = [2.0 * (ge - 2400000.0) / 500000.0 - 1.0 for ge in grid_eastings]
    n_ticks = [2.0 * (gn - 1000000.0) / 350000.0 - 1.0 for gn in grid_northings]
    ax.set_xticks(e_ticks)
    ax.set_yticks(n_ticks)

    ax.set_title(
        f"Swiss Landscape Signatures — {scale_name} ({n_clusters} clusters, N={len(labels):,})",
        fontsize=16, color='white', pad=14, fontweight='bold'
    )
    ax.set_xlabel("EPSG:2056 Easting", color='#aaaaaa', fontsize=11)
    ax.set_ylabel("EPSG:2056 Northing", color='#aaaaaa', fontsize=11)
    ax.tick_params(colors='#888888', labelsize=9)
    ax.set_facecolor('#0a0a0a')
    
    # Correct aspect ratio: easting spans 500km, northing spans 350km
    # In normalized space, 1 unit Y = 175km, 1 unit X = 250km
    # So Y must be scaled by 350/500 = 0.7 relative to X
    ax.set_aspect(350.0 / 500.0)
    
    # Clip view to Swiss extent with padding
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.75, 0.85)

    for spine in ax.spines.values():
        spine.set_color('#333333')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    logger.info(f"Saved spatial map → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # --- Step 1: Load embeddings ---
    logger.info("=" * 60)
    logger.info("STEP 1: Loading .pt embeddings")
    logger.info("=" * 60)
    embeddings, coords = load_pt_embeddings(
        PT_DIR,
        cache_path=None  # Skip caching raw — 26GB npz is slow; .pt reload is fast
    )

    # --- Step 2: PCA reduction ---
    logger.info("=" * 60)
    logger.info("STEP 2: PCA dimensionality reduction")
    logger.info("=" * 60)
    pca_emb = reduce_dimensions_pca(
        embeddings, n_components=PCA_DIMS,
        cache_path=os.path.join(OUTPUT_DIR, "pca_128d_cache.npz")
    )
    # Free raw embeddings after PCA
    del embeddings

    # --- Step 3: UMAP ---
    logger.info("=" * 60)
    logger.info("STEP 3: UMAP projection")
    logger.info("=" * 60)
    umap_2d = run_umap(
        pca_emb,
        cache_path=os.path.join(OUTPUT_DIR, "umap_2d_cache.npz")
    )
    del pca_emb

    # --- Step 4: HDBSCAN ---
    logger.info("=" * 60)
    logger.info("STEP 4: Multi-resolution HDBSCAN clustering")
    logger.info("=" * 60)
    labels_dict = run_hdbscan_multiresolution(umap_2d)

    # --- Step 5: Visualize ---
    logger.info("=" * 60)
    logger.info("STEP 5: Visualization")
    logger.info("=" * 60)
    plot_results(umap_2d, labels_dict,
                 os.path.join(OUTPUT_DIR, "zeroshot_7b_hdbscan_typologies.png"))

    # Spatial maps per scale
    for scale_key, params in HDBSCAN_SCALES.items():
        plot_spatial_map(
            coords, labels_dict[scale_key], params['label'],
            os.path.join(OUTPUT_DIR, f"spatial_map_{scale_key}.png")
        )

    # --- Step 6: Export parquet ---
    logger.info("=" * 60)
    logger.info("STEP 6: Export parquet")
    logger.info("=" * 60)
    df = pd.DataFrame({
        "norm_easting":    coords[:, 0],
        "norm_northing":   coords[:, 1],
        "umap_x":          umap_2d[:, 0],
        "umap_y":          umap_2d[:, 1],
        "cluster_micro":   labels_dict["micro"],
        "cluster_meso":    labels_dict["meso"],
        "cluster_macro":   labels_dict["macro"],
    })
    parquet_path = os.path.join(OUTPUT_DIR, "zeroshot_7b_hdbscan_data.parquet")
    df.to_parquet(parquet_path)
    
    # Summary statistics
    for scale_key in HDBSCAN_SCALES:
        col = f"cluster_{scale_key}"
        n_clusters = df[col].nunique() - (1 if -1 in df[col].values else 0)
        n_noise = (df[col] == -1).sum()
        logger.info(f"  {scale_key}: {n_clusters} clusters, {n_noise:,} noise "
                     f"({n_noise/len(df)*100:.1f}%)")

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(df):,} rows → {parquet_path}")
    logger.info(f"Total time: {elapsed/60:.1f} min")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
