"""
Full UMAP + HDBSCAN Clustering — No Shortcuts
================================================
Loads ALL 6.9M 7B embeddings, runs PCA on ALL data,
UMAP fit_transform on ALL data, then multi-resolution HDBSCAN.

No subsampling. No approximations. Checkpointed at every step.

Hardware: 128 GB RAM, 24 cores
Expected time: ~12-20 hours (UMAP dominates)
"""

import os
import sys
import glob
import time
import logging
import gc
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('full_clustering_7b.log')
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PT_DIR       = "/home/jubooz/landscape_signatures/training_data_7b"
OUTPUT_DIR   = "/home/jubooz/landscape_signatures/clustering_7b_results"
PCA_DIMS     = 128
UMAP_DIMS    = 2
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.0
RANDOM_STATE   = 42
PCA_BATCH_SIZE = 50000

# HDBSCAN multi-resolution scales
HDBSCAN_SCALES = {
    "micro":  {"label": "Micro-Typologies (Neighborhoods)",    "min_cluster_size": 30,   "min_samples": 10},
    "meso":   {"label": "Meso-Typologies (Cities & Cantons)",  "min_cluster_size": 200,  "min_samples": 50},
    "macro":  {"label": "Macro-Typologies (Bioregional)",      "min_cluster_size": 800,  "min_samples": 100},
}


# ---------------------------------------------------------------------------
# Step 1: Load all .pt files
# ---------------------------------------------------------------------------
def load_pt_embeddings(pt_dir: str):
    """Two-pass load: count, then pre-allocate float16 array."""
    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files")

    # Pass 1: count
    logger.info("Pass 1: Counting rows...")
    row_counts = []
    for pt_path in pt_files:
        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        row_counts.append(d['embeddings'].shape[0])
        del d
    total = sum(row_counts)
    logger.info(f"  Total: {total:,} embeddings")

    # Pass 2: fill
    logger.info(f"Pass 2: Allocating float16 array ({total:,} × 4096) = {total*4096*2/1e9:.1f} GB")
    embeddings = np.empty((total, 4096), dtype=np.float16)
    coords = np.empty((total, 2), dtype=np.float32)

    offset = 0
    for i, pt_path in enumerate(pt_files):
        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        n = d['embeddings'].shape[0]
        embeddings[offset:offset+n] = d['embeddings'].float().numpy().astype(np.float16)
        coords[offset:offset+n] = d['coords'].numpy()
        offset += n
        del d
        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(pt_files)} files ({offset:,} embeddings)")

    logger.info(f"Loaded: {embeddings.shape[0]:,} × {embeddings.shape[1]}D")
    logger.info(f"Memory: embeddings={embeddings.nbytes/1e9:.1f} GB, coords={coords.nbytes/1e6:.0f} MB")
    return embeddings, coords


# ---------------------------------------------------------------------------
# Step 2: PCA — full fit on ALL data (no subsampling)
# ---------------------------------------------------------------------------
def reduce_pca_full(embeddings: np.ndarray, n_components: int, cache_path: str):
    """IncrementalPCA fit on ALL data in batches, then transform ALL."""
    if os.path.exists(cache_path):
        logger.info(f"[CHECKPOINT] Loading PCA from {cache_path}")
        return np.load(cache_path)['pca_emb']

    n = len(embeddings)
    logger.info(f"PCA: FULL fit on {n:,} points ({embeddings.shape[1]}D → {n_components}D)")
    t0 = time.time()

    ipca = IncrementalPCA(n_components=n_components)

    # Fit on ALL data in batches
    logger.info(f"  Fitting on ALL {n:,} points (batch_size={PCA_BATCH_SIZE})...")
    for start in range(0, n, PCA_BATCH_SIZE):
        end = min(start + PCA_BATCH_SIZE, n)
        batch = embeddings[start:end].astype(np.float32)
        batch = normalize(batch, norm='l2', axis=1)
        ipca.partial_fit(batch)
        if (start // PCA_BATCH_SIZE) % 20 == 0:
            logger.info(f"    PCA fit: {end:,}/{n:,}")

    explained = ipca.explained_variance_ratio_.sum()
    logger.info(f"  PCA fit done: {explained:.1%} variance explained in {time.time()-t0:.0f}s")

    # Transform ALL data in batches
    logger.info(f"  Transforming all {n:,} points...")
    pca_result = np.empty((n, n_components), dtype=np.float32)
    for start in range(0, n, PCA_BATCH_SIZE):
        end = min(start + PCA_BATCH_SIZE, n)
        batch = embeddings[start:end].astype(np.float32)
        batch = normalize(batch, norm='l2', axis=1)
        pca_result[start:end] = ipca.transform(batch)
        if (start // PCA_BATCH_SIZE) % 20 == 0:
            logger.info(f"    PCA transform: {end:,}/{n:,}")

    elapsed = time.time() - t0
    logger.info(f"PCA complete in {elapsed/60:.1f} min")
    logger.info(f"  Output: {pca_result.shape}, {pca_result.nbytes/1e9:.1f} GB")

    np.savez(cache_path, pca_emb=pca_result)
    logger.info(f"  Checkpoint saved → {cache_path}")
    return pca_result


# ---------------------------------------------------------------------------
# Step 3: UMAP — full fit_transform on ALL data (no subsampling)
# ---------------------------------------------------------------------------
def run_umap_full(pca_emb: np.ndarray, cache_path: str):
    """UMAP fit_transform on ALL points. No subsampling."""
    if os.path.exists(cache_path):
        logger.info(f"[CHECKPOINT] Loading UMAP from {cache_path}")
        return np.load(cache_path)['umap_2d']

    import umap

    n = len(pca_emb)
    logger.info(f"UMAP: FULL fit_transform on {n:,} × {pca_emb.shape[1]}D")
    logger.info(f"  n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric=cosine")
    logger.info(f"  This will take many hours. Go do something else.")
    t0 = time.time()

    reducer = umap.UMAP(
        n_components=UMAP_DIMS,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=RANDOM_STATE,
        low_memory=True,
        n_jobs=-1,
        verbose=True,
    )
    umap_2d = reducer.fit_transform(pca_emb)

    elapsed = time.time() - t0
    logger.info(f"UMAP complete in {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")

    np.savez(cache_path, umap_2d=umap_2d.astype(np.float32))
    logger.info(f"  Checkpoint saved → {cache_path}")
    return umap_2d.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: HDBSCAN
# ---------------------------------------------------------------------------
def run_hdbscan(umap_2d: np.ndarray):
    """Multi-resolution HDBSCAN on all points."""
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
        logger.info(f"  → {n_clusters} clusters, {n_noise:,} noise ({time.time()-t0:.1f}s)")

    return labels_dict


# ---------------------------------------------------------------------------
# Step 5: Visualization
# ---------------------------------------------------------------------------
def plot_results(umap_2d, labels_dict, output_path):
    n = len(umap_2d)
    fig, axes = plt.subplots(1, 3, figsize=(30, 10), facecolor='#0a0a0a')
    fig.suptitle(
        f"DINOv3 ViT-7B16 Landscape Typologies — Full HDBSCAN (N={n:,})",
        fontsize=22, color='white', y=1.02, fontweight='bold'
    )
    for i, (scale_key, params) in enumerate(HDBSCAN_SCALES.items()):
        labels = labels_dict[scale_key]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cmap = plt.get_cmap('turbo', max(1, n_clusters))
        colors = np.array([
            [0.12, 0.12, 0.12, 0.03] if l == -1
            else list(cmap(l % cmap.N)[:3]) + [0.6]
            for l in labels
        ])
        sort_idx = np.argsort(labels)
        axes[i].scatter(
            umap_2d[sort_idx, 0], umap_2d[sort_idx, 1],
            c=colors[sort_idx], s=0.8, edgecolors='none', rasterized=True
        )
        axes[i].set_title(f"{params['label']}\n{n_clusters} Signatures",
                          fontsize=15, color='white', pad=12)
        axes[i].set_facecolor('#0a0a0a')
        axes[i].axis('off')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    logger.info(f"Saved → {output_path}")
    plt.close(fig)


def _get_swiss_border_normalized():
    """Load Swiss border as normalized [-1,1] coordinates."""
    import geopandas as gpd
    geojson_path = os.path.join(OUTPUT_DIR, "switzerland_border_2056.geojson")
    if os.path.exists(geojson_path):
        ch = gpd.read_file(geojson_path)
    else:
        logger.info("Downloading Swiss border...")
        url = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        world = gpd.read_file(url)
        ch = world[world['NAME'] == 'Switzerland'].to_crs('EPSG:2056')
        ch.to_file(geojson_path, driver='GeoJSON')

    MIN_E, MAX_E = 2400000.0, 2900000.0
    MIN_N, MAX_N = 1000000.0, 1350000.0
    rings = []
    geom = ch.geometry.values[0]
    def _norm(ring_coords):
        c = np.array(ring_coords)
        return (2.0*(c[:,0]-MIN_E)/(MAX_E-MIN_E)-1.0,
                2.0*(c[:,1]-MIN_N)/(MAX_N-MIN_N)-1.0)
    if geom.geom_type == 'Polygon':
        rings.append(_norm(geom.exterior.coords))
        for interior in geom.interiors:
            rings.append(_norm(interior.coords))
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            rings.append(_norm(poly.exterior.coords))
            for interior in poly.interiors:
                rings.append(_norm(interior.coords))
    return rings


def _norm_to_lv95(val, is_easting=True):
    if is_easting:
        return (val + 1.0) / 2.0 * 500000.0 + 2400000.0
    else:
        return (val + 1.0) / 2.0 * 350000.0 + 1000000.0


def plot_spatial_map(coords, labels, scale_name, output_path):
    from matplotlib.ticker import FuncFormatter
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cmap = plt.get_cmap('turbo', max(1, n_clusters))

    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='#0a0a0a')
    try:
        for ring_e, ring_n in _get_swiss_border_normalized():
            ax.plot(ring_e, ring_n, color='white', linewidth=1.2, alpha=0.85, zorder=10)
            ax.fill(ring_e, ring_n, color='#1a1a1a', alpha=0.3, zorder=1)
    except Exception as e:
        logger.warning(f"Border load failed: {e}")

    for ge in [2500000, 2600000, 2700000, 2800000]:
        ax.axvline(2.0*(ge-2400000.0)/500000.0-1.0, color='#333333', lw=0.5, alpha=0.6, zorder=2)
    for gn in [1100000, 1200000, 1300000]:
        ax.axhline(2.0*(gn-1000000.0)/350000.0-1.0, color='#333333', lw=0.5, alpha=0.6, zorder=2)

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(coords[noise_mask,0], coords[noise_mask,1],
                   c='#1a1a1a', s=0.3, alpha=0.05, edgecolors='none', rasterized=True, zorder=3)
    cluster_mask = ~noise_mask
    if cluster_mask.any():
        ax.scatter(coords[cluster_mask,0], coords[cluster_mask,1],
                   c=[cmap(l%cmap.N) for l in labels[cluster_mask]],
                   s=0.5, alpha=0.4, edgecolors='none', rasterized=True, zorder=4)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{_norm_to_lv95(v,True)/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{_norm_to_lv95(v,False)/1e6:.2f}M"))
    ax.set_title(f"Swiss Landscape Signatures — {scale_name} ({n_clusters} clusters, N={len(labels):,})",
                 fontsize=16, color='white', pad=14, fontweight='bold')
    ax.set_xlabel("EPSG:2056 Easting", color='#aaaaaa', fontsize=11)
    ax.set_ylabel("EPSG:2056 Northing", color='#aaaaaa', fontsize=11)
    ax.tick_params(colors='#888888', labelsize=9)
    ax.set_facecolor('#0a0a0a')
    ax.set_aspect(350.0/500.0)
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.75, 0.85)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    logger.info(f"Saved spatial map → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # --- Step 1: Load ---
    logger.info("=" * 60)
    logger.info("STEP 1: Loading ALL .pt embeddings")
    logger.info("=" * 60)
    embeddings, coords = load_pt_embeddings(PT_DIR)

    # --- Step 2: PCA (full fit) ---
    logger.info("=" * 60)
    logger.info("STEP 2: PCA — FULL fit on ALL data")
    logger.info("=" * 60)
    pca_cache = os.path.join(OUTPUT_DIR, "full_pca_128d.npz")
    pca_emb = reduce_pca_full(embeddings, PCA_DIMS, pca_cache)

    # Free raw embeddings — recover ~56 GB
    logger.info("Freeing raw embeddings...")
    del embeddings
    gc.collect()
    logger.info(f"  PCA output in memory: {pca_emb.nbytes/1e9:.1f} GB")

    # --- Step 3: UMAP (full fit_transform) ---
    logger.info("=" * 60)
    logger.info("STEP 3: UMAP — FULL fit_transform on ALL data")
    logger.info("=" * 60)
    umap_cache = os.path.join(OUTPUT_DIR, "full_umap_2d.npz")
    umap_2d = run_umap_full(pca_emb, umap_cache)

    del pca_emb
    gc.collect()

    # --- Step 4: HDBSCAN ---
    logger.info("=" * 60)
    logger.info("STEP 4: Multi-resolution HDBSCAN")
    logger.info("=" * 60)
    labels_dict = run_hdbscan(umap_2d)

    # --- Step 5: Visualize ---
    logger.info("=" * 60)
    logger.info("STEP 5: Visualization")
    logger.info("=" * 60)
    plot_results(umap_2d, labels_dict,
                 os.path.join(OUTPUT_DIR, "full_7b_hdbscan_typologies.png"))
    for scale_key, params in HDBSCAN_SCALES.items():
        plot_spatial_map(coords, labels_dict[scale_key], params['label'],
                         os.path.join(OUTPUT_DIR, f"full_spatial_map_{scale_key}.png"))

    # --- Step 6: Export ---
    logger.info("=" * 60)
    logger.info("STEP 6: Export parquet")
    logger.info("=" * 60)
    df = pd.DataFrame({
        "norm_easting":  coords[:, 0],
        "norm_northing": coords[:, 1],
        "umap_x":        umap_2d[:, 0],
        "umap_y":        umap_2d[:, 1],
        "cluster_micro": labels_dict["micro"],
        "cluster_meso":  labels_dict["meso"],
        "cluster_macro": labels_dict["macro"],
    })
    parquet_path = os.path.join(OUTPUT_DIR, "full_7b_hdbscan_data.parquet")
    df.to_parquet(parquet_path)

    for scale_key in HDBSCAN_SCALES:
        col = f"cluster_{scale_key}"
        nc = df[col].nunique() - (1 if -1 in df[col].values else 0)
        nn = (df[col] == -1).sum()
        logger.info(f"  {scale_key}: {nc} clusters, {nn:,} noise ({nn/len(df)*100:.1f}%)")

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(df):,} rows → {parquet_path}")
    logger.info(f"Total time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
