"""
10-Scale HDBSCAN Re-clustering
================================
Loads cached UMAP 2D results and coords, runs HDBSCAN at 10 resolution
levels from ultra-fine (~40K clusters) to ultra-coarse (~50 clusters).

Uses cached full_umap_2d.npz — no need to redo PCA/UMAP.
Expected time: ~30 min (10 × ~3 min per scale)
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('recluster_10_scales.log')
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PT_DIR     = "/home/jubooz/landscape_signatures/training_data_7b"
OUTPUT_DIR = "/home/jubooz/landscape_signatures/clustering_7b_results"
UMAP_CACHE = os.path.join(OUTPUT_DIR, "full_umap_2d.npz")

# 10 resolution scales — from finest to coarsest
# Tuned based on prior run: mcs=800 → 518 clusters on 6.9M points
HDBSCAN_SCALES = {
    "scale_01_hyperlocal":  {"label": "Hyper-Local (Street-Level)",       "min_cluster_size": 25,    "min_samples": 5},
    "scale_02_micro":       {"label": "Micro (Neighborhoods)",            "min_cluster_size": 50,    "min_samples": 10},
    "scale_03_local":       {"label": "Local (Districts)",                "min_cluster_size": 150,   "min_samples": 25},
    "scale_04_district":    {"label": "District (Quarters)",              "min_cluster_size": 400,   "min_samples": 50},
    "scale_05_municipal":   {"label": "Municipal (Towns)",                "min_cluster_size": 1000,  "min_samples": 80},
    "scale_06_cantonal":    {"label": "Cantonal (Sub-Regions)",           "min_cluster_size": 2500,  "min_samples": 150},
    "scale_07_regional":    {"label": "Regional (Cantons)",               "min_cluster_size": 6000,  "min_samples": 300},
    "scale_08_landscape":   {"label": "Landscape (Macro-Regions)",        "min_cluster_size": 15000, "min_samples": 500},
    "scale_09_bioregional": {"label": "Bioregional (Major Zones)",        "min_cluster_size": 35000, "min_samples": 1000},
    "scale_10_continental": {"label": "Continental (Fundamental Types)",   "min_cluster_size": 80000, "min_samples": 2000},
}


def load_coords(pt_dir):
    """Reload coords from .pt files (small, fast)."""
    pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
    logger.info(f"Loading coords from {len(pt_files)} .pt files...")
    
    # Count first
    counts = []
    for f in pt_files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        counts.append(d['embeddings'].shape[0])
        del d
    total = sum(counts)
    
    coords = np.empty((total, 2), dtype=np.float32)
    offset = 0
    for f in pt_files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        n = d['coords'].shape[0]
        coords[offset:offset+n] = d['coords'].numpy()
        offset += n
        del d
    
    logger.info(f"  Loaded {total:,} coordinate pairs")
    return coords


def run_hdbscan_10_scales(umap_2d):
    """Run HDBSCAN at all 10 resolution scales."""
    import hdbscan

    labels_dict = {}
    for scale_key, params in HDBSCAN_SCALES.items():
        logger.info(f"HDBSCAN [{scale_key}]: min_cluster_size={params['min_cluster_size']}, "
                     f"min_samples={params['min_samples']}")
        
        checkpoint_path = os.path.join(OUTPUT_DIR, f"{scale_key}_labels.npy")
        if os.path.exists(checkpoint_path):
            logger.info(f"  Loading checkpoint from {checkpoint_path}")
            labels = np.load(checkpoint_path)
            labels_dict[scale_key] = labels
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            pct_noise = n_noise / len(labels) * 100
            logger.info(f"  → {n_clusters} clusters, {n_noise:,} noise ({pct_noise:.1f}%) [from cache]")
            continue

        t0 = time.time()
        n_jobs = 1 if params['min_samples'] >= 1000 else 4
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            gen_min_span_tree=True,
            core_dist_n_jobs=n_jobs,
        )
        labels = clusterer.fit_predict(umap_2d)
        labels_dict[scale_key] = labels

        # Save checkpoint
        np.save(checkpoint_path, labels)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        pct_noise = n_noise / len(labels) * 100
        logger.info(f"  → {n_clusters} clusters, {n_noise:,} noise ({pct_noise:.1f}%) [{time.time()-t0:.1f}s]")

        # Clean up memory to prevent OOM on subsequent iterations
        del clusterer
        gc.collect()

    return labels_dict


def plot_all_scales(umap_2d, labels_dict, output_dir):
    """Generate a grid of all 10 scales + individual plots."""
    n = len(umap_2d)
    
    # --- Big 2×5 grid ---
    fig, axes = plt.subplots(2, 5, figsize=(50, 20), facecolor='#0a0a0a')
    fig.suptitle(
        f"DINOv3 ViT-7B16 — 10-Scale Landscape Typologies (N={n:,})",
        fontsize=28, color='white', y=1.02, fontweight='bold'
    )
    
    for i, (scale_key, params) in enumerate(HDBSCAN_SCALES.items()):
        ax = axes[i // 5, i % 5]
        labels = labels_dict[scale_key]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        cmap = plt.get_cmap('turbo', max(1, n_clusters))
        colors = np.array([
            [0.12, 0.12, 0.12, 0.03] if l == -1
            else list(cmap(l % cmap.N)[:3]) + [0.6]
            for l in labels
        ])
        sort_idx = np.argsort(labels)
        ax.scatter(
            umap_2d[sort_idx, 0], umap_2d[sort_idx, 1],
            c=colors[sort_idx], s=0.3, edgecolors='none', rasterized=True
        )
        n_noise = np.sum(labels == -1)
        ax.set_title(f"{params['label']}\n{n_clusters} clusters ({n_noise/n*100:.0f}% noise)",
                      fontsize=12, color='white', pad=8)
        ax.set_facecolor('#0a0a0a')
        ax.axis('off')
    
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "full_10scale_grid.png")
    fig.savefig(grid_path, dpi=200, bbox_inches='tight', facecolor='#0a0a0a')
    logger.info(f"Saved grid → {grid_path}")
    plt.close(fig)


def _get_swiss_border_normalized():
    import geopandas as gpd
    geojson_path = os.path.join(OUTPUT_DIR, "switzerland_border_2056.geojson")
    if os.path.exists(geojson_path):
        ch = gpd.read_file(geojson_path)
    else:
        url = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        world = gpd.read_file(url)
        ch = world[world['NAME'] == 'Switzerland'].to_crs('EPSG:2056')
        ch.to_file(geojson_path, driver='GeoJSON')
    MIN_E, MAX_E = 2400000.0, 2900000.0
    MIN_N, MAX_N = 1000000.0, 1350000.0
    rings = []
    geom = ch.geometry.values[0]
    def _norm(rc):
        c = np.array(rc)
        return (2.0*(c[:,0]-MIN_E)/(MAX_E-MIN_E)-1.0, 2.0*(c[:,1]-MIN_N)/(MAX_N-MIN_N)-1.0)
    if geom.geom_type == 'Polygon':
        rings.append(_norm(geom.exterior.coords))
        for interior in geom.interiors: rings.append(_norm(interior.coords))
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            rings.append(_norm(poly.exterior.coords))
            for interior in poly.interiors: rings.append(_norm(interior.coords))
    return rings


def plot_spatial_maps(coords, labels_dict, output_dir):
    """Spatial maps for each scale."""
    from matplotlib.ticker import FuncFormatter
    
    def _lv95(v, is_e=True):
        return (v+1.0)/2.0*(500000.0 if is_e else 350000.0)+(2400000.0 if is_e else 1000000.0)
    
    try:
        border = _get_swiss_border_normalized()
    except Exception as e:
        logger.warning(f"Border failed: {e}")
        border = []

    for scale_key, params in HDBSCAN_SCALES.items():
        labels = labels_dict[scale_key]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cmap = plt.get_cmap('turbo', max(1, n_clusters))

        fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor='#0a0a0a')
        for ring_e, ring_n in border:
            ax.plot(ring_e, ring_n, color='white', linewidth=1.2, alpha=0.85, zorder=10)
            ax.fill(ring_e, ring_n, color='#1a1a1a', alpha=0.3, zorder=1)

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(coords[noise_mask,0], coords[noise_mask,1],
                       c='#1a1a1a', s=0.3, alpha=0.05, edgecolors='none', rasterized=True, zorder=3)
        cm = ~noise_mask
        if cm.any():
            ax.scatter(coords[cm,0], coords[cm,1],
                       c=[cmap(l%cmap.N) for l in labels[cm]],
                       s=0.5, alpha=0.4, edgecolors='none', rasterized=True, zorder=4)

        ax.set_title(f"{params['label']} — {n_clusters} clusters (N={len(labels):,})",
                     fontsize=16, color='white', pad=14, fontweight='bold')
        ax.set_facecolor('#0a0a0a')
        ax.set_aspect(350.0/500.0)
        ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.75, 0.85)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{_lv95(v,True)/1e6:.1f}M"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{_lv95(v,False)/1e6:.2f}M"))
        ax.tick_params(colors='#888888', labelsize=9)
        for spine in ax.spines.values(): spine.set_color('#333333')
        plt.tight_layout()
        
        path = os.path.join(output_dir, f"full_spatial_{scale_key}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
        plt.close(fig)
        logger.info(f"  Saved {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    # Load cached UMAP
    logger.info("Loading cached UMAP results...")
    umap_2d = np.load(UMAP_CACHE)['umap_2d']
    logger.info(f"  UMAP: {umap_2d.shape}")

    # Load coords
    coords = load_coords(PT_DIR)

    # Run 10-scale HDBSCAN
    logger.info("=" * 60)
    logger.info("HDBSCAN: 10 resolution scales")
    logger.info("=" * 60)
    labels_dict = run_hdbscan_10_scales(umap_2d)

    # Visualize
    logger.info("=" * 60)
    logger.info("Generating visualizations...")
    logger.info("=" * 60)
    plot_all_scales(umap_2d, labels_dict, OUTPUT_DIR)
    plot_spatial_maps(coords, labels_dict, OUTPUT_DIR)

    # Export parquet
    logger.info("=" * 60)
    logger.info("Exporting parquet...")
    logger.info("=" * 60)
    df_data = {
        "norm_easting":  coords[:, 0],
        "norm_northing": coords[:, 1],
        "umap_x":        umap_2d[:, 0],
        "umap_y":        umap_2d[:, 1],
    }
    for scale_key in HDBSCAN_SCALES:
        df_data[f"cluster_{scale_key}"] = labels_dict[scale_key]

    df = pd.DataFrame(df_data)
    parquet_path = os.path.join(OUTPUT_DIR, "full_10scale_hdbscan_data.parquet")
    df.to_parquet(parquet_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for scale_key, params in HDBSCAN_SCALES.items():
        col = f"cluster_{scale_key}"
        nc = df[col].nunique() - (1 if -1 in df[col].values else 0)
        nn = (df[col] == -1).sum()
        logger.info(f"  {scale_key:30s}: {nc:>6} clusters, {nn:>10,} noise ({nn/len(df)*100:.1f}%)")

    elapsed = time.time() - t_start
    logger.info(f"\nCOMPLETE: {len(df):,} rows → {parquet_path}")
    logger.info(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
