"""
Step 11: Per-Cluster Spatial & Geodata Validation
=================================================
For each cluster in the reference clustering, computes spatial coherence 
(geographic dispersion), spatial extent, and geodata distributions 
(elevation, distance to roads/water/buildings, category purity).

Outputs:
  - cluster_spatial_validation.parquet
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance, ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from leiden_parallel import REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED

logger = config.setup_logging("s11_spatial")

def get_labels(pca_dim, k, graph_type, res, seed):
    path = config.leiden_labels_path(pca_dim, k, graph_type, res, seed)
    if not os.path.exists(path):
        return None
    return np.load(path)

def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts + 1e-12))

def run():
    logger.info("=" * 60)
    logger.info("STEP 11: Spatial & Geodata Validation")
    logger.info("=" * 60)

    t0 = time.time()

    ref_labels = get_labels(REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED)
    if ref_labels is None:
        logger.error("Reference clustering not found.")
        return

    out_path = config.cluster_spatial_validation_path()
    if os.path.exists(out_path):
        logger.info(f"[CHECKPOINT] Spatial validation already exists: {out_path}")
        return

    # Load data
    logger.info("Loading manifest and geodata...")
    manifest = pd.read_parquet(config.manifest_path(), columns=['lv95_easting', 'lv95_northing', 'global_index'])
    geodata = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "geodata_enriched.parquet"))
    
    # Merge
    df = pd.merge(manifest, geodata, on='global_index', how='inner')
    df['cluster_id'] = ref_labels[df['global_index'].values]

    # Pre-select columns for fast operations
    dist_cols = [
        'nearest_road_distance_m', 'nearest_rail_distance_m',
        'nearest_water_distance_m', 'nearest_forest_distance_m',
        'nearest_building_distance_m'
    ]
    
    cat_cols = ['landcover_class', 'nearest_building_category']

    records = []

    # Group once instead of doing O(N) boolean masks per cluster
    grouped = df.groupby('cluster_id', sort=False)
    n_clusters = len(grouped)
    logger.info(f"Computing spatial/geodata metrics for {n_clusters} clusters...")

    for i, (c, c_df) in enumerate(grouped):
        n_members = len(c_df)

        if n_members == 0:
            continue
            
        record = {
            'cluster_id': c,
            'cluster_size': n_members
        }
        
        # Spatial Extent (Bounding box area)
        e_pts = c_df['lv95_easting'].values
        n_pts = c_df['lv95_northing'].values
        
        bbox_area = (e_pts.max() - e_pts.min()) * (n_pts.max() - n_pts.min())
        record['bbox_area_m2'] = bbox_area
        
        # Convex hull area (only if >= 3 unique points)
        pts = np.column_stack((e_pts, n_pts))
        unique_pts = np.unique(pts, axis=0)
        if len(unique_pts) >= 3:
            try:
                hull = ConvexHull(unique_pts)
                record['convex_hull_area_m2'] = hull.volume # For 2D, volume is area
            except Exception:
                record['convex_hull_area_m2'] = 0.0
        else:
            record['convex_hull_area_m2'] = 0.0

        # Sample for O(n^2) operations
        if n_members > 1000:
            sample_df = c_df.sample(n=1000, random_state=42)
        else:
            sample_df = c_df
            
        # Geographic dispersion (median pairwise distance)
        sample_pts = sample_df[['lv95_easting', 'lv95_northing']].values
        if len(sample_pts) > 1:
            pw_dists = distance.pdist(sample_pts, 'euclidean')
            record['spatial_coherence_score'] = np.median(pw_dists) # smaller is more coherent
        else:
            record['spatial_coherence_score'] = 0.0

        # Elevation summary
        elevs = c_df['elevation_m'].dropna()
        if len(elevs) > 0:
            record['elevation_mean'] = elevs.mean()
            record['elevation_std'] = elevs.std()
            record['elevation_p25'] = elevs.quantile(0.25)
            record['elevation_p50'] = elevs.median()
            record['elevation_p75'] = elevs.quantile(0.75)
        else:
            record['elevation_mean'] = np.nan
            record['elevation_std'] = np.nan
            record['elevation_p25'] = np.nan
            record['elevation_p50'] = np.nan
            record['elevation_p75'] = np.nan

        # Distances to features
        for col in dist_cols:
            vals = c_df[col].dropna()
            if len(vals) > 0:
                record[f'{col}_mean'] = vals.mean()
                record[f'{col}_median'] = vals.median()
                record[f'{col}_p90'] = vals.quantile(0.90)
            else:
                record[f'{col}_mean'] = np.nan
                record[f'{col}_median'] = np.nan
                record[f'{col}_p90'] = np.nan

        # Dominant geodata categories & purity
        # We will combine landcover and building category to find a "dominant category"
        cats = c_df['landcover_class'].astype(str) + " / " + c_df['nearest_building_category'].astype(str)
        val_counts = cats.value_counts()
        
        if len(val_counts) > 0:
            top3 = val_counts.head(3).index.tolist()
            record['dominant_geodata_1'] = top3[0] if len(top3) > 0 else ""
            record['dominant_geodata_2'] = top3[1] if len(top3) > 1 else ""
            record['dominant_geodata_3'] = top3[2] if len(top3) > 2 else ""
            
            record['geodata_purity_score'] = val_counts.iloc[0] / len(cats)
            record['geodata_entropy'] = compute_entropy(cats)
        else:
            record['dominant_geodata_1'] = ""
            record['geodata_purity_score'] = 0.0
            record['geodata_entropy'] = 0.0
            
        records.append(record)

        if (i+1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{n_clusters} clusters...")

    res_df = pd.DataFrame(records)
    res_df.to_parquet(out_path, index=False)
    
    logger.info(f"Saved cluster spatial validation to {out_path}")
    logger.info(f"Time: {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    run()
