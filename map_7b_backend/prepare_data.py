"""
Prepare 7B embedding data for the web application.
Walks .pt shards to rebuild shard/image mapping, converts normalized coords
back to WGS84, and exports compressed binary tiles for the frontend.

Strategy: 3.2M points is too large for JSON. We export:
  1. A parquet with all metadata (served via backend)
  2. Compressed binary ArrayBuffer tiles for frontend rendering (lat/lon/umap/cluster as float32)
"""

import os
import sys
import glob
import time
import struct
import logging
import numpy as np
import pandas as pd
import torch
import tarfile
import json
from pyproj import Transformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PT_DIR = "/home/jubooz/landscape_signatures/training_data_7b"
SHARDS_DIR = "/home/jubooz/landscape_signatures/baukultur_vpr/data/shards"
CLUSTER_PARQUET = "/home/jubooz/landscape_signatures/clustering_7b_results/zeroshot_7b_hdbscan_data.parquet"
OUTPUT_DIR = "/home/jubooz/landscape_signatures/map_7b_frontend"

# EPSG:2056 bounds (from extraction script)
MIN_E, MAX_E = 2400000.0, 2900000.0
MIN_N, MAX_N = 1000000.0, 1350000.0

lv95_to_wgs84 = Transformer.from_crs('epsg:2056', 'epsg:4326', always_xy=True)


def denormalize_to_wgs84(norm_e, norm_n):
    """Convert normalized [-1,1] coords back to WGS84 lat/lon."""
    e = (norm_e + 1.0) / 2.0 * (MAX_E - MIN_E) + MIN_E
    n = (norm_n + 1.0) / 2.0 * (MAX_N - MIN_N) + MIN_N
    lon, lat = lv95_to_wgs84.transform(e, n)
    return lat, lon


def build_shard_image_mapping():
    """Walk .pt files and corresponding .tar shards to build image ID mapping.
    
    Returns list of (shard_name.tar, image_id) tuples in the same order as
    the clustering parquet rows.
    """
    pt_files = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))
    logger.info(f"Building shard→image mapping for {len(pt_files)} .pt files...")
    
    mapping = []
    for i, pt_path in enumerate(pt_files):
        pt_name = os.path.basename(pt_path).replace('.pt', '')
        tar_name = pt_name + '.tar'
        tar_path = os.path.join(SHARDS_DIR, tar_name)
        
        # Load .pt to get count of embeddings in this shard
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        n_emb = data['embeddings'].shape[0]
        
        if os.path.exists(tar_path):
            # Extract image IDs from tar metadata
            try:
                t = tarfile.open(tar_path, 'r:')
                # Get meta.json files to extract image IDs
                meta_members = [m for m in t.getmembers() if m.name.endswith('.meta.json')]
                meta_members.sort(key=lambda m: m.name)
                
                image_ids = []
                for mm in meta_members:
                    f = t.extractfile(mm)
                    if f:
                        meta = json.loads(f.read())
                        image_ids.append(str(meta.get('id', mm.name.split('.')[0])))
                t.close()
                
                # Each image pair produces 2 embeddings (img1 + img2)
                # So n_emb = 2 * len(meta_members)
                for img_id in image_ids:
                    mapping.append((tar_name, img_id))  # img1
                    mapping.append((tar_name, img_id))  # img2
                    
            except Exception as e:
                logger.warning(f"Could not read tar {tar_name}: {e}")
                for j in range(n_emb):
                    mapping.append((tar_name, f"unknown_{j}"))
        else:
            # Tar not available — still record shard name
            for j in range(n_emb):
                mapping.append((tar_name, f"idx_{j}"))
        
        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i+1}/{len(pt_files)} shards ({len(mapping):,} entries)")
    
    logger.info(f"Mapping complete: {len(mapping):,} entries")
    return mapping


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load clustering results
    logger.info("Loading clustering parquet...")
    df = pd.read_parquet(CLUSTER_PARQUET)
    n = len(df)
    logger.info(f"Loaded {n:,} rows")
    
    # Convert normalized coords to WGS84
    logger.info("Converting normalized coords → WGS84...")
    ne = df['norm_easting'].values
    nn = df['norm_northing'].values
    
    # Vectorized conversion
    eastings = (ne + 1.0) / 2.0 * (MAX_E - MIN_E) + MIN_E
    northings = (nn + 1.0) / 2.0 * (MAX_N - MIN_N) + MIN_N
    lons, lats = lv95_to_wgs84.transform(eastings, northings)
    
    df['lat'] = lats.astype(np.float32)
    df['lon'] = lons.astype(np.float32)
    
    logger.info(f"Lat range: [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    logger.info(f"Lon range: [{df['lon'].min():.4f}, {df['lon'].max():.4f}]")
    
    # Build shard mapping
    logger.info("Building shard → image mapping...")
    shard_mapping = build_shard_image_mapping()
    
    if len(shard_mapping) == n:
        df['tar_file'] = [m[0] for m in shard_mapping]
        df['image_id'] = [m[1] for m in shard_mapping]
    else:
        logger.warning(f"Mapping size {len(shard_mapping)} != parquet size {n}. Using index-based IDs.")
        # Reconstruct from .pt file ordering
        pt_files = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))
        tar_files = []
        image_ids = []
        for pt_path in pt_files:
            pt_name = os.path.basename(pt_path).replace('.pt', '')
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
            count = data['embeddings'].shape[0]
            for j in range(count):
                tar_files.append(pt_name + '.tar')
                image_ids.append(f"idx_{j}")
        df['tar_file'] = tar_files[:n]
        df['image_id'] = image_ids[:n]
    
    # Save enriched parquet
    enriched_path = os.path.join(OUTPUT_DIR, "data_7b.parquet")
    df.to_parquet(enriched_path, index=False)
    logger.info(f"Saved enriched parquet: {enriched_path} ({os.path.getsize(enriched_path)/1e6:.1f} MB)")
    
    # Export compact binary for frontend (ArrayBuffer format)
    # Format: 6 float32 per point: [lat, lon, umap_x, umap_y, cluster_meso, cluster_macro]
    logger.info("Exporting binary tile for frontend...")
    binary_data = np.column_stack([
        df['lat'].values.astype(np.float32),
        df['lon'].values.astype(np.float32),
        df['umap_x'].values.astype(np.float32),
        df['umap_y'].values.astype(np.float32),
        df['cluster_micro'].values.astype(np.float32),
        df['cluster_meso'].values.astype(np.float32),
        df['cluster_macro'].values.astype(np.float32),
        df['norm_easting'].values.astype(np.float32),
        df['norm_northing'].values.astype(np.float32),
    ])
    
    bin_path = os.path.join(OUTPUT_DIR, "points.bin")
    binary_data.tofile(bin_path)
    logger.info(f"Binary tile: {bin_path} ({os.path.getsize(bin_path)/1e6:.1f} MB)")
    
    # Also export shard mapping as a separate compact file
    # (shard names as index for lookup)
    unique_shards = sorted(df['tar_file'].unique())
    shard_to_idx = {s: i for i, s in enumerate(unique_shards)}
    shard_indices = df['tar_file'].map(shard_to_idx).values.astype(np.uint16)
    
    shard_idx_path = os.path.join(OUTPUT_DIR, "shard_indices.bin")
    shard_indices.tofile(shard_idx_path)
    
    shard_names_path = os.path.join(OUTPUT_DIR, "shard_names.json")
    with open(shard_names_path, 'w') as f:
        json.dump(unique_shards, f)
    
    logger.info(f"Shard index: {shard_idx_path} ({os.path.getsize(shard_idx_path)/1e6:.1f} MB)")
    
    elapsed = time.time() - t0
    logger.info(f"Data preparation complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
