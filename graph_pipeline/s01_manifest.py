"""
Step 01: Manifest Construction
================================
Builds a manifest DataFrame with one row per image embedding.
Scans all .pt files, assigns stable image IDs, records metadata.

Output: manifest.parquet
"""

import os
import sys
import json
import glob
import time
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s01_manifest")


def run():
    output_path = config.manifest_path()
    if os.path.exists(output_path):
        logger.info(f"[CHECKPOINT] Manifest already exists: {output_path}")
        df = pd.read_parquet(output_path)
        logger.info(f"  {len(df):,} rows")
        return df

    pt_files = sorted(glob.glob(os.path.join(config.PT_DIR, "*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files in {config.PT_DIR}")

    t0 = time.time()

    # ---- Pass 1: Count rows per file ----
    logger.info("Pass 1: Counting rows per file...")
    file_infos = []
    for i, pt_path in enumerate(pt_files):
        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        n = d['embeddings'].shape[0]
        file_infos.append({
            'file_index': i,
            'file_path': pt_path,
            'file_name': os.path.basename(pt_path),
            'n_rows': n,
        })
        del d
        if (i + 1) % 500 == 0:
            logger.info(f"  Scanned {i + 1}/{len(pt_files)} files")

    total_rows = sum(f['n_rows'] for f in file_infos)
    logger.info(f"  Total: {total_rows:,} embeddings across {len(pt_files)} files")

    # ---- Pass 2: Build manifest ----
    logger.info("Pass 2: Building manifest with metadata...")

    records = []
    global_idx = 0

    for fi_idx, fi in enumerate(file_infos):
        pt_path = fi['file_path']
        fname = fi['file_name']
        n = fi['n_rows']

        # Determine source dataset
        if 'apple-recovery' in fname:
            source_dataset = 'apple-recovery'
        elif 'unified' in fname:
            source_dataset = 'unified'
        else:
            source_dataset = 'unknown'

        # Extract file number from name
        parts = fname.replace('.pt', '').split('-')
        file_number = parts[-1] if parts else str(fi['file_index'])

        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        embeddings = d['embeddings']  # bfloat16
        coords = d['coords'].numpy()  # float32 [N, 2]

        # Compute embedding norms (cast to float32 for accuracy)
        emb_f32 = embeddings.float().numpy()
        norms = np.linalg.norm(emb_f32, axis=1)

        for row_idx in range(n):
            ne = float(coords[row_idx, 0])
            nn = float(coords[row_idx, 1])

            # Reverse normalized coords to LV95
            lv95_e = (ne + 1.0) / 2.0 * (config.MAX_EASTING - config.MIN_EASTING) + config.MIN_EASTING
            lv95_n = (nn + 1.0) / 2.0 * (config.MAX_NORTHING - config.MIN_NORTHING) + config.MIN_NORTHING

            image_id = f"{source_dataset}_{file_number}_{row_idx:06d}"

            records.append({
                'image_id': image_id,
                'global_index': global_idx,
                'embedding_file': fname,
                'file_index': fi['file_index'],
                'row_index_in_file': row_idx,
                'norm_easting': ne,
                'norm_northing': nn,
                'lv95_easting': lv95_e,
                'lv95_northing': lv95_n,
                'source_dataset': source_dataset,
                'embedding_norm': float(norms[row_idx]),
                'is_zero_vector': bool(norms[row_idx] < 1e-6),
                'is_low_norm': bool(norms[row_idx] < 0.1),
            })
            global_idx += 1

        del d, embeddings, emb_f32, norms, coords

        if (fi_idx + 1) % 200 == 0:
            logger.info(f"  Processed {fi_idx + 1}/{len(file_infos)} files "
                        f"({global_idx:,} rows)")

    logger.info(f"  Built {len(records):,} manifest rows")

    # ---- Create DataFrame ----
    df = pd.DataFrame(records)

    # Quality flags
    n_zero = df['is_zero_vector'].sum()
    n_low = df['is_low_norm'].sum()
    logger.info(f"  Quality: {n_zero} zero vectors, {n_low} low-norm vectors")

    # Check for duplicate coordinates (potential duplicate images)
    coord_pairs = df[['norm_easting', 'norm_northing']].round(8)
    dup_mask = coord_pairs.duplicated(keep=False)
    df['is_duplicate_coord'] = dup_mask
    n_dup = dup_mask.sum()
    logger.info(f"  Duplicate coordinates: {n_dup:,} rows "
                f"({n_dup / len(df) * 100:.1f}%)")

    # Save
    df.to_parquet(output_path + ".tmp", index=False)
    os.rename(output_path + ".tmp", output_path)

    elapsed = time.time() - t0
    logger.info(f"Manifest saved: {output_path}")
    logger.info(f"  {len(df):,} rows, {elapsed / 60:.1f} min")

    return df


if __name__ == "__main__":
    run()
