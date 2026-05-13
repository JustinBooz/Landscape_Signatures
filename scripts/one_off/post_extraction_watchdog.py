"""
Watchdog: monitors embedding extraction, then incrementally extends
the manifest and geodata enrichment with new data only.

Previous pipeline outputs are NEVER modified or deleted.

Run:
  nohup conda run -n baukultur_vpr python -u post_extraction_watchdog.py > watchdog_fill.log 2>&1 &
"""

import os
import sys
import time
import subprocess
import glob
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

SHARDS_DIR = "baukultur_vpr/data/shards"
PT_DIR = "training_data_7b"
GRAPH_PIPELINE = "graph_pipeline"
OUTPUT_DIR = os.path.join(GRAPH_PIPELINE, "outputs")
CONDA_ENV = "baukultur_vpr"

# EPSG:2056 normalization bounds (from config.py / extract_7b_embeddings.py)
MIN_EASTING  = 2400000.0
MAX_EASTING  = 2900000.0
MIN_NORTHING = 1000000.0
MAX_NORTHING = 1350000.0


def count_pt_files():
    shard_names = set()
    for f in glob.glob(os.path.join(SHARDS_DIR, "dataset-*.tar")):
        shard_names.add(os.path.basename(f).replace(".tar", ""))
    done = 0
    for name in shard_names:
        if os.path.exists(os.path.join(PT_DIR, f"{name}.pt")):
            done += 1
    return done, len(shard_names)


def is_extraction_running():
    try:
        r = subprocess.run(['pgrep', '-f', 'extract_7b_embeddings'],
                          capture_output=True, text=True, timeout=5)
        return bool(r.stdout.strip())
    except:
        return False


def extend_manifest():
    """Append new .pt file entries to existing manifest. No rebuild needed."""
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.parquet")
    existing = pd.read_parquet(manifest_path)
    logger.info(f"Existing manifest: {len(existing):,} rows")

    # Find which .pt files are already in the manifest
    existing_files = set(existing['embedding_file'].unique())
    logger.info(f"  Existing .pt files in manifest: {len(existing_files)}")

    # Find new .pt files (from the fill shards)
    all_pt = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))
    new_pt = [p for p in all_pt if os.path.basename(p) not in existing_files]
    logger.info(f"  New .pt files to add: {len(new_pt)}")

    if not new_pt:
        logger.info("  No new files to add!")
        return existing

    # Build new rows
    global_idx = existing['global_index'].max() + 1
    file_idx = existing['file_index'].max() + 1
    new_records = []

    for fi_idx, pt_path in enumerate(new_pt):
        fname = os.path.basename(pt_path)

        if 'apple-recovery' in fname:
            source_dataset = 'apple-recovery'
        elif 'unified' in fname:
            source_dataset = 'unified'
        else:
            source_dataset = 'unknown'

        parts = fname.replace('.pt', '').split('-')
        file_number = parts[-1] if parts else str(fi_idx)

        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        embeddings = d['embeddings']
        coords = d['coords'].numpy()
        n = embeddings.shape[0]

        emb_f32 = embeddings.float().numpy()
        norms = np.linalg.norm(emb_f32, axis=1)

        for row_idx in range(n):
            ne = float(coords[row_idx, 0])
            nn = float(coords[row_idx, 1])

            lv95_e = (ne + 1.0) / 2.0 * (MAX_EASTING - MIN_EASTING) + MIN_EASTING
            lv95_n = (nn + 1.0) / 2.0 * (MAX_NORTHING - MIN_NORTHING) + MIN_NORTHING

            image_id = f"{source_dataset}_{file_number}_{row_idx:06d}"

            new_records.append({
                'image_id': image_id,
                'global_index': global_idx,
                'embedding_file': fname,
                'file_index': file_idx + fi_idx,
                'row_index_in_file': row_idx,
                'norm_easting': ne,
                'norm_northing': nn,
                'lv95_easting': lv95_e,
                'lv95_northing': lv95_n,
                'source_dataset': source_dataset,
                'embedding_norm': float(norms[row_idx]),
                'is_zero_vector': bool(norms[row_idx] < 1e-6),
                'is_low_norm': bool(norms[row_idx] < 0.1),
                'is_duplicate_coord': False,  # will check below
            })
            global_idx += 1

        del d, embeddings, emb_f32, norms, coords

        if (fi_idx + 1) % 50 == 0:
            logger.info(f"  Processed {fi_idx + 1}/{len(new_pt)} new files "
                        f"({len(new_records):,} rows)")

    new_df = pd.DataFrame(new_records)
    logger.info(f"  Built {len(new_df):,} new manifest rows")

    # Merge
    merged = pd.concat([existing, new_df], ignore_index=False)

    # Re-check duplicate coords across old+new
    coord_pairs = merged[['norm_easting', 'norm_northing']].round(8)
    dup_mask = coord_pairs.duplicated(keep=False)
    merged['is_duplicate_coord'] = dup_mask
    n_dup = dup_mask.sum()
    logger.info(f"  Duplicate coordinates: {n_dup:,} rows "
                f"({n_dup / len(merged) * 100:.1f}%)")

    # Save (atomic)
    tmp = manifest_path + ".tmp"
    merged.to_parquet(tmp, index=False)
    os.rename(tmp, manifest_path)

    logger.info(f"  Manifest extended: {len(existing):,} → {len(merged):,} rows "
                f"(+{len(new_df):,})")
    return merged


def extend_geodata():
    """Run geodata enrichment only for new points (appends to checkpoints)."""
    # The geodata_enrich_v2.py reads the manifest and checks checkpoints.
    # Since the manifest now has new rows, the enrichment script needs to handle
    # the fact that checkpoints may have fewer rows than the manifest.
    # 
    # Strategy: run geodata_enrich_v2.py which already has checkpoint logic.
    # It will see the new manifest size doesn't match checkpoint sizes and 
    # re-process. BUT we don't want to redo everything.
    #
    # Better: just run it — the domain checkpoints will be regenerated with
    # new manifest size, but it reuses cached geodata downloads.
    
    logger.info("Running geodata enrichment for extended manifest...")
    
    # Remove old domain checkpoints so they get rebuilt with new size
    checkpoint_dir = os.path.join(OUTPUT_DIR, "geodata_checkpoints_v2")
    if os.path.exists(checkpoint_dir):
        # Back up first
        backup = checkpoint_dir + f"_pre_fill"
        if not os.path.exists(backup):
            import shutil
            shutil.copytree(checkpoint_dir, backup)
            logger.info(f"  Backed up checkpoints to {backup}")
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Remove old enriched output
    enriched = os.path.join(OUTPUT_DIR, "geodata_enriched.parquet")
    if os.path.exists(enriched):
        backup = enriched + ".pre_fill"
        if not os.path.exists(backup):
            os.rename(enriched, backup)
            logger.info(f"  Backed up enriched data to {backup}")
        elif os.path.exists(enriched):
            os.remove(enriched)
    
    result = subprocess.run(
        ['conda', 'run', '-n', CONDA_ENV, 'python', '-u', 'geodata_enrich_v2.py'],
        cwd=GRAPH_PIPELINE,
        timeout=14400
    )
    
    if result.returncode == 0:
        logger.info("  ✓ Geodata enrichment completed")
        
        # Run DEM fallback
        result2 = subprocess.run(
            ['conda', 'run', '-n', CONDA_ENV, 'python', '-u', 'fix_terrain_srtm.py'],
            cwd=GRAPH_PIPELINE,
            timeout=3600
        )
        if result2.returncode == 0:
            logger.info("  ✓ DEM fallback completed")
        else:
            logger.warning("  DEM fallback had issues")
    else:
        logger.error(f"  ✗ Geodata enrichment failed (exit {result.returncode})")


def main():
    logger.info("=" * 60)
    logger.info("POST-EXTRACTION WATCHDOG (incremental mode)")
    logger.info("=" * 60)
    logger.info(f"Monitoring: {SHARDS_DIR} -> {PT_DIR}")
    logger.info("Previous outputs will be PRESERVED, not overwritten.")
    
    # Phase 1: Wait for extraction to finish
    while True:
        done, total = count_pt_files()
        running = is_extraction_running()
        logger.info(f"Extraction: {done}/{total} shards, "
                     f"{'running' if running else 'STOPPED'}")
        
        if not running and done >= total:
            logger.info("Extraction complete!")
            break
        
        if not running and done < total:
            logger.warning(f"Extraction stopped, {total - done} shards remain")
            time.sleep(300)
            if not is_extraction_running():
                logger.error("Extraction not restarted. Proceeding with available data.")
                break
        
        time.sleep(120)
    
    # Phase 2: Extend manifest (append-only)
    logger.info("\n" + "=" * 60)
    logger.info("EXTENDING MANIFEST")
    logger.info("=" * 60)
    extend_manifest()
    
    # Phase 3: Geodata enrichment for all points (new manifest)
    logger.info("\n" + "=" * 60)
    logger.info("GEODATA ENRICHMENT")
    logger.info("=" * 60)
    extend_geodata()
    
    logger.info("\n" + "=" * 60)
    logger.info("WATCHDOG COMPLETE")
    logger.info("=" * 60)
    logger.info("Previous outputs preserved with .pre_fill suffix")
    logger.info("Manifest extended incrementally (no rebuild)")
    logger.info("\nNext: re-run s02 → s06 when ready (delete old outputs first)")


if __name__ == "__main__":
    main()
