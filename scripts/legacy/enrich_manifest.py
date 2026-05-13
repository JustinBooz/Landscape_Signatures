"""
Manifest Enrichment — Incremental Metadata Extraction from WebDataset Shards
===============================================================================
Designed to run incrementally across multiple external drives. Each time you
connect a drive containing .tar shards, run this script to extract metadata
and merge it into the manifest.

Usage:
    # Auto-scan default shards directory
    conda run -n baukultur_vpr python enrich_manifest.py

    # Scan a specific directory (e.g., external drive)
    conda run -n baukultur_vpr python enrich_manifest.py --shards /media/jubooz/External_2/shards

    # Scan multiple directories
    conda run -n baukultur_vpr python enrich_manifest.py \
        --shards /home/jubooz/landscape_signatures/baukultur_vpr/data/shards \
        --shards /media/jubooz/External_2/shards \
        --shards /media/jubooz/Drive3/shards

    # Check current enrichment progress
    conda run -n baukultur_vpr python enrich_manifest.py --status

Metadata extracted per image:
    - sample_key:    WebDataset sample identifier (e.g., unified_4333098889791628270_face0_face2)
    - image_type:    'img1' or 'img2' (which image in the pair)
    - lat:           WGS84 latitude
    - lon:           WGS84 longitude
    - capture_date:  ISO datetime of image capture
    - source:        Data source (e.g., 'apple', 'google', 'mapillary')
    - is_ugc:        Whether image is user-generated content
    - pano_id:       Panorama identifier
    - suffix:        Face pair suffix (e.g., 'face0_face2')

The manifest is updated in-place with new columns. Already-enriched rows
are skipped. Progress is tracked in a sidecar JSON file.
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("enrich_manifest")

# Columns added by enrichment
ENRICHMENT_COLUMNS = [
    'sample_key', 'image_type', 'lat', 'lon', 'capture_date',
    'source', 'is_ugc', 'pano_id', 'suffix',
]

PROGRESS_FILE = os.path.join(config.OUTPUT_DIR, "enrichment_progress.json")

# Coordinate transform for matching
_wgs84_to_lv95 = Transformer.from_crs('epsg:4326', 'epsg:2056', always_xy=True)

def _normalize_coord(lat, lon):
    """Convert WGS84 to normalized LV95, matching extract_7b_embeddings.py."""
    e, n = _wgs84_to_lv95.transform(lon, lat)
    ne = 2.0 * (e - config.MIN_EASTING) / (config.MAX_EASTING - config.MIN_EASTING) - 1.0
    nn = 2.0 * (n - config.MIN_NORTHING) / (config.MAX_NORTHING - config.MIN_NORTHING) - 1.0
    return ne, nn


def _load_progress():
    """Load set of already-processed shard filenames."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
        return set(data.get('completed_shards', []))
    return set()


def _save_progress(completed):
    """Save set of completed shard filenames."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            'completed_shards': sorted(completed),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_completed': len(completed),
        }, f, indent=2)


def _extract_metadata_from_shard(tar_path):
    """Extract metadata from all samples in a .tar shard.
    
    Returns a list of dicts, two per sample (one for img1, one for img2),
    in the same order as the extraction script produces embeddings:
    [all_img1_records..., all_img2_records...]
    """
    import webdataset as wds

    img1_records = []
    img2_records = []

    ds = wds.WebDataset([tar_path], shardshuffle=False, empty_check=False,
                         handler=wds.warn_and_continue)

    for sample in ds:
        # Extract key
        key = sample.get('__key__', b'')
        if isinstance(key, bytes):
            key = key.decode('utf-8', errors='replace')

        # Extract metadata
        meta_raw = sample.get('meta.json', b'{}')
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode('utf-8', errors='replace')
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = {}

        record = {
            'sample_key': key,
            'lat': meta.get('lat'),
            'lon': meta.get('lon'),
            'capture_date': meta.get('date'),
            'source': meta.get('source'),
            'is_ugc': meta.get('is_ugc'),
            'pano_id': str(meta.get('pano_id', '')),
            'suffix': meta.get('suffix', ''),
        }

        # img1 record
        r1 = record.copy()
        r1['image_type'] = 'img1'
        img1_records.append(r1)

        # img2 record
        r2 = record.copy()
        r2['image_type'] = 'img2'
        img2_records.append(r2)

    # The extraction script collates as [all_img1..., all_img2...]
    # But it processes in DataLoader batches. Within each shard the
    # full shard is processed in sequence, so the order is:
    # [img1_sample0, img1_sample1, ..., img2_sample0, img2_sample1, ...]
    return img1_records + img2_records


def _find_shards(shard_dirs):
    """Find all .tar shards across directories, deduplicating by filename."""
    shard_map = {}  # basename -> full_path
    for d in shard_dirs:
        if not os.path.exists(d):
            logger.warning(f"Directory not found: {d}")
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.tar"))):
            basename = os.path.basename(f)
            if basename not in shard_map:
                shard_map[basename] = f
    return shard_map


def _get_manifest_shard_index(manifest_df):
    """Build a lookup: embedding_file -> list of (global_index, row_index_in_file).
    This maps PT filenames to their manifest rows.
    """
    index = {}
    for _, row in manifest_df[['embedding_file', 'global_index', 'row_index_in_file']].iterrows():
        fname = row['embedding_file']
        if fname not in index:
            index[fname] = []
        index[fname].append((row['global_index'], row['row_index_in_file']))
    return index


def run_enrichment(shard_dirs, dry_run=False):
    """Main enrichment loop."""
    t0 = time.time()

    # Load manifest
    manifest_path = config.manifest_path()
    if not os.path.exists(manifest_path):
        logger.error("Manifest not found. Run the pipeline first (step 1).")
        return

    logger.info(f"Loading manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)
    df = df.set_index('global_index', drop=False)  # index for fast df.update()
    n_total = len(df)
    logger.info(f"  {n_total:,} rows, columns: {list(df.columns)}")

    # Initialize enrichment columns if missing
    for col in ENRICHMENT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Load progress
    completed = _load_progress()
    logger.info(f"Previously enriched shards: {len(completed)}")

    # Find available shards
    shard_map = _find_shards(shard_dirs)
    logger.info(f"Found {len(shard_map)} unique .tar shards across {len(shard_dirs)} directories")

    # Filter to shards that have corresponding PT files and aren't done yet
    pending = {}
    for tar_basename, tar_path in shard_map.items():
        pt_basename = tar_basename.replace('.tar', '.pt')
        if pt_basename in df['embedding_file'].values and tar_basename not in completed:
            pending[tar_basename] = tar_path

    logger.info(f"Pending shards to enrich: {len(pending)}")

    if not pending:
        logger.info("Nothing to do — all available shards are already enriched.")
        return

    if dry_run:
        logger.info("DRY RUN — would process:")
        for name in sorted(pending.keys())[:20]:
            logger.info(f"  {name}")
        if len(pending) > 20:
            logger.info(f"  ... and {len(pending) - 20} more")
        return

    # Build fast index: embedding_file -> sorted list of global indices
    # Using numpy sort instead of pandas groupby (orders of magnitude faster)
    logger.info("Building manifest index...")
    emb_files = df['embedding_file'].values
    global_idxs = df['global_index'].values
    sort_order = np.argsort(global_idxs)  # sort by global_index
    sorted_files = emb_files[sort_order]
    sorted_gidxs = global_idxs[sort_order]

    file_groups = {}
    current_file = None
    current_list = []
    for i in range(len(sorted_files)):
        f = sorted_files[i]
        if f != current_file:
            if current_file is not None:
                file_groups[current_file] = current_list
            current_file = f
            current_list = []
        current_list.append(int(sorted_gidxs[i]))
    if current_file is not None:
        file_groups[current_file] = current_list
    logger.info(f"  Index built: {len(file_groups)} embedding files")

    # Process each shard — collect all records, merge at end
    n_enriched = 0
    n_failed = 0
    batch_save_interval = 500  # save every N shards
    all_enrichment_rows = []  # list of (global_index, record_dict)

    for i, (tar_basename, tar_path) in enumerate(sorted(pending.items())):
        pt_basename = tar_basename.replace('.tar', '.pt')

        try:
            # Extract metadata from shard
            records = _extract_metadata_from_shard(tar_path)

            # Get the global indices for this PT file
            if pt_basename not in file_groups:
                logger.warning(f"  {pt_basename} not in manifest, skipping")
                n_failed += 1
                continue

            global_indices = file_groups[pt_basename]

            if len(records) != len(global_indices):
                # Coordinate-matching fallback: tar has more samples than PT
                # (some images failed to decode during extraction)
                n_tar = len(records)
                n_pt = len(global_indices)
                logger.info(
                    f"  {tar_basename}: {n_tar} meta vs {n_pt} PT rows — "
                    f"using coordinate matching"
                )

                # Build coord lookup from tar records (img1 half only)
                half_tar = n_tar // 2
                tar_coords = np.zeros((half_tar, 2), dtype=np.float64)
                for j in range(half_tar):
                    r = records[j]  # img1 records are first half
                    lat, lon = r.get('lat', 0), r.get('lon', 0)
                    if lat and lon:
                        tar_coords[j] = _normalize_coord(float(lat), float(lon))

                # Get manifest coords for this PT file's rows
                half_pt = n_pt // 2
                pt_coords = np.column_stack([
                    df.loc[global_indices[:half_pt], 'norm_easting'].values,
                    df.loc[global_indices[:half_pt], 'norm_northing'].values
                ]).astype(np.float64)

                # Match each PT row to nearest tar sample by coordinate
                matched_records = []
                n_coord_matched = 0
                for j in range(half_pt):
                    dists = np.abs(tar_coords - pt_coords[j]).sum(axis=1)
                    best = np.argmin(dists)
                    if dists[best] < 1e-4:  # ~5m tolerance
                        # img1 row
                        r1 = records[best].copy()
                        r1['global_index'] = global_indices[j]
                        matched_records.append(r1)
                        # img2 row (second half of records + indices)
                        r2 = records[half_tar + best].copy()
                        r2['global_index'] = global_indices[half_pt + j]
                        matched_records.append(r2)
                        n_coord_matched += 1

                if n_coord_matched > 0:
                    all_enrichment_rows.extend(matched_records)
                    completed.add(tar_basename)
                    n_enriched += 1
                    logger.info(
                        f"    coord-matched {n_coord_matched}/{half_pt} "
                        f"samples ({n_coord_matched * 2} rows)"
                    )
                else:
                    n_failed += 1
                    logger.warning(f"    coord matching failed entirely")
                continue

            # Collect records with their global indices
            for record, gidx in zip(records, global_indices):
                record['global_index'] = gidx
                all_enrichment_rows.append(record)

            completed.add(tar_basename)
            n_enriched += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  {i + 1}/{len(pending)} shards processed "
                            f"({n_enriched} enriched, {n_failed} failed, "
                            f"{len(all_enrichment_rows):,} rows collected)")

            # Periodic merge + save
            if (i + 1) % batch_save_interval == 0:
                logger.info(f"  Merging {len(all_enrichment_rows):,} rows into manifest...")
                enrich_df = pd.DataFrame(all_enrichment_rows)
                enrich_df = enrich_df.set_index('global_index')
                df.update(enrich_df)
                all_enrichment_rows = []
                logger.info(f"  Saving checkpoint ({n_enriched} shards enriched)...")
                df.to_parquet(manifest_path + ".tmp", index=False)
                os.rename(manifest_path + ".tmp", manifest_path)
                _save_progress(completed)

        except Exception as e:
            logger.warning(f"  {tar_basename}: ERROR — {e}")
            n_failed += 1
            continue

    # Final merge of remaining rows
    if all_enrichment_rows:
        logger.info(f"  Final merge: {len(all_enrichment_rows):,} rows...")
        enrich_df = pd.DataFrame(all_enrichment_rows)
        enrich_df = enrich_df.set_index('global_index')
        df.update(enrich_df)

    # Final save
    logger.info("Saving final enriched manifest...")
    df.to_parquet(manifest_path + ".tmp", index=False)
    os.rename(manifest_path + ".tmp", manifest_path)
    _save_progress(completed)

    # Summary
    elapsed = time.time() - t0
    n_enriched_total = df['sample_key'].notna().sum()
    n_remaining = df['sample_key'].isna().sum()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Enrichment complete in {elapsed / 60:.1f} min")
    logger.info(f"  This session: {n_enriched} shards enriched, {n_failed} failed")
    logger.info(f"  Total enriched rows: {n_enriched_total:,} / {n_total:,} "
                f"({n_enriched_total / n_total * 100:.1f}%)")
    logger.info(f"  Remaining: {n_remaining:,} rows ({n_remaining / n_total * 100:.1f}%)")
    logger.info(f"  Total shards processed: {len(completed)}")
    logger.info(f"{'=' * 60}")


def show_status():
    """Show current enrichment status."""
    manifest_path = config.manifest_path()
    if not os.path.exists(manifest_path):
        print("Manifest not found.")
        return

    df = pd.read_parquet(manifest_path)
    n_total = len(df)

    print(f"\n{'=' * 60}")
    print(f"  Manifest Enrichment Status")
    print(f"{'=' * 60}")
    print(f"  Total rows: {n_total:,}")

    if 'sample_key' in df.columns:
        n_enriched = df['sample_key'].notna().sum()
        n_remaining = n_total - n_enriched
        print(f"  Enriched:   {n_enriched:,} ({n_enriched / n_total * 100:.1f}%)")
        print(f"  Remaining:  {n_remaining:,} ({n_remaining / n_total * 100:.1f}%)")

        if n_enriched > 0:
            print(f"\n  Enrichment columns: {ENRICHMENT_COLUMNS}")
            for col in ENRICHMENT_COLUMNS:
                if col in df.columns:
                    filled = df[col].notna().sum()
                    unique = df[col].nunique()
                    print(f"    {col:20s}: {filled:>10,} filled, {unique:>8,} unique")

            # Source distribution
            if 'source' in df.columns and df['source'].notna().any():
                print(f"\n  Source distribution (enriched rows):")
                src_counts = df['source'].value_counts()
                for src, count in src_counts.items():
                    print(f"    {src:20s}: {count:>10,}")

            # Date range
            if 'capture_date' in df.columns and df['capture_date'].notna().any():
                dates = pd.to_datetime(df['capture_date'].dropna(), errors='coerce')
                print(f"\n  Date range: {dates.min()} → {dates.max()}")
    else:
        print("  Not yet enriched (no enrichment columns found)")

    # Progress file
    completed = _load_progress()
    print(f"\n  Processed shards: {len(completed)}")

    # Check which PT files still need enrichment
    if 'sample_key' in df.columns:
        unenriched_files = df[df['sample_key'].isna()]['embedding_file'].unique()
        # Map to tar names
        tar_names = set(f.replace('.pt', '.tar') for f in unenriched_files)
        print(f"  PT files needing enrichment: {len(unenriched_files)}")

        # Group by dataset prefix
        prefixes = {}
        for t in tar_names:
            prefix = '-'.join(t.split('-')[:-1])
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        if prefixes:
            print(f"\n  Missing shards by dataset:")
            for prefix, count in sorted(prefixes.items()):
                print(f"    {prefix}: {count} shards")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incrementally enrich manifest with WebDataset metadata",
        epilog="Connect an external drive, then run with --shards pointing to its shard directory."
    )
    parser.add_argument("--shards", action="append", default=None,
                        help="Path to a directory containing .tar shards. "
                             "Can be specified multiple times for multiple drives. "
                             "Default: baukultur_vpr/data/shards/")
    parser.add_argument("--status", action="store_true",
                        help="Show current enrichment progress and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without making changes")
    args = parser.parse_args()

    if args.status:
        show_status()
        sys.exit(0)

    # Default shard directory
    if args.shards is None:
        args.shards = [
            "/home/jubooz/landscape_signatures/baukultur_vpr/data/shards",
        ]

    run_enrichment(args.shards, dry_run=args.dry_run)
