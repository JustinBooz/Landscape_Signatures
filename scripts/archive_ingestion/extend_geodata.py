"""
Incremental geodata enrichment — only processes NEW manifest rows
and appends them to existing domain checkpoints.
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'graph_pipeline'))
import config

# Reuse all the domain functions from geodata_enrich_v2
from geodata_enrich_v2 import (
    enrich_roads, enrich_rail, enrich_water, enrich_landcover,
    enrich_buildings_tlm, enrich_gwr, enrich_settlements, enrich_terrain,
    _checkpoint_path, _save_domain, CHECKPOINT_DIR, OUTPUT_PATH,
    logger
)

INCREMENTAL_CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "geodata_checkpoints_v2_incremental")
os.makedirs(INCREMENTAL_CHECKPOINT_DIR, exist_ok=True)


def _inc_checkpoint_path(domain):
    return os.path.join(INCREMENTAL_CHECKPOINT_DIR, f"geodata_{domain}.parquet")


def _inc_done(domain):
    return os.path.exists(_inc_checkpoint_path(domain))


def _inc_save(domain, df):
    tmp = _inc_checkpoint_path(domain) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.rename(tmp, _inc_checkpoint_path(domain))
    logger.info(f"  Saved incremental checkpoint: {domain} ({len(df):,} rows)")


def run_domain_incremental(domain_func, domain_name, gidx_new, coords_new):
    """Run a single domain on new points only, then append to existing checkpoint."""
    if _inc_done(domain_name):
        logger.info(f"[{domain_name}] Incremental already complete, skipping")
        return

    logger.info(f"[{domain_name}] Running on {len(gidx_new):,} new points...")

    # Temporarily override checkpoint functions so domain_func writes to incremental dir
    import geodata_enrich_v2 as gev2
    orig_cp_path = gev2._checkpoint_path
    orig_domain_done = gev2._domain_done
    gev2._checkpoint_path = _inc_checkpoint_path
    gev2._domain_done = lambda d, expected_n=None: _inc_done(d)
    gev2._save_domain = _inc_save

    try:
        domain_func(gidx_new, coords_new)
    finally:
        gev2._checkpoint_path = orig_cp_path
        gev2._domain_done = orig_domain_done


def merge_incremental():
    """Append incremental results to existing checkpoints, then rebuild final merge."""
    domains = ["roads", "rail", "water", "landcover", "buildings_tlm",
               "gwr", "settlements", "terrain"]

    for domain in domains:
        old_cp = _checkpoint_path(domain)
        inc_cp = _inc_checkpoint_path(domain)

        if not os.path.exists(inc_cp):
            logger.warning(f"  [{domain}] No incremental result, skipping")
            continue

        old_df = pd.read_parquet(old_cp)
        new_df = pd.read_parquet(inc_cp)
        merged = pd.concat([old_df, new_df], ignore_index=True)

        # Atomic save back to original checkpoint
        tmp = old_cp + ".tmp"
        merged.to_parquet(tmp, index=False)
        os.rename(tmp, old_cp)
        logger.info(f"  [{domain}] Extended: {len(old_df):,} → {len(merged):,} (+{len(new_df):,})")

    # Now do the final merge across all domains
    logger.info("Building final geodata_enriched.parquet...")
    manifest = pd.read_parquet(config.manifest_path(), columns=['global_index'])
    gidx = manifest['global_index'].values

    result = pd.DataFrame({'global_index': gidx})
    for domain in domains:
        cp = _checkpoint_path(domain)
        if os.path.exists(cp):
            df = pd.read_parquet(cp)
            cols = [c for c in df.columns if c != 'global_index']
            for c in cols:
                result[c] = df[c].values
            logger.info(f"  Merged {domain}: {len(cols)} columns")

    tmp = OUTPUT_PATH + ".tmp"
    result.to_parquet(tmp, index=False)
    os.rename(tmp, OUTPUT_PATH)
    logger.info(f"Saved: {OUTPUT_PATH} ({len(result):,} rows, {len(result.columns)} cols)")


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("INCREMENTAL GEODATA ENRICHMENT — NEW POINTS ONLY")
    logger.info("=" * 60)

    # Load full manifest
    manifest = pd.read_parquet(config.manifest_path(),
                                columns=['global_index', 'lv95_easting', 'lv95_northing'])

    # Figure out which rows are new (not in existing checkpoints)
    old_n = pd.read_parquet(_checkpoint_path("roads"), columns=['global_index']).shape[0]
    new_manifest = manifest.iloc[old_n:]

    logger.info(f"Total manifest: {len(manifest):,} | Old: {old_n:,} | New: {len(new_manifest):,}")

    if len(new_manifest) == 0:
        logger.info("Nothing new to enrich.")
        return

    gidx_new = new_manifest['global_index'].values
    coords_new = np.column_stack([new_manifest['lv95_easting'].values,
                                   new_manifest['lv95_northing'].values])

    # Run each domain on new points only
    run_domain_incremental(enrich_roads, "roads", gidx_new, coords_new)
    run_domain_incremental(enrich_rail, "rail", gidx_new, coords_new)
    run_domain_incremental(enrich_water, "water", gidx_new, coords_new)
    run_domain_incremental(enrich_landcover, "landcover", gidx_new, coords_new)
    run_domain_incremental(enrich_buildings_tlm, "buildings_tlm", gidx_new, coords_new)
    run_domain_incremental(enrich_gwr, "gwr", gidx_new, coords_new)
    run_domain_incremental(enrich_settlements, "settlements", gidx_new, coords_new)
    run_domain_incremental(enrich_terrain, "terrain", gidx_new, coords_new)

    # Merge incremental into existing checkpoints
    merge_incremental()

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nIncremental enrichment complete in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
