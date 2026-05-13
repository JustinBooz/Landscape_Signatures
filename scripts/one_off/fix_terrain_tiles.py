"""
Fix failed DEM tiles — retry downloads with exponential backoff.
Patches the existing terrain checkpoint and re-merges the final parquet.
"""

import os, sys, time, json, re, urllib.request, urllib.error
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("fix_terrain",
                               log_file=os.path.join(config.BASE_DIR, "fix_terrain.log"))

GEODATA_DIR = "/home/jubooz/landscape_signatures/geodata"
DEM_INDEX = os.path.join(GEODATA_DIR, "ch.swisstopo.swissalti3d-IpCuIEGd.csv")
DEM_CACHE = os.path.join(GEODATA_DIR, "dem_cache")
CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "geodata_checkpoints_v2")
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "geodata_enriched.parquet")

MAX_RETRIES = 5
RETRY_BACKOFF = [2, 5, 10, 30, 60]  # seconds between retries


def download_with_retry(url, dest_path, max_retries=MAX_RETRIES):
    """Download a file with exponential backoff retries."""
    for attempt in range(max_retries):
        try:
            tmp = dest_path + ".tmp"
            urllib.request.urlretrieve(url, tmp)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                os.rename(tmp, dest_path)
                return True
            else:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise ValueError("Downloaded file is empty")
        except Exception as e:
            if os.path.exists(dest_path + ".tmp"):
                try: os.remove(dest_path + ".tmp")
                except: pass
            if attempt < max_retries - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
                logger.warning(f"    Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"    All {max_retries} attempts failed: {e}")
                return False
    return False


def main():
    import rasterio

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FIX TERRAIN — Retry failed DEM tiles")
    logger.info("=" * 60)

    # Load manifest coordinates
    manifest = pd.read_parquet(config.manifest_path(),
                                columns=['global_index', 'lv95_easting', 'lv95_northing'])
    coords = np.column_stack([manifest.lv95_easting.values, manifest.lv95_northing.values])
    gidx = manifest.global_index.values
    N = len(coords)

    # Load existing terrain checkpoint
    terrain_path = os.path.join(CHECKPOINT_DIR, "geodata_terrain.parquet")
    terrain_df = pd.read_parquet(terrain_path)
    elevation = terrain_df['elevation_m'].values.copy().astype(np.float32)
    slope = terrain_df['slope_deg'].values.copy().astype(np.float32)
    aspect = terrain_df['aspect_deg'].values.copy().astype(np.float32)

    n_valid_before = int(np.isfinite(elevation).sum())
    n_missing = N - n_valid_before
    logger.info(f"  Before: {n_valid_before:,}/{N:,} valid elevation ({n_missing:,} missing)")

    # Parse tile index
    tile_urls = {}
    with open(DEM_INDEX) as f:
        for line in f:
            url = line.strip()
            m = re.search(r'swissalti3d_\d{4}_(\d{4})-(\d{4})', url)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                tile_urls[key] = url

    # Group ALL points by tile
    e_km = (coords[:, 0] // 1000).astype(int)
    n_km = (coords[:, 1] // 1000).astype(int)
    tile_groups = defaultdict(list)
    for i in range(N):
        tile_groups[(int(e_km[i]), int(n_km[i]))].append(i)

    # Find tiles that have missing elevation data
    missing_tiles = set()
    for tile_key, pt_indices in tile_groups.items():
        if tile_key not in tile_urls:
            continue  # No DEM available for this tile
        # Check if any points in this tile still need elevation
        for idx in pt_indices:
            if not np.isfinite(elevation[idx]):
                missing_tiles.add(tile_key)
                break

    logger.info(f"  {len(missing_tiles)} tiles with missing points to retry")
    logger.info(f"  {len(tile_urls)} total tiles in index")

    # Count tiles that have NO url (outside coverage)
    no_url_tiles = set()
    for tile_key in tile_groups:
        if tile_key not in tile_urls:
            no_url_tiles.add(tile_key)
    n_no_url_pts = sum(len(tile_groups[tk]) for tk in no_url_tiles)
    logger.info(f"  {len(no_url_tiles)} tiles outside DEM coverage ({n_no_url_pts:,} points)")

    os.makedirs(DEM_CACHE, exist_ok=True)
    fixed = 0
    still_failed = 0
    total_new_pts = 0

    for i, tile_key in enumerate(sorted(missing_tiles)):
        url = tile_urls[tile_key]
        cache_path = os.path.join(DEM_CACHE, f"dem_{tile_key[0]}_{tile_key[1]}.tif")

        # Download with retry
        if not os.path.exists(cache_path):
            ok = download_with_retry(url, cache_path)
            if not ok:
                still_failed += 1
                continue

        try:
            with rasterio.open(cache_path) as src:
                dem = src.read(1)
                transform = src.transform
                new_in_tile = 0

                for idx in tile_groups[tile_key]:
                    if np.isfinite(elevation[idx]):
                        continue  # Already has data

                    col, row = ~transform * (coords[idx, 0], coords[idx, 1])
                    r, c = int(row), int(col)
                    if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
                        val = dem[r, c]
                        if val > -9000:
                            elevation[idx] = val
                            new_in_tile += 1

                            if 1 <= r < dem.shape[0]-1 and 1 <= c < dem.shape[1]-1:
                                patch = dem[r-1:r+2, c-1:c+2]
                                if np.all(patch > -9000):
                                    dz_dx = (dem[r-1,c+1] + 2*dem[r,c+1] + dem[r+1,c+1] -
                                             dem[r-1,c-1] - 2*dem[r,c-1] - dem[r+1,c-1]) / (8 * src.res[0])
                                    dz_dy = (dem[r+1,c-1] + 2*dem[r+1,c] + dem[r+1,c+1] -
                                             dem[r-1,c-1] - 2*dem[r-1,c] - dem[r-1,c+1]) / (8 * src.res[1])
                                    slope[idx] = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
                                    aspect[idx] = np.degrees(np.arctan2(-dz_dy, dz_dx))
                                    if aspect[idx] < 0:
                                        aspect[idx] += 360

                total_new_pts += new_in_tile
                fixed += 1

        except Exception as e:
            logger.warning(f"  Tile {tile_key}: read failed: {e}")
            still_failed += 1

        # Clean up
        if os.path.exists(cache_path):
            os.remove(cache_path)

        if (i + 1) % 50 == 0 or (i + 1) == len(missing_tiles):
            logger.info(f"  Progress: {i+1}/{len(missing_tiles)} tiles, "
                        f"{fixed} fixed, {still_failed} failed, {total_new_pts:,} new points")

    # Save patched terrain checkpoint
    n_valid_after = int(np.isfinite(elevation).sum())
    logger.info(f"\n  After: {n_valid_after:,}/{N:,} valid elevation "
                f"({total_new_pts:,} new, {still_failed} permanently failed tiles)")

    result = pd.DataFrame({'global_index': gidx})
    result['elevation_m'] = elevation
    result['slope_deg'] = slope
    result['aspect_deg'] = aspect

    tmp = terrain_path + ".tmp"
    result.to_parquet(tmp, index=False)
    os.rename(tmp, terrain_path)
    logger.info(f"  Saved patched terrain checkpoint")

    # Re-merge all domains into final parquet
    logger.info("Re-merging all domain results...")
    domains = ["roads", "rail", "water", "landcover", "buildings_tlm",
               "gwr", "settlements", "terrain"]

    merged = pd.DataFrame({'global_index': gidx})
    for domain in domains:
        cp = os.path.join(CHECKPOINT_DIR, f"geodata_{domain}.parquet")
        if os.path.exists(cp):
            df = pd.read_parquet(cp)
            cols = [c for c in df.columns if c != 'global_index']
            for c in cols:
                merged[c] = df[c].values
            logger.info(f"  Merged {domain}: {len(cols)} columns")

    tmp = OUTPUT_PATH + ".tmp"
    merged.to_parquet(tmp, index=False)
    os.rename(tmp, OUTPUT_PATH)
    logger.info(f"Saved: {OUTPUT_PATH} ({len(merged):,} rows, {len(merged.columns)} cols)")

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nTerrain fix complete in {elapsed:.1f} min")
    logger.info(f"Coverage: {n_valid_before:,} → {n_valid_after:,} "
                f"({100*n_valid_after/N:.1f}%)")


if __name__ == "__main__":
    main()
