"""
Fix missing terrain data using Copernicus DEM 30m as fallback.
Downloads 1-degree tiles from AWS, samples missing points, patches checkpoint.
"""

import os, sys, time, zipfile, io
import numpy as np
import pandas as pd
import urllib.request
import rasterio
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("fix_terrain_srtm",
                               log_file=os.path.join(config.BASE_DIR, "fix_terrain_srtm.log"))

CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "geodata_checkpoints_v2")
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "geodata_enriched.parquet")
DEM_CACHE = os.path.join(config.BASE_DIR, "geodata", "srtm_cache")
os.makedirs(DEM_CACHE, exist_ok=True)

# Copernicus DEM 30m on AWS (no auth needed)
COP_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def cop_tile_name(lat, lon):
    """Build Copernicus DEM 30m tile name from integer lat/lon."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def download_cop_tile(lat, lon):
    """Download Copernicus DEM 30m tile, return path to local GeoTIFF."""
    name = cop_tile_name(lat, lon)
    local_path = os.path.join(DEM_CACHE, f"{name}.tif")
    if os.path.exists(local_path):
        return local_path

    url = f"{COP_BASE}/{name}/{name}.tif"
    logger.info(f"  Downloading: {name}...")

    for attempt in range(3):
        try:
            tmp = local_path + ".tmp"
            urllib.request.urlretrieve(url, tmp)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                os.rename(tmp, local_path)
                logger.info(f"    OK ({os.path.getsize(local_path) / 1e6:.1f} MB)")
                return local_path
        except Exception as e:
            if os.path.exists(local_path + ".tmp"):
                try: os.remove(local_path + ".tmp")
                except: pass
            wait = [2, 10, 30][attempt]
            logger.warning(f"    Attempt {attempt+1}/3 failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"  Failed to download tile {name}")
    return None


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FIX TERRAIN — Copernicus DEM 30m fallback")
    logger.info("=" * 60)

    # Load coordinates and current elevation
    manifest = pd.read_parquet(config.manifest_path(),
                                columns=['global_index', 'lv95_easting', 'lv95_northing'])
    coords = np.column_stack([manifest.lv95_easting.values, manifest.lv95_northing.values])
    gidx = manifest.global_index.values
    N = len(coords)

    terrain_path = os.path.join(CHECKPOINT_DIR, "geodata_terrain.parquet")
    terrain_df = pd.read_parquet(terrain_path)
    elevation = terrain_df['elevation_m'].values.copy().astype(np.float32)
    slope = terrain_df['slope_deg'].values.copy().astype(np.float32)
    aspect = terrain_df['aspect_deg'].values.copy().astype(np.float32)

    missing_mask = ~np.isfinite(elevation)
    n_missing = int(missing_mask.sum())
    logger.info(f"  Before: {N - n_missing:,}/{N:,} valid, {n_missing:,} missing")

    if n_missing == 0:
        logger.info("  Nothing to fix!")
        return

    # Transform missing points to WGS84
    transformer = Transformer.from_crs('EPSG:2056', 'EPSG:4326', always_xy=True)
    miss_indices = np.where(missing_mask)[0]
    miss_lv95 = coords[miss_indices]
    miss_lons, miss_lats = transformer.transform(miss_lv95[:, 0], miss_lv95[:, 1])

    # Find needed SRTM tiles
    tile_set = set()
    for lon, lat in zip(miss_lons, miss_lats):
        tile_set.add((int(np.floor(lat)), int(np.floor(lon))))
    logger.info(f"  {len(tile_set)} Copernicus DEM tiles needed")

    # Download and process each tile
    total_fixed = 0
    for tile_lat, tile_lon in sorted(tile_set):
        # Find points in this tile
        in_tile = (miss_lats >= tile_lat) & (miss_lats < tile_lat + 1) & \
                  (miss_lons >= tile_lon) & (miss_lons < tile_lon + 1)
        n_in_tile = int(in_tile.sum())
        if n_in_tile == 0:
            continue

        logger.info(f"  Tile N{tile_lat}E{tile_lon}: {n_in_tile:,} missing points")

        local_path = download_cop_tile(tile_lat, tile_lon)
        if local_path is None:
            continue

        try:
            with rasterio.open(local_path) as src:
                dem = src.read(1)
                transform = src.transform
                nodata = src.nodata

                tile_indices = miss_indices[in_tile]
                tile_lons = miss_lons[in_tile]
                tile_lats = miss_lats[in_tile]
                H, W = dem.shape

                # Vectorised inverse transform + bounds check
                cols_f, rows_f = ~transform * (tile_lons, tile_lats)
                rr = np.floor(rows_f).astype(np.int64)
                cc = np.floor(cols_f).astype(np.int64)
                in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)

                vals = np.full(rr.shape, -9999.0, dtype=np.float64)
                vals[in_bounds] = dem[rr[in_bounds], cc[in_bounds]]
                if nodata is not None:
                    valid = in_bounds & (vals != nodata) & (vals > -500)
                else:
                    valid = in_bounds & (vals > -500)
                elevation[tile_indices[valid]] = vals[valid].astype(np.float32)
                fixed_in_tile = int(valid.sum())

                # Slope/aspect for valid interior points
                interior = valid & (rr > 0) & (rr < H - 1) & (cc > 0) & (cc < W - 1)
                if interior.any():
                    rr_i = rr[interior]
                    cc_i = cc[interior]
                    n_tl = dem[rr_i - 1, cc_i - 1]; n_t = dem[rr_i - 1, cc_i]; n_tr = dem[rr_i - 1, cc_i + 1]
                    n_l  = dem[rr_i,     cc_i - 1]; n_c = dem[rr_i,     cc_i]; n_r  = dem[rr_i,     cc_i + 1]
                    n_bl = dem[rr_i + 1, cc_i - 1]; n_b = dem[rr_i + 1, cc_i]; n_br = dem[rr_i + 1, cc_i + 1]
                    if nodata is not None:
                        patch_valid = ((n_tl != nodata) & (n_t != nodata) & (n_tr != nodata) &
                                       (n_l  != nodata) & (n_c != nodata) & (n_r  != nodata) &
                                       (n_bl != nodata) & (n_b != nodata) & (n_br != nodata))
                    else:
                        patch_valid = ((n_tl > -500) & (n_t > -500) & (n_tr > -500) &
                                       (n_l  > -500) & (n_c > -500) & (n_r  > -500) &
                                       (n_bl > -500) & (n_b > -500) & (n_br > -500))
                    res_x = abs(transform.a) * 111320 * np.cos(np.radians(tile_lats[interior]))
                    res_y = abs(transform.e) * 110540
                    dz_dx = (n_tr + 2 * n_r + n_br - n_tl - 2 * n_l - n_bl) / (8.0 * res_x)
                    dz_dy = (n_bl + 2 * n_b + n_br - n_tl - 2 * n_t - n_tr) / (8.0 * res_y)
                    slope_vals = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
                    aspect_vals = np.degrees(np.arctan2(-dz_dy, dz_dx))
                    aspect_vals = np.where(aspect_vals < 0, aspect_vals + 360, aspect_vals)
                    target_idx = tile_indices[interior]
                    slope[target_idx[patch_valid]]  = slope_vals[patch_valid].astype(np.float32)
                    aspect[target_idx[patch_valid]] = aspect_vals[patch_valid].astype(np.float32)

                total_fixed += fixed_in_tile
                logger.info(f"    Fixed {fixed_in_tile:,}/{n_in_tile:,} points")

        except Exception as e:
            logger.error(f"    Error processing tile: {e}")

    # Save patched terrain checkpoint
    n_valid_after = int(np.isfinite(elevation).sum())
    still_missing = N - n_valid_after
    logger.info(f"\n  After: {n_valid_after:,}/{N:,} valid elevation "
                f"({total_fixed:,} new from SRTM, {still_missing:,} still missing)")

    result = pd.DataFrame({'global_index': gidx})
    result['elevation_m'] = elevation
    result['slope_deg'] = slope
    result['aspect_deg'] = aspect

    tmp = terrain_path + ".tmp"
    result.to_parquet(tmp, index=False)
    os.rename(tmp, terrain_path)
    logger.info("  Saved patched terrain checkpoint")

    # Re-merge all domains
    logger.info("Re-merging all domain results...")
    domains = ["roads", "rail", "water", "landcover", "buildings_tlm",
               "gwr", "settlements", "terrain"]

    merged = pd.DataFrame({'global_index': gidx})
    for domain in domains:
        cp = os.path.join(CHECKPOINT_DIR, f"geodata_{domain}.parquet")
        if os.path.exists(cp):
            df_d = pd.read_parquet(cp)
            cols = [c for c in df_d.columns if c != 'global_index']
            for c in cols:
                merged[c] = df_d[c].values
            logger.info(f"  Merged {domain}: {len(cols)} columns")

    tmp = OUTPUT_PATH + ".tmp"
    merged.to_parquet(tmp, index=False)
    os.rename(tmp, OUTPUT_PATH)
    logger.info(f"Saved: {OUTPUT_PATH} ({len(merged):,} rows, {len(merged.columns)} cols)")

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nCopernicus DEM fix complete in {elapsed:.1f} min")
    logger.info(f"Coverage: {N - n_missing:,} → {n_valid_after:,} ({100*n_valid_after/N:.1f}%)")


if __name__ == "__main__":
    main()
