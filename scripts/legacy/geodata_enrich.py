"""
Geodata Enrichment — Domain-by-domain spatial enrichment of the manifest.
Uses Swiss national datasets (TLM3D, GWR, swissALTI3D DEM).
Checkpoints after each domain. Runs independently from the graph pipeline.
"""

import os, sys, time, json, re, tempfile, urllib.request
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("geodata_enrich")

GEODATA_DIR = "/home/jubooz/landscape_signatures/geodata"
TLM_GPKG = os.path.join(GEODATA_DIR, "SWISSTLM3D_2026_LV95_LN02.gpkg")
GWR_CSV = os.path.join(GEODATA_DIR, "ch/gebaeude_batiment_edificio.csv")
DEM_INDEX = os.path.join(GEODATA_DIR, "ch.swisstopo.swissalti3d-IpCuIEGd.csv")
DEM_CACHE = os.path.join(GEODATA_DIR, "dem_cache")
CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "geodata_checkpoints")
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "geodata_enriched.parquet")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEM_CACHE, exist_ok=True)

BATCH_SIZE = 500_000  # points per cKDTree query batch


def _checkpoint_path(domain):
    return os.path.join(CHECKPOINT_DIR, f"geodata_{domain}.parquet")


def _domain_done(domain):
    return os.path.exists(_checkpoint_path(domain))


def _save_domain(domain, result_df):
    """Atomically save domain result."""
    tmp = _checkpoint_path(domain) + ".tmp"
    result_df.to_parquet(tmp, index=False)
    os.rename(tmp, _checkpoint_path(domain))
    logger.info(f"  Saved checkpoint: {domain} ({len(result_df):,} rows)")


def _load_manifest_coords():
    """Load manifest with LV95 coordinates."""
    logger.info("Loading manifest...")
    df = pd.read_parquet(config.manifest_path(),
                         columns=['global_index', 'lv95_easting', 'lv95_northing'])
    coords = np.column_stack([df['lv95_easting'].values, df['lv95_northing'].values])
    logger.info(f"  {len(df):,} points loaded")
    return df['global_index'].values, coords


def _load_gpkg_layer(layer_name, columns=None):
    """Load a layer from the TLM3D GeoPackage."""
    import geopandas as gpd
    logger.info(f"  Loading TLM layer: {layer_name}...")
    t0 = time.time()
    gdf = gpd.read_file(TLM_GPKG, layer=layer_name, columns=columns)
    logger.info(f"    {len(gdf):,} features ({time.time()-t0:.1f}s)")
    return gdf


def _nearest_from_coords(query_pts, ref_pts, batch_size=BATCH_SIZE):
    """Find nearest reference point for each query point. Returns (distances, indices)."""
    tree = cKDTree(ref_pts)
    n = len(query_pts)
    dists = np.empty(n, dtype=np.float64)
    idxs = np.empty(n, dtype=np.int64)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        d, idx = tree.query(query_pts[i:j], k=1)
        dists[i:j] = d
        idxs[i:j] = idx
    return dists, idxs


def _count_within_radius(query_pts, ref_pts, radius, batch_size=BATCH_SIZE):
    """Count reference points within radius of each query point."""
    tree = cKDTree(ref_pts)
    n = len(query_pts)
    counts = np.empty(n, dtype=np.int32)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        results = tree.query_ball_point(query_pts[i:j], r=radius)
        counts[i:j] = [len(r) for r in results]
    return counts


# =========================================================================
# Domain: Roads
# =========================================================================
def enrich_roads(gidx, coords):
    if _domain_done("roads"):
        logger.info("[roads] Already complete, skipping")
        return
    logger.info("[roads] Starting...")

    gdf = _load_gpkg_layer("tlm_strassen_strasse")
    # Extract road class
    road_classes = gdf['objektart'].values if 'objektart' in gdf.columns else None

    # Get centroids/representative points for nearest search
    centroids = np.column_stack([gdf.geometry.centroid.x.values,
                                  gdf.geometry.centroid.y.values])

    logger.info("  Computing nearest road...")
    dists, idxs = _nearest_from_coords(coords, centroids)

    result = pd.DataFrame({'global_index': gidx})
    result['nearest_road_distance_m'] = dists
    if road_classes is not None:
        result['nearest_road_class'] = road_classes[idxs]

    # Road density within 100m and 250m (using centroids as proxy)
    logger.info("  Computing road density (100m)...")
    result['road_density_100m'] = _count_within_radius(coords, centroids, 100)
    logger.info("  Computing road density (250m)...")
    result['road_density_250m'] = _count_within_radius(coords, centroids, 250)

    _save_domain("roads", result)
    del gdf, centroids


# =========================================================================
# Domain: Rail
# =========================================================================
def enrich_rail(gidx, coords):
    if _domain_done("rail"):
        logger.info("[rail] Already complete, skipping")
        return
    logger.info("[rail] Starting...")

    rail = _load_gpkg_layer("tlm_oev_eisenbahn")
    rail_pts = np.column_stack([rail.geometry.centroid.x.values,
                                 rail.geometry.centroid.y.values])

    logger.info("  Computing nearest rail...")
    dists_rail, _ = _nearest_from_coords(coords, rail_pts)

    # Stations
    stations = _load_gpkg_layer("tlm_oev_haltestelle")
    station_pts = np.column_stack([stations.geometry.x.values,
                                    stations.geometry.y.values])
    station_names = stations['name'].values if 'name' in stations.columns else None

    logger.info("  Computing nearest station...")
    dists_sta, idxs_sta = _nearest_from_coords(coords, station_pts)

    result = pd.DataFrame({'global_index': gidx})
    result['nearest_rail_distance_m'] = dists_rail
    result['nearest_station_distance_m'] = dists_sta
    if station_names is not None:
        result['nearest_station_name'] = station_names[idxs_sta]

    _save_domain("rail", result)
    del rail, stations


# =========================================================================
# Domain: Water
# =========================================================================
def enrich_water(gidx, coords):
    if _domain_done("water"):
        logger.info("[water] Already complete, skipping")
        return
    logger.info("[water] Starting...")

    rivers = _load_gpkg_layer("tlm_gewaesser_fliessgewaesser")
    river_pts = np.column_stack([rivers.geometry.centroid.x.values,
                                  rivers.geometry.centroid.y.values])

    lakes = _load_gpkg_layer("tlm_gewaesser_stehendes_gewaesser")
    lake_pts = np.column_stack([lakes.geometry.centroid.x.values,
                                 lakes.geometry.centroid.y.values])

    logger.info("  Computing nearest river...")
    d_river, _ = _nearest_from_coords(coords, river_pts)
    logger.info("  Computing nearest lake...")
    d_lake, _ = _nearest_from_coords(coords, lake_pts)

    result = pd.DataFrame({'global_index': gidx})
    result['nearest_river_distance_m'] = d_river
    result['nearest_lake_distance_m'] = d_lake
    result['nearest_water_distance_m'] = np.minimum(d_river, d_lake)

    _save_domain("water", result)
    del rivers, lakes


# =========================================================================
# Domain: Land cover / forest
# =========================================================================
def enrich_landcover(gidx, coords):
    if _domain_done("landcover"):
        logger.info("[landcover] Already complete, skipping")
        return
    logger.info("[landcover] Starting...")

    gdf = _load_gpkg_layer("tlm_bb_bodenbedeckung")
    objektart = gdf['objektart'].values if 'objektart' in gdf.columns else None

    # Nearest land cover feature
    centroids = np.column_stack([gdf.geometry.centroid.x.values,
                                  gdf.geometry.centroid.y.values])

    logger.info("  Computing nearest land cover...")
    dists, idxs = _nearest_from_coords(coords, centroids)

    result = pd.DataFrame({'global_index': gidx})
    if objektart is not None:
        result['landcover_class'] = objektart[idxs]

    # Forest: filter for forest types and compute nearest
    forest_mask = gdf['objektart'].str.contains('Wald', case=False, na=False) if 'objektart' in gdf.columns else None
    if forest_mask is not None and forest_mask.any():
        forest_pts = centroids[forest_mask.values]
        logger.info(f"  Computing nearest forest ({forest_pts.shape[0]:,} features)...")
        d_forest, _ = _nearest_from_coords(coords, forest_pts)
        result['nearest_forest_distance_m'] = d_forest
        result['in_forest'] = d_forest < 25  # within 25m of forest centroid

    _save_domain("landcover", result)
    del gdf


# =========================================================================
# Domain: Buildings (TLM footprints)
# =========================================================================
def enrich_buildings_tlm(gidx, coords):
    if _domain_done("buildings_tlm"):
        logger.info("[buildings_tlm] Already complete, skipping")
        return
    logger.info("[buildings_tlm] Starting...")

    gdf = _load_gpkg_layer("tlm_bauten_gebaeude_footprint")
    bld_pts = np.column_stack([gdf.geometry.centroid.x.values,
                                gdf.geometry.centroid.y.values])

    logger.info("  Computing nearest building...")
    dists, _ = _nearest_from_coords(coords, bld_pts)

    logger.info("  Computing building density (100m)...")
    dens100 = _count_within_radius(coords, bld_pts, 100)
    logger.info("  Computing building density (250m)...")
    dens250 = _count_within_radius(coords, bld_pts, 250)

    result = pd.DataFrame({'global_index': gidx})
    result['nearest_building_distance_m'] = dists
    result['building_density_100m'] = dens100
    result['building_density_250m'] = dens250

    _save_domain("buildings_tlm", result)
    del gdf


# =========================================================================
# Domain: GWR Building Registry
# =========================================================================
def enrich_gwr(gidx, coords):
    if _domain_done("gwr"):
        logger.info("[gwr] Already complete, skipping")
        return
    logger.info("[gwr] Starting...")

    logger.info("  Loading GWR CSV...")
    gwr = pd.read_csv(GWR_CSV, sep='\t', usecols=[
        'EGID', 'GDEKT', 'GGDENAME', 'GKODE', 'GKODN', 'GBAUJ',
        'GASTW', 'GKAT', 'GKLAS', 'GANZWHG', 'GAREA'
    ], dtype={'GDEKT': str, 'GGDENAME': str})
    logger.info(f"    {len(gwr):,} buildings")

    # Filter valid coords
    valid = gwr['GKODE'].notna() & gwr['GKODN'].notna()
    gwr = gwr[valid].copy()
    gwr_pts = np.column_stack([gwr['GKODE'].values, gwr['GKODN'].values])

    logger.info("  Computing nearest GWR building...")
    dists, idxs = _nearest_from_coords(coords, gwr_pts)

    result = pd.DataFrame({'global_index': gidx})
    result['nearest_gwr_distance_m'] = dists
    result['nearest_building_year'] = gwr['GBAUJ'].values[idxs]
    result['nearest_building_floors'] = gwr['GASTW'].values[idxs]
    result['nearest_building_height_proxy_m'] = gwr['GASTW'].values[idxs] * 3.0
    result['nearest_building_category'] = gwr['GKAT'].values[idxs]
    result['nearest_building_class'] = gwr['GKLAS'].values[idxs]
    result['nearest_building_dwellings'] = gwr['GANZWHG'].values[idxs]
    result['nearest_building_area_m2'] = gwr['GAREA'].values[idxs]
    result['canton'] = gwr['GDEKT'].values[idxs]
    result['municipality_name'] = gwr['GGDENAME'].values[idxs]

    _save_domain("gwr", result)
    del gwr


# =========================================================================
# Domain: Settlement names
# =========================================================================
def enrich_settlements(gidx, coords):
    if _domain_done("settlements"):
        logger.info("[settlements] Already complete, skipping")
        return
    logger.info("[settlements] Starting...")

    gdf = _load_gpkg_layer("tlm_namen_siedlungsname")
    sett_pts = np.column_stack([gdf.geometry.centroid.x.values,
                                 gdf.geometry.centroid.y.values])
    names = gdf['name'].values if 'name' in gdf.columns else None

    logger.info("  Computing nearest settlement...")
    dists, idxs = _nearest_from_coords(coords, sett_pts)

    result = pd.DataFrame({'global_index': gidx})
    result['nearest_settlement_distance_m'] = dists
    if names is not None:
        result['nearest_settlement_name'] = names[idxs]

    _save_domain("settlements", result)
    del gdf


# =========================================================================
# Domain: Terrain (DEM)
# =========================================================================
def enrich_terrain(gidx, coords):
    if _domain_done("terrain"):
        logger.info("[terrain] Already complete, skipping")
        return
    logger.info("[terrain] Starting DEM terrain extraction...")

    import rasterio

    # Parse tile index
    tile_urls = {}
    with open(DEM_INDEX) as f:
        for line in f:
            url = line.strip()
            m = re.search(r'swissalti3d_\d{4}_(\d{4})-(\d{4})', url)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                tile_urls[key] = url

    logger.info(f"  {len(tile_urls)} tiles in index")

    # Group points by tile
    e_km = (coords[:, 0] // 1000).astype(int)
    n_km = (coords[:, 1] // 1000).astype(int)

    from collections import defaultdict
    tile_groups = defaultdict(list)
    for i in range(len(coords)):
        tile_groups[(e_km[i], n_km[i])].append(i)

    n_tiles = len(tile_groups)
    logger.info(f"  {n_tiles} tiles to process")

    # Results arrays
    elevation = np.full(len(coords), np.nan, dtype=np.float32)
    slope = np.full(len(coords), np.nan, dtype=np.float32)
    aspect = np.full(len(coords), np.nan, dtype=np.float32)

    # Progress: append-only text file (one "e_km,n_km" per line)
    done_file = os.path.join(CHECKPOINT_DIR, "terrain_done_tiles.txt")
    partial_path = os.path.join(CHECKPOINT_DIR, "terrain_partial.npz")

    done_tiles = set()
    if os.path.exists(done_file):
        with open(done_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    done_tiles.add((int(parts[0]), int(parts[1])))
        if os.path.exists(partial_path):
            data = np.load(partial_path)
            elevation = data['elevation']
            slope = data['slope']
            aspect = data['aspect']
        logger.info(f"  Resuming: {len(done_tiles)}/{n_tiles} tiles done")

    processed = 0
    failed = 0
    t_start = time.time()

    for tile_key, pt_indices in sorted(tile_groups.items()):
        if tile_key in done_tiles:
            processed += 1
            continue

        if tile_key not in tile_urls:
            processed += 1
            with open(done_file, 'a') as f:
                f.write(f"{tile_key[0]},{tile_key[1]}\n")
            continue

        url = tile_urls[tile_key]
        cache_path = os.path.join(DEM_CACHE, f"dem_{tile_key[0]}_{tile_key[1]}.tif")

        try:
            # Download with timeout
            if not os.path.exists(cache_path):
                urllib.request.urlretrieve(url, cache_path + ".tmp")
                os.rename(cache_path + ".tmp", cache_path)

            # Read tile and sample points
            with rasterio.open(cache_path) as src:
                dem = src.read(1)
                transform = src.transform

                for idx in pt_indices:
                    col, row = ~transform * (coords[idx, 0], coords[idx, 1])
                    r, c = int(row), int(col)
                    if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
                        elevation[idx] = dem[r, c]

                        # Slope/aspect from 3x3 Sobel
                        if 1 <= r < dem.shape[0]-1 and 1 <= c < dem.shape[1]-1:
                            dz_dx = (dem[r-1,c+1] + 2*dem[r,c+1] + dem[r+1,c+1] -
                                     dem[r-1,c-1] - 2*dem[r,c-1] - dem[r+1,c-1]) / (8 * src.res[0])
                            dz_dy = (dem[r+1,c-1] + 2*dem[r+1,c] + dem[r+1,c+1] -
                                     dem[r-1,c-1] - 2*dem[r-1,c] - dem[r-1,c+1]) / (8 * src.res[1])
                            slope[idx] = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
                            aspect[idx] = np.degrees(np.arctan2(-dz_dy, dz_dx))
                            if aspect[idx] < 0:
                                aspect[idx] += 360

            # Delete tile after use
            if os.path.exists(cache_path):
                os.remove(cache_path)

        except Exception as e:
            failed += 1
            if failed <= 10 or failed % 100 == 0:
                logger.warning(f"  Tile {tile_key}: {e}")

        # Mark done (append to file)
        with open(done_file, 'a') as f:
            f.write(f"{tile_key[0]},{tile_key[1]}\n")
        processed += 1

        # Checkpoint every 200 tiles
        if processed % 200 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta_min = (n_tiles - processed) / rate / 60 if rate > 0 else 0
            valid = np.isfinite(elevation).sum()
            logger.info(f"  Terrain: {processed}/{n_tiles} tiles, "
                        f"{valid:,} valid pts, {failed} failed, "
                        f"{rate:.1f} tiles/s, ETA {eta_min:.0f}min")
            np.savez(partial_path, elevation=elevation, slope=slope, aspect=aspect)

    result = pd.DataFrame({'global_index': gidx})
    result['elevation_m'] = elevation
    result['slope_deg'] = slope
    result['aspect_deg'] = aspect

    _save_domain("terrain", result)

    # Clean up progress files
    for f in [done_file, partial_path]:
        if os.path.exists(f):
            os.remove(f)

    logger.info(f"  Terrain complete: {np.isfinite(elevation).sum():,}/{len(elevation):,} "
                f"points with elevation, {failed} failed tiles")


# =========================================================================
# Final merge
# =========================================================================
def merge_all(gidx):
    """Merge all domain checkpoints into final enriched parquet."""
    logger.info("Merging all domain results...")

    domains = ["roads", "rail", "water", "landcover", "buildings_tlm",
               "gwr", "settlements", "terrain"]

    result = pd.DataFrame({'global_index': gidx})
    for domain in domains:
        cp = _checkpoint_path(domain)
        if os.path.exists(cp):
            df = pd.read_parquet(cp)
            # Drop global_index from domain df before merge (it's in result)
            cols = [c for c in df.columns if c != 'global_index']
            for c in cols:
                result[c] = df[c].values
            logger.info(f"  Merged {domain}: {len(cols)} columns")
        else:
            logger.warning(f"  {domain} checkpoint not found, skipping")

    # Save
    tmp = OUTPUT_PATH + ".tmp"
    result.to_parquet(tmp, index=False)
    os.rename(tmp, OUTPUT_PATH)
    logger.info(f"Saved: {OUTPUT_PATH} ({len(result):,} rows, {len(result.columns)} cols)")
    return result


# =========================================================================
# Main
# =========================================================================
def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("GEODATA ENRICHMENT PIPELINE")
    logger.info("=" * 60)

    gidx, coords = _load_manifest_coords()

    # Run each domain (checkpointed, skips if already done)
    enrich_roads(gidx, coords)
    enrich_rail(gidx, coords)
    enrich_water(gidx, coords)
    enrich_landcover(gidx, coords)
    enrich_buildings_tlm(gidx, coords)
    enrich_gwr(gidx, coords)
    enrich_settlements(gidx, coords)
    enrich_terrain(gidx, coords)

    # Merge
    merge_all(gidx)

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nGeodata enrichment complete in {elapsed:.1f} min")


if __name__ == "__main__":
    main()


