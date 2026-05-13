"""
Geodata Enrichment v2 — TRUE GEOMETRY DISTANCES
================================================
Same domain-by-domain enrichment as v1, but uses Shapely STRtree for
true nearest-geometry distances instead of centroid-based cKDTree.
This is slower (~1-3 hours vs ~3 minutes) but produces accurate distances.

Checkpoints after each domain. Runs independently from the graph pipeline.
"""

import os, sys, time, json, re, tempfile, urllib.request
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("geodata_enrich_v2",
                               log_file=os.path.join(config.BASE_DIR, "geodata_enrich_v2.log"))

GEODATA_DIR = "/home/jubooz/landscape_signatures/geodata"
TLM_GPKG = os.path.join(GEODATA_DIR, "SWISSTLM3D_2026_LV95_LN02.gpkg")
GWR_CSV = os.path.join(GEODATA_DIR, "ch/gebaeude_batiment_edificio.csv")
DEM_INDEX = os.path.join(GEODATA_DIR, "ch.swisstopo.swissalti3d-IpCuIEGd.csv")
DEM_CACHE = os.path.join(GEODATA_DIR, "dem_cache")
CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "geodata_checkpoints_v2")
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "geodata_enriched.parquet")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEM_CACHE, exist_ok=True)

BATCH_SIZE = 200_000  # points per STRtree batch (smaller than v1 due to geometry overhead)


def _checkpoint_path(domain):
    return os.path.join(CHECKPOINT_DIR, f"geodata_{domain}.parquet")


def _domain_done(domain, expected_n=None):
    cp = _checkpoint_path(domain)
    if not os.path.exists(cp):
        return False
    if expected_n is not None:
        # Check row count matches current manifest
        try:
            n = pd.read_parquet(cp, columns=['global_index']).shape[0]
            if n != expected_n:
                logger.warning(f"  [{domain}] Stale checkpoint: {n:,} rows vs {expected_n:,} expected. Re-running.")
                os.rename(cp, cp + ".old")
                return False
        except Exception:
            return False
    return True


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


# =========================================================================
# Core spatial functions — TRUE GEOMETRY distances
# =========================================================================
def _nearest_from_geom_fast(query_pts, geom_array, batch_size=BATCH_SIZE):
    """
    Find nearest geometry for each query point using Shapely STRtree.
    Returns (distances, indices) — true geometry distances, not centroid.
    Uses vectorized query_nearest with all_matches=False for 1:1 mapping.
    """
    from shapely import STRtree, points as shapely_points

    tree = STRtree(geom_array)
    n = len(query_pts)
    dists = np.empty(n, dtype=np.float64)
    idxs = np.empty(n, dtype=np.int64)

    n_batches = (n + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, n, batch_size)):
        j = min(i + batch_size, n)
        batch_pts = shapely_points(query_pts[i:j])

        # all_matches=False → returns (2D_indices, 1D_distances)
        # 2D_indices[0] = input indices, 2D_indices[1] = tree indices
        # Guaranteed 1:1 mapping with input
        result_idxs, result_dists = tree.query_nearest(
            batch_pts, all_matches=False, return_distance=True
        )
        qidx = result_idxs[0]  # input point indices within batch
        tidx = result_idxs[1]  # tree geometry indices

        # Vectorized assignment
        dists[i + qidx] = result_dists
        idxs[i + qidx] = tidx

        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            logger.info(f"    Nearest: batch {b+1}/{n_batches} "
                        f"({j:,}/{n:,} points)")

    return dists, idxs


def _count_intersecting_radius(query_pts, geom_array, radius, batch_size=BATCH_SIZE):
    """
    Count geometries that intersect a buffer of given radius around each query point.
    Uses STRtree.query with 'intersects' predicate for true geometry intersection.
    """
    from shapely import STRtree, points as shapely_points, buffer as shapely_buffer

    tree = STRtree(geom_array)
    n = len(query_pts)
    counts = np.empty(n, dtype=np.int32)

    n_batches = (n + batch_size - 1) // batch_size
    for b, i in enumerate(range(0, n, batch_size)):
        j = min(i + batch_size, n)
        batch_pts = shapely_points(query_pts[i:j])
        # Create buffer circles around each point
        batch_buffers = shapely_buffer(batch_pts, radius)

        # Query: returns (input_idx, tree_idx) pairs
        result = tree.query(batch_buffers, predicate='intersects')
        qidx = result[0]

        # Count per query point
        batch_counts = np.zeros(j - i, dtype=np.int32)
        if len(qidx) > 0:
            unique, cnts = np.unique(qidx, return_counts=True)
            batch_counts[unique] = cnts

        counts[i:j] = batch_counts

        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            logger.info(f"    Density (r={radius}m): batch {b+1}/{n_batches} "
                        f"({j:,}/{n:,} points)")

    return counts


def _nearest_from_coords(query_pts, ref_pts, batch_size=BATCH_SIZE):
    """
    Fallback: cKDTree for point-to-point distances (used for GWR, stations etc.
    where reference data is already point geometry).
    """
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


# =========================================================================
# Domain: Roads
# =========================================================================
def enrich_roads(gidx, coords):
    if _domain_done("roads", len(gidx)):
        logger.info("[roads] Already complete, skipping")
        return
    logger.info("[roads] Starting (true geometry distances)...")

    gdf = _load_gpkg_layer("tlm_strassen_strasse")
    geom_array = gdf.geometry.values  # actual line geometries
    road_classes = gdf['objektart'].values if 'objektart' in gdf.columns else None

    logger.info("  Computing nearest road (true geometry)...")
    dists, idxs = _nearest_from_geom_fast(coords, geom_array)

    result = pd.DataFrame({'global_index': gidx})
    result['nearest_road_distance_m'] = dists
    if road_classes is not None:
        result['nearest_road_class'] = road_classes[idxs]

    # Road density: count road geometries intersecting buffer circles
    logger.info("  Computing road density (100m, true intersection)...")
    result['road_density_100m'] = _count_intersecting_radius(coords, geom_array, 100)
    logger.info("  Computing road density (250m, true intersection)...")
    result['road_density_250m'] = _count_intersecting_radius(coords, geom_array, 250)

    _save_domain("roads", result)
    del gdf, geom_array


# =========================================================================
# Domain: Rail
# =========================================================================
def enrich_rail(gidx, coords):
    if _domain_done("rail", len(gidx)):
        logger.info("[rail] Already complete, skipping")
        return
    logger.info("[rail] Starting (true geometry distances)...")

    rail = _load_gpkg_layer("tlm_oev_eisenbahn")
    rail_geoms = rail.geometry.values  # line geometries

    logger.info("  Computing nearest rail (true geometry)...")
    dists_rail, _ = _nearest_from_geom_fast(coords, rail_geoms)

    # Stations are POINT geometries → cKDTree is exact
    stations = _load_gpkg_layer("tlm_oev_haltestelle")
    station_pts = np.column_stack([stations.geometry.x.values,
                                    stations.geometry.y.values])
    station_names = stations['name'].values if 'name' in stations.columns else None

    logger.info("  Computing nearest station (point geometry, exact)...")
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
    if _domain_done("water", len(gidx)):
        logger.info("[water] Already complete, skipping")
        return
    logger.info("[water] Starting (true geometry distances)...")

    rivers = _load_gpkg_layer("tlm_gewaesser_fliessgewaesser")
    river_geoms = rivers.geometry.values

    lakes = _load_gpkg_layer("tlm_gewaesser_stehendes_gewaesser")
    lake_geoms = lakes.geometry.values

    logger.info("  Computing nearest river (true geometry)...")
    d_river, _ = _nearest_from_geom_fast(coords, river_geoms)
    logger.info("  Computing nearest lake (true geometry)...")
    d_lake, _ = _nearest_from_geom_fast(coords, lake_geoms)

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
    if _domain_done("landcover", len(gidx)):
        logger.info("[landcover] Already complete, skipping")
        return
    logger.info("[landcover] Starting (true geometry distances)...")

    gdf = _load_gpkg_layer("tlm_bb_bodenbedeckung")
    geom_array = gdf.geometry.values
    objektart = gdf['objektart'].values if 'objektart' in gdf.columns else None

    logger.info("  Computing nearest land cover (true geometry)...")
    dists, idxs = _nearest_from_geom_fast(coords, geom_array)

    result = pd.DataFrame({'global_index': gidx})
    if objektart is not None:
        result['landcover_class'] = objektart[idxs]

    # Forest: filter for forest types and compute nearest
    forest_mask = gdf['objektart'].str.contains('Wald', case=False, na=False) if 'objektart' in gdf.columns else None
    if forest_mask is not None and forest_mask.any():
        forest_geoms = geom_array[forest_mask.values]
        logger.info(f"  Computing nearest forest ({forest_geoms.shape[0]:,} features, true geometry)...")
        d_forest, _ = _nearest_from_geom_fast(coords, forest_geoms)
        result['nearest_forest_distance_m'] = d_forest
        result['in_forest'] = d_forest < 25  # within 25m of forest geometry

    _save_domain("landcover", result)
    del gdf


# =========================================================================
# Domain: Buildings (TLM footprints)
# =========================================================================
def enrich_buildings_tlm(gidx, coords):
    if _domain_done("buildings_tlm", len(gidx)):
        logger.info("[buildings_tlm] Already complete, skipping")
        return
    logger.info("[buildings_tlm] Starting (true geometry distances)...")

    gdf = _load_gpkg_layer("tlm_bauten_gebaeude_footprint")
    geom_array = gdf.geometry.values  # polygon footprints

    logger.info("  Computing nearest building (true geometry)...")
    dists, _ = _nearest_from_geom_fast(coords, geom_array)

    logger.info("  Computing building density (100m, true intersection)...")
    dens100 = _count_intersecting_radius(coords, geom_array, 100)
    logger.info("  Computing building density (250m, true intersection)...")
    dens250 = _count_intersecting_radius(coords, geom_array, 250)

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
    """GWR uses point coordinates — cKDTree is already exact. No change needed."""
    if _domain_done("gwr", len(gidx)):
        logger.info("[gwr] Already complete, skipping")
        return
    logger.info("[gwr] Starting (point geometry, cKDTree exact)...")

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
    """Settlements are POINT geometries — cKDTree is already exact."""
    if _domain_done("settlements", len(gidx)):
        logger.info("[settlements] Already complete, skipping")
        return
    logger.info("[settlements] Starting (point geometry, cKDTree exact)...")

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
# Domain: Terrain (DEM) — unchanged from v1, uses raster sampling
# =========================================================================
def enrich_terrain(gidx, coords):
    if _domain_done("terrain", len(gidx)):
        logger.info("[terrain] Already complete, skipping")
        return

    # Check if v1 terrain checkpoint exists and reuse it
    v1_terrain = os.path.join(config.OUTPUT_DIR, "geodata_checkpoints", "geodata_terrain.parquet")
    if os.path.exists(v1_terrain):
        logger.info("[terrain] Reusing v1 terrain checkpoint (raster-based, identical logic)...")
        import shutil
        shutil.copy2(v1_terrain, _checkpoint_path("terrain"))
        logger.info(f"  Copied from {v1_terrain}")
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

    # Group points by tile (use Python int to avoid numpy int64 serialization issues)
    e_km = (coords[:, 0] // 1000).astype(int)
    n_km = (coords[:, 1] // 1000).astype(int)
    tile_keys = [(int(e), int(n)) for e, n in zip(e_km, n_km)]

    from collections import defaultdict
    tile_groups = defaultdict(list)
    for i, tk in enumerate(tile_keys):
        tile_groups[tk].append(i)

    n_tiles = len(tile_groups)
    logger.info(f"  {n_tiles} tiles to process")

    # Results arrays
    elevation = np.full(len(coords), np.nan, dtype=np.float32)
    slope = np.full(len(coords), np.nan, dtype=np.float32)
    aspect = np.full(len(coords), np.nan, dtype=np.float32)

    # Progress tracking
    progress_file = os.path.join(CHECKPOINT_DIR, "terrain_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        done_tiles = set(tuple(t) for t in progress.get('done_tiles', []))
        # Load partial results
        partial_path = os.path.join(CHECKPOINT_DIR, "terrain_partial.npz")
        if os.path.exists(partial_path):
            data = np.load(partial_path)
            elevation = data['elevation']
            slope = data['slope']
            aspect = data['aspect']
        logger.info(f"  Resuming: {len(done_tiles)}/{n_tiles} tiles done")
    else:
        done_tiles = set()

    processed = 0
    for tile_key, pt_indices in sorted(tile_groups.items()):
        if tile_key in done_tiles:
            continue

        if tile_key not in tile_urls:
            processed += 1
            done_tiles.add((int(tile_key[0]), int(tile_key[1])))
            continue

        url = tile_urls[tile_key]
        cache_path = os.path.join(DEM_CACHE, f"dem_{tile_key[0]}_{tile_key[1]}.tif")

        try:
            # Download if not cached
            if not os.path.exists(cache_path):
                urllib.request.urlretrieve(url, cache_path + ".tmp")
                os.rename(cache_path + ".tmp", cache_path)

            # Read tile and sample points (vectorised)
            with rasterio.open(cache_path) as src:
                dem = src.read(1)
                transform = src.transform
                res_x = abs(transform.a)
                res_y = abs(transform.e)
                H, W = dem.shape

                pt_idx_arr = np.asarray(pt_indices, dtype=np.int64)
                xs = coords[pt_idx_arr, 0]
                ys = coords[pt_idx_arr, 1]
                # rasterio's inverse transform: pixel = ~transform * (x, y)
                cols_f, rows_f = ~transform * (xs, ys)
                rr = np.floor(rows_f).astype(np.int64)
                cc = np.floor(cols_f).astype(np.int64)

                in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                vals = np.full(pt_idx_arr.shape, -9999.0, dtype=np.float32)
                vals[in_bounds] = dem[rr[in_bounds], cc[in_bounds]]
                valid = in_bounds & (vals > -9000)
                elevation[pt_idx_arr[valid]] = vals[valid]

                # Slope/aspect: only for points whose 3x3 neighborhood is in bounds
                interior = valid & (rr > 0) & (rr < H - 1) & (cc > 0) & (cc < W - 1)
                if interior.any():
                    rr_i = rr[interior]
                    cc_i = cc[interior]
                    # 8 neighbors via fancy indexing
                    n_tl = dem[rr_i - 1, cc_i - 1]; n_t = dem[rr_i - 1, cc_i]; n_tr = dem[rr_i - 1, cc_i + 1]
                    n_l  = dem[rr_i,     cc_i - 1]; n_c = dem[rr_i,     cc_i]; n_r  = dem[rr_i,     cc_i + 1]
                    n_bl = dem[rr_i + 1, cc_i - 1]; n_b = dem[rr_i + 1, cc_i]; n_br = dem[rr_i + 1, cc_i + 1]
                    patch_valid = ((n_tl > -9000) & (n_t > -9000) & (n_tr > -9000) &
                                   (n_l  > -9000) & (n_c > -9000) & (n_r  > -9000) &
                                   (n_bl > -9000) & (n_b > -9000) & (n_br > -9000))
                    dz_dx = (n_tr + 2 * n_r + n_br - n_tl - 2 * n_l - n_bl) / (8.0 * res_x)
                    dz_dy = (n_bl + 2 * n_b + n_br - n_tl - 2 * n_t - n_tr) / (8.0 * res_y)
                    slope_vals = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
                    aspect_vals = np.degrees(np.arctan2(-dz_dy, dz_dx))
                    aspect_vals = np.where(aspect_vals < 0, aspect_vals + 360, aspect_vals)
                    target_idx = pt_idx_arr[interior]
                    slope[target_idx[patch_valid]]  = slope_vals[patch_valid].astype(np.float32)
                    aspect[target_idx[patch_valid]] = aspect_vals[patch_valid].astype(np.float32)

            # Delete tile after use to save disk
            if os.path.exists(cache_path):
                os.remove(cache_path)

        except Exception as e:
            logger.warning(f"  Tile {tile_key}: {e}")

        done_tiles.add((int(tile_key[0]), int(tile_key[1])))
        processed += 1

        # Checkpoint every 500 tiles
        if processed % 500 == 0:
            logger.info(f"  Terrain: {processed}/{n_tiles} tiles "
                        f"({len(done_tiles)}/{n_tiles} total)")
            np.savez(os.path.join(CHECKPOINT_DIR, "terrain_partial.npz"),
                     elevation=elevation, slope=slope, aspect=aspect)
            with open(progress_file, 'w') as f:
                json.dump({'done_tiles': [[int(t[0]), int(t[1])] for t in done_tiles]}, f)

    result = pd.DataFrame({'global_index': gidx})
    result['elevation_m'] = elevation
    result['slope_deg'] = slope
    result['aspect_deg'] = aspect

    _save_domain("terrain", result)

    # Clean up progress files
    for f in [progress_file, os.path.join(CHECKPOINT_DIR, "terrain_partial.npz")]:
        if os.path.exists(f):
            os.remove(f)

    logger.info(f"  Terrain complete: {np.isfinite(elevation).sum():,}/{len(elevation):,} "
                f"points with elevation")


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
    logger.info("GEODATA ENRICHMENT v2 — TRUE GEOMETRY DISTANCES")
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
    logger.info(f"\nGeodata enrichment v2 complete in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
