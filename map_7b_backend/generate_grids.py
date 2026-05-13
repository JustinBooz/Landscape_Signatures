"""
Generate Interpolated Image Grids for 7B WebApp
===============================================
Creates nearest-neighbor raster images of the 3.2M points over Switzerland.
Masks out everything outside the Swiss boundary.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib.path as mpath
from shapely.geometry import Polygon, MultiPolygon

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_PARQUET = "/home/jubooz/landscape_signatures/map_7b_frontend/data_7b.parquet"
BORDER_GEOJSON = "/home/jubooz/landscape_signatures/clustering_7b_results/switzerland_border_2056.geojson"
OUTPUT_DIR = "/home/jubooz/landscape_signatures/map_7b_frontend"

# We create the grid uniformly in WGS84 (lon/lat) so it aligns perfectly in Deck.gl's BitmapLayer
# Rough bounds for Switzerland
MIN_LON, MAX_LON = 5.8, 10.6
MIN_LAT, MAX_LAT = 45.7, 47.9

# Grid resolution (e.g. 3000 x 2000 pixels)
WIDTH = 3000
HEIGHT = 2000

def get_cluster_color(cluster_id):
    """Deterministic hashing to RGBA, matching app.js logic"""
    if cluster_id == -1 or np.isnan(cluster_id):
        return (40, 40, 40, 255) # Noise: solid dark grey
    
    hue = (int(cluster_id) * 137.508) % 360
    s, l = 0.78, 0.58
    
    c = hue / 360.0
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    
    def h2r(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    
    r = int(h2r(p, q, c + 1/3) * 255)
    g = int(h2r(p, q, c) * 255)
    b = int(h2r(p, q, c - 1/3) * 255)
    return (r, g, b, 255) # Fully opaque, transparency handles mask

def get_polygon_paths(geometry):
    """Extract matplotlib Paths from shapely geometry for fast point-in-polygon"""
    paths = []
    if isinstance(geometry, Polygon):
        paths.append(mpath.Path(np.asarray(geometry.exterior.coords)))
    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            paths.append(mpath.Path(np.asarray(poly.exterior.coords)))
    return paths

def main():
    t0 = time.time()
    
    # 1. Load Data
    logger.info("Loading parquet data...")
    df = pd.read_parquet(DATA_PARQUET)
    points = np.column_stack([df['lon'].values, df['lat'].values])
    logger.info(f"Loaded {len(df):,} points")
    
    # 2. Load and prep border mask
    logger.info("Loading Swiss border mask...")
    border_gdf = gpd.read_file(BORDER_GEOJSON)
    # Project to WGS84 to match grid
    border_gdf = border_gdf.to_crs("EPSG:4326")
    swiss_geom = border_gdf.geometry.iloc[0]
    
    # 3. Create KDTree
    logger.info("Building KDTree...")
    kdtree = cKDTree(points)
    
    # 4. Generate Pixel Grid (Lon/Lat)
    logger.info(f"Generating {WIDTH}x{HEIGHT} pixel grid...")
    x = np.linspace(MIN_LON, MAX_LON, WIDTH)
    # y must go top-to-bottom for image coordinates (row 0 is max lat)
    y = np.linspace(MAX_LAT, MIN_LAT, HEIGHT)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # 5. Query KDTree
    logger.info("Querying nearest neighbors for all pixels...")
    distances, indices = kdtree.query(grid_points, k=1, workers=-1)
    
    # 6. Apply Border Mask
    # Mask out points far outside the border using matplotlib Path
    logger.info("Applying polygon mask...")
    paths = get_polygon_paths(swiss_geom)
    mask = np.zeros(len(grid_points), dtype=bool)
    for path in paths:
        mask |= path.contains_points(grid_points)
    
    # Pre-compute color palettes for speed
    def render_resolution(scale, col_name):
        logger.info(f"Rendering {scale} resolution...")
        clusters = df[col_name].values
        pixel_clusters = clusters[indices]
        
        # Initialize image array as transparent
        img_data = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        img_data_flat = img_data.reshape(-1, 4)
        
        # Unique clusters to optimize color generation
        unique_clusters = np.unique(pixel_clusters[mask])
        color_map = {c: get_cluster_color(c) for c in unique_clusters}
        
        # Fill only masked pixels
        for i, m in enumerate(mask):
            if m:
                img_data_flat[i] = color_map[pixel_clusters[i]]
                
        img = Image.fromarray(img_data, 'RGBA')
        out_path = os.path.join(OUTPUT_DIR, f"grid_{scale}.png")
        img.save(out_path, format="PNG")
        logger.info(f"Saved {out_path}")

    # Generate the 3 maps
    render_resolution("micro", "cluster_micro")
    render_resolution("meso", "cluster_meso")
    render_resolution("macro", "cluster_macro")
    
    # Also save the bounds for the frontend config
    bounds = {
        "bounds": [MIN_LON, MIN_LAT, MAX_LON, MAX_LAT]
    }
    import json
    with open(os.path.join(OUTPUT_DIR, "grid_bounds.json"), "w") as f:
        json.dump(bounds, f)
        
    logger.info(f"All grids generated in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
