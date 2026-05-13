"""
Export High-Res WebApp States
=============================
Generates 6 high-resolution images matching the webapp's new aesthetic:
- 3x UMAP scatter plots on black background (Micro, Meso, Macro)
- 3x Geographic interpolated grids with white Swiss border on black background
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image

DATA_PARQUET = "/home/jubooz/landscape_signatures/map_7b_frontend/data_7b.parquet"
BORDER_GEOJSON = "/home/jubooz/landscape_signatures/map_7b_frontend/ch_border_4326.geojson"
GRID_BOUNDS = [5.8, 45.7, 10.6, 47.9] # [MIN_LON, MIN_LAT, MAX_LON, MAX_LAT] from generate_grids.py
OUTPUT_DIR = "/home/jubooz/landscape_signatures/clustering_7b_results/highres_exports"

def get_cluster_colors_hex(clusters):
    """Convert cluster IDs to hex colors for matplotlib matching the webapp"""
    colors = []
    for c in clusters:
        if c == -1 or np.isnan(c):
            colors.append('#282828') # Noise: dark grey
            continue
            
        hue = (int(c) * 137.508) % 360
        s, l = 0.78, 0.58
        c_val = hue / 360.0
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        
        def h2r(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
            
        r = int(h2r(p, q, c_val + 1/3) * 255)
        g = int(h2r(p, q, c_val) * 255)
        b = int(h2r(p, q, c_val - 1/3) * 255)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors

def export_umap(df, scale, col_name):
    print(f"Exporting UMAP {scale}...")
    fig, ax = plt.subplots(figsize=(24, 24), facecolor='#000000')
    
    # Sort so noise is plotted first (at the bottom)
    df_sorted = df.sort_values(by=col_name)
    colors = get_cluster_colors_hex(df_sorted[col_name].values)
    
    # Sizes: noise is smaller
    sizes = np.where(df_sorted[col_name] == -1, 0.5, 2.0)
    alphas = np.where(df_sorted[col_name] == -1, 0.2, 0.8)
    
    ax.scatter(df_sorted['umap_x'], df_sorted['umap_y'], c=colors, s=sizes, alpha=alphas, edgecolors='none')
    
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout(pad=0)
    
    out_path = os.path.join(OUTPUT_DIR, f"export_umap_{scale}.png")
    fig.savefig(out_path, dpi=300, facecolor='#000000', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  -> {out_path}")

def export_geo(df, scale, col_name):
    print(f"Exporting Geo {scale} as points...")
    border = gpd.read_file(BORDER_GEOJSON)
    
    # Calculate aspect ratio based on physical distance at this latitude
    # 1 deg lon at 46.8 lat is ~ 76km. 1 deg lat is ~ 111km.
    # Aspect ratio = 111 / 76 = 1.46
    
    fig, ax = plt.subplots(figsize=(30, 20), facecolor='#000000')
    
    # Sort so noise is plotted first
    df_sorted = df.sort_values(by=col_name)
    colors = get_cluster_colors_hex(df_sorted[col_name].values)
    
    # Noise points are slightly smaller and more transparent
    sizes = np.where(df_sorted[col_name] == -1, 0.2, 0.5)
    alphas = np.where(df_sorted[col_name] == -1, 0.2, 0.8)
    
    ax.scatter(df_sorted['lon'], df_sorted['lat'], c=colors, s=sizes, alpha=alphas, edgecolors='none')
    
    # Overlay border
    border.boundary.plot(ax=ax, color='white', linewidth=1.5)
    
    ax.axis('off')
    ax.set_aspect(1.46)
    
    # Set bounds to Switzerland
    ax.set_xlim(GRID_BOUNDS[0], GRID_BOUNDS[2])
    ax.set_ylim(GRID_BOUNDS[1], GRID_BOUNDS[3])
    
    plt.tight_layout(pad=0)
    out_path = os.path.join(OUTPUT_DIR, f"export_geo_{scale}.png")
    fig.savefig(out_path, dpi=400, facecolor='#000000', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"  -> {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading parquet data...")
    df = pd.read_parquet(DATA_PARQUET)
    
    # 1. Export UMAPs
    export_umap(df, "micro", "cluster_micro")
    export_umap(df, "meso", "cluster_meso")
    export_umap(df, "macro", "cluster_macro")
    
    # 2. Export Geo Grids
    export_geo(df, "micro", "cluster_micro")
    export_geo(df, "meso", "cluster_meso")
    export_geo(df, "macro", "cluster_macro")
    
    print("All exports complete!")

if __name__ == "__main__":
    main()
