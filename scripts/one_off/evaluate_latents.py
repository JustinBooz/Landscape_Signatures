import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from umap import UMAP
import geopandas as gpd
import contextily as cx
from sklearn.cluster import MiniBatchKMeans

# Local imports
from baukultur_vpr.models.encoder import VisionEncoder
from baukultur_vpr.models.aggregator import AggregatorHead
from baukultur_vpr.data.dataloader import get_dataloader

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)

@torch.no_grad()
def extract_embeddings(config_path, ckpt_dir, adapter_name, num_samples=250000):
    """Loads a model, runs inference, and extracts L2-normalized signatures."""
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nLoading model from {config_path}...")
    encoder = VisionEncoder(config).to(device)
    encoder.eval()
    
    lora_ckpt = os.path.abspath(os.path.join(ckpt_dir, "lora_latest"))
    if os.path.exists(os.path.join(lora_ckpt, adapter_name)):
        lora_ckpt = os.path.join(lora_ckpt, adapter_name)
        
    if os.path.exists(lora_ckpt):
        encoder.backbone.load_adapter(lora_ckpt, adapter_name)
        encoder.backbone.set_adapter(adapter_name)
    
    embed_dim = config["model"]["encoder"]["embed_dim"]
    
    is_v1 = "v1" in config_path.lower()
    if is_v1:
        out_dim = config["byol"]["output_dim"]
        aggregator = AggregatorHead(embed_dim=embed_dim, hidden_dim=config["byol"]["hidden_dim"], out_dim=out_dim, online=True).to(device)
        agg_ckpt = os.path.join(ckpt_dir, "agg_online_latest.pt")
    else:
        out_dim = config["model"]["aggregator"]["output_dim"]
        aggregator = AggregatorHead(embed_dim=embed_dim, out_dim=out_dim).to(device)
        agg_ckpt = os.path.join(ckpt_dir, "agg_latest.pt")
        
    if os.path.exists(agg_ckpt):
        aggregator.load_state_dict(torch.load(agg_ckpt, map_location=device))
    aggregator.eval()

    import glob
    shards_dir = config["data"]["output_shards_dir"]
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "dataset-*.tar")))
    
    # Increased batch size for faster inference
    dataloader = get_dataloader(
        shards_pattern=shard_files,
        config=config,
        batch_size=64, 
        num_workers=8
    )
    
    embeddings = []
    coordinates = []
    
    print(f"Extracting {num_samples} embeddings...")
    collected = 0
    for batch in dataloader:
        if len(batch) == 4:
            imgs1, _, lats, lons = batch
        else:
            imgs1, _ = batch
            lats = torch.zeros(imgs1.size(0))
            lons = torch.zeros(imgs1.size(0))
            
        imgs1 = imgs1.to(device)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if hasattr(encoder, 'use_target'):
                try:
                    feats = encoder(imgs1, use_target=False)
                except TypeError:
                    feats = encoder(imgs1)
            else:
                feats = encoder(imgs1)
            sigs = aggregator(feats)
            
        embeddings.append(sigs.cpu().numpy())
        coords = torch.stack([lats, lons], dim=1)
        coordinates.append(coords.numpy())
        
        collected += imgs1.size(0)
        
        if collected % 10000 == 0 or collected >= num_samples:
            print(f"Processed {min(collected, num_samples)} / {num_samples} images...")
            
        if collected >= num_samples:
            break
            
    del encoder, aggregator, imgs1, feats, sigs
    torch.cuda.empty_cache()
            
    return np.vstack(embeddings)[:num_samples], np.vstack(coordinates)[:num_samples]

def get_rgb_projection(embeddings):
    """Reduces 1280D embeddings to 3D UMAP components and scales to RGB."""
    print("Fitting 3D UMAP for RGB mapping (This will take a few minutes for 250k points)...")
    reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    proj3d = reducer.fit_transform(embeddings)
    proj3d = (proj3d - proj3d.min(axis=0)) / (proj3d.max(axis=0) - proj3d.min(axis=0))
    return proj3d

def create_dual_interactive_map(lats, lons, rgb_v1, rgb_v2):
    """Creates a side-by-side interactive cluster map over an ESRI basemap."""
    print("Preparing spatial data and fetching ESRI imagery...")
    
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats), 
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs(epsg=3857)
    x = gdf.geometry.x
    y = gdf.geometry.y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    
    ax1.set_title("V1 (BYOL) Landscape Clusters", fontsize=14)
    ax2.set_title("V2 (InfoNCE) Landscape Clusters", fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])

    # Marker size reduced to 1 to accommodate extreme density
    scatter1 = ax1.scatter(x, y, c=rgb_v1, s=1, alpha=0.5, edgecolors='none', zorder=2)
    scatter2 = ax2.scatter(x, y, c=rgb_v2, s=1, alpha=0.5, edgecolors='none', zorder=2)

    print("Downloading ESRI basemap tiles (this may take a moment)...")
    for ax in [ax1, ax2]:
        cx.add_basemap(
            ax, 
            crs=gdf.crs.to_string(), 
            source=cx.providers.Esri.WorldImagery,
            alpha=0.7,
            zorder=1
        )

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgray')
    cluster_slider = Slider(
        ax=ax_slider,
        label='Cluster Resolution (K)',
        valmin=2,
        valmax=150,
        valinit=150,
        valstep=1,
        color='#1f77b4'
    )
    cluster_slider.valtext.set_text('Raw UMAP')

    def update(val):
        k = int(cluster_slider.val)
        
        # Increased batch_size for MiniBatchKMeans to handle 250k points smoothly
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=42, n_init=3)
        
        labels_v1 = kmeans.fit_predict(rgb_v1)
        colors_v1 = np.clip(kmeans.cluster_centers_[labels_v1], 0.0, 1.0)
        scatter1.set_facecolors(colors_v1)
        
        labels_v2 = kmeans.fit_predict(rgb_v2)
        colors_v2 = np.clip(kmeans.cluster_centers_[labels_v2], 0.0, 1.0)
        scatter2.set_facecolors(colors_v2)
        
        cluster_slider.valtext.set_text(str(k))
        fig.canvas.draw_idle()

    cluster_slider.on_changed(update)
    
    print("Launching interface...")
    plt.show()

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./baukultur_vpr")
    
    # Increased to 250,000 for national scale coverage
    num_eval_samples = 250000 
    
    v1_embeddings, coords = extract_embeddings("config_v1.yaml", "models/weights/checkpoints_v1", "online", num_samples=num_eval_samples)
    v2_embeddings, _      = extract_embeddings("config_v2.yaml", "models/weights/checkpoints_v2", "default", num_samples=num_eval_samples)
    
    rgb_v1 = get_rgb_projection(v1_embeddings)
    rgb_v2 = get_rgb_projection(v2_embeddings)
    
    create_dual_interactive_map(coords[:, 0], coords[:, 1], rgb_v1, rgb_v2)
