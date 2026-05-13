"""
DINOv3 ViT-7B16 Embedding Extraction Pipeline
===============================================
Extracts 4096D visual embeddings from WebDataset .tar shards with
EPSG:2056 spatial normalization. Production-grade, checkpoint-safe.

Hardware: 1x NVIDIA RTX 5090 (32GB VRAM)
Input:    .tar shards on external Thunderbolt drive
Output:   .pt files with bfloat16 embeddings + float32 coordinates
"""

import os
import sys
import json
import glob
import time
import logging
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import webdataset as wds
from PIL import Image
from torchvision.transforms import v2 as T
from pyproj import Transformer

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEIGHTS_PATH = "/home/jubooz/landscape_signatures/baukultur_vpr/models/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
SHARDS_DIR   = "/home/jubooz/landscape_signatures/baukultur_vpr/data/shards"
OUTPUT_DIR   = "/home/jubooz/landscape_signatures/training_data_7b"
BATCH_SIZE   = 48    # yields 96 images per forward pass; safe without torch.compile CUDA Graphs
NUM_WORKERS  = 16    # ingestion is done — max out CPU for image decode/transform throughput
TARGET_SIZE  = 518
HUB_REPO     = "facebookresearch/dinov3"
HUB_MODEL    = "dinov3_vit7b16"

# EPSG:2056 (CH1903+/LV95) fixed global bounds for stable normalization
MIN_EASTING  = 2400000.0
MAX_EASTING  = 2900000.0
MIN_NORTHING = 1000000.0
MAX_NORTHING = 1350000.0

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('extract_7b_embeddings.log')
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coordinate transformer (WGS84 → CH1903+/LV95)
# ---------------------------------------------------------------------------
_wgs84_to_lv95 = Transformer.from_crs("epsg:4326", "epsg:2056", always_xy=True)

def normalize_coords_lv95(lat: float, lon: float) -> tuple:
    """Convert WGS84 to EPSG:2056 and min-max normalize to [-1, 1]."""
    easting, northing = _wgs84_to_lv95.transform(lon, lat)  # pyproj: (x=lon, y=lat)
    norm_e = 2.0 * (easting  - MIN_EASTING)  / (MAX_EASTING  - MIN_EASTING)  - 1.0
    norm_n = 2.0 * (northing - MIN_NORTHING) / (MAX_NORTHING - MIN_NORTHING) - 1.0
    return norm_e, norm_n

# ---------------------------------------------------------------------------
# Variable-resolution transform pipeline
# ---------------------------------------------------------------------------
_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def adaptive_transform(img: Image.Image) -> torch.Tensor:
    """Handle variable resolutions:
    - 518x518: direct normalize
    - Other: resize shortest edge to 518, center crop, normalize
    """
    t = T.ToImage()(img)
    t = T.ToDtype(torch.float32, scale=True)(t)
    
    if t.shape[1] == TARGET_SIZE and t.shape[2] == TARGET_SIZE:
        return _normalize(t)
    else:
        t = T.Resize(TARGET_SIZE, interpolation=T.InterpolationMode.BICUBIC, antialias=True)(t)
        t = T.CenterCrop(TARGET_SIZE)(t)
        return _normalize(t)

# ---------------------------------------------------------------------------
# WebDataset decode pipeline
# ---------------------------------------------------------------------------
def decode_sample(sample):
    """Yields (img1_tensor, img2_tensor, norm_easting, norm_northing)."""
    img1_pil, img2_pil, meta = sample
    try:
        lat = float(meta.get("lat", 0.0))
        lon = float(meta.get("lon", 0.0))
    except Exception:
        lat, lon = 0.0, 0.0
    
    t1 = adaptive_transform(img1_pil)
    t2 = adaptive_transform(img2_pil)
    ne, nn_ = normalize_coords_lv95(lat, lon)
    
    return t1, t2, ne, nn_

def collate_fn(batch):
    """Stack img1 and img2 into a single mega-batch for maximum GPU throughput."""
    img1s, img2s, eastings, northings = zip(*batch)
    
    # Concatenate img1 and img2 into one tall batch → 2*B images
    all_imgs = torch.stack(img1s + img2s, dim=0)
    
    # Coordinates are shared per pair — duplicate for both images
    coords = torch.tensor(
        list(zip(eastings, northings)) * 2,  # duplicate for img1 and img2
        dtype=torch.float32
    )
    
    return all_imgs, coords

# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------
def log_vram(tag: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resrv = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[VRAM {tag}] Allocated: {alloc:.2f} GB | Reserved: {resrv:.2f} GB | Total: {total:.2f} GB")

# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
@torch.inference_mode()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---- Model setup ----
    logger.info(f"Loading {HUB_MODEL} backbone (pretrained=False)...")
    model = torch.hub.load(HUB_REPO, HUB_MODEL, pretrained=False)
    
    logger.info(f"Injecting weights from: {WEIGHTS_PATH}")
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    log_vram("After model load")
    
    # Skip torch.compile — it locks 8.5 GB in CUDA Graphs but we're I/O bound, not compute bound
    log_vram("Model ready")
    
    # ---- Enumerate shards ----
    shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "dataset-*.tar")))
    logger.info(f"Found {len(shard_files)} shards in {SHARDS_DIR}")
    
    # ---- Identify already-completed shards for resumption ----
    completed = set()
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*.pt")):
        completed.add(os.path.basename(f).replace(".pt", ".tar"))
    
    remaining = [s for s in shard_files if os.path.basename(s) not in completed]
    logger.info(f"Already completed: {len(completed)} | Remaining: {len(remaining)}")
    
    if not remaining:
        logger.info("All shards processed. Nothing to do.")
        return
    
    # ---- Process each shard individually ----
    total_start = time.time()
    
    for shard_idx, shard_path in enumerate(remaining):
        shard_name = os.path.basename(shard_path).replace(".tar", "")
        out_path = os.path.join(OUTPUT_DIR, f"{shard_name}.pt")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{shard_idx+1}/{len(remaining)}] Processing: {shard_name}")
        log_vram("Shard start")
        shard_start = time.time()
        
        dataset = (
            wds.WebDataset([shard_path], shardshuffle=False, empty_check=False, handler=wds.warn_and_continue)
            .decode("pil", handler=wds.warn_and_continue)
            .to_tuple("img1.jpg", "img2.jpg", "meta.json", handler=wds.warn_and_continue)
            .map(decode_sample)
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=4,
        )
        
        shard_embeddings = []
        shard_coords = []
        shard_samples = 0
        
        try:
            for batch_idx, (imgs, coords) in enumerate(dataloader):
                imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                
                # Forward pass — CLS token extraction
                emb = model(imgs)  # [2*B, 4096]
                emb = F.normalize(emb, p=2, dim=-1)
                
                shard_embeddings.append(emb.cpu())       # bfloat16
                shard_coords.append(coords)               # float32
                shard_samples += imgs.size(0)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Batch {batch_idx+1}: {shard_samples} images processed")
                    log_vram("Mid-shard")
        except Exception as e:
            logger.warning(f"  ⚠️ Error mid-shard {shard_name} (discarding partial data): {e}")
            shard_embeddings.clear()
            shard_coords.clear()
            torch.cuda.empty_cache()
        
        if not shard_embeddings:
            logger.warning(f"  No valid samples in {shard_name}, skipping.")
            continue
        
        # Concatenate and save
        all_emb = torch.cat(shard_embeddings, dim=0)      # [N, 4096] bfloat16
        all_coords = torch.cat(shard_coords, dim=0)        # [N, 2]    float32
        
        torch.save({"embeddings": all_emb, "coords": all_coords}, out_path + ".tmp")
        os.rename(out_path + ".tmp", out_path)  # atomic on Linux — no corrupt files on kill
        
        elapsed = time.time() - shard_start
        rate = shard_samples / elapsed if elapsed > 0 else 0
        logger.info(f"  ✓ Saved {shard_name}.pt — {shard_samples} embeddings, "
                     f"{all_emb.shape}, {elapsed:.1f}s ({rate:.0f} img/s)")
        log_vram("Shard end")
        
        # Explicitly free
        del shard_embeddings, shard_coords, all_emb, all_coords
        torch.cuda.empty_cache()
    
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION COMPLETE — {len(remaining)} shards in {total_elapsed/60:.1f} minutes")
    logger.info(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
