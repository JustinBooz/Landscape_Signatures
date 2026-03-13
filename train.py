"""
Contrastive Training Loop (InfoNCE) — Baukultur VPR v2

Network:
  Encoder (DoRA on DINOv3) -> Aggregator (MixVPR/GeM) -> L2 Normalized Signature

Loss:
  InfoNCE (NT-Xent) with Geographic Hard-Negative Mining (Masking out False Negatives)
"""

import os
import glob
import math
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed — telemetry disabled.")

from models.encoder import VisionEncoder
from models.aggregator import AggregatorHead
from data.dataloader import get_dataloader

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# InfoNCE / Contrastive Loss with Geographic Masking
# ---------------------------------------------------------------------------

def haversine_dist(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in meters"""
    R = 6371000.0  # Earth radius in meters
    phi1 = lat1 * math.pi / 180.0
    phi2 = lat2 * math.pi / 180.0
    delta_phi = (lat2 - lat1) * math.pi / 180.0
    delta_lambda = (lon2 - lon1) * math.pi / 180.0
    
    a = torch.sin(delta_phi/2.0)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2.0)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return R * c

def info_nce_loss(q1: torch.Tensor, q2: torch.Tensor, lats: torch.Tensor, lons: torch.Tensor, temp: float, safe_radius: float):
    """
    NT-Xent with False Negative Geo-masking.
    q1, q2: [B, D]
    """
    device = q1.device
    B = q1.size(0)
    
    # 1. Concat
    features = torch.cat([q1, q2], dim=0) # [2B, D]
    
    # 2. Similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temp # [2B, 2B]
    
    # 3. True Positive Indices
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)], dim=0).to(device)
    pos_sim = sim_matrix[torch.arange(2*B), labels] # [2B]
    
    # 4. Geometry Distances
    lat_all = torch.cat([lats, lats], dim=0)
    lon_all = torch.cat([lons, lons], dim=0)
    
    lat_m1, lat_m2 = torch.meshgrid(lat_all, lat_all, indexing='ij')
    lon_m1, lon_m2 = torch.meshgrid(lon_all, lon_all, indexing='ij')
    dist_matrix = haversine_dist(lat_m1, lon_m1, lat_m2, lon_m2) # [2B, 2B]
    
    # Mask out pairs that are too close geographically (treat as false negatives)
    safe_mask = dist_matrix >= safe_radius
    
    # The valid mask for the denominator: We include "safe" negatives AND the True Positives
    valid_mask = safe_mask.clone()
    valid_mask[torch.arange(2*B), labels] = True  # Always keep the true positive
    valid_mask.fill_diagonal_(False)              # Exclude self-similarity
    
    # 5. Denominator
    exp_sim = torch.exp(sim_matrix)
    exp_sim_valid = exp_sim * valid_mask.float()
    
    denom = exp_sim_valid.sum(dim=1) # [2B]
    
    # 6. Loss
    loss = -pos_sim + torch.log(denom + 1e-8)
    return loss.mean()

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

CKPT_DIR = "models/weights/checkpoints"

def _ckpt(name: str) -> str:
    return os.path.join(CKPT_DIR, name)

def save_checkpoints(encoder, aggregator, optimizer, epoch: int):
    os.makedirs(CKPT_DIR, exist_ok=True)
    encoder.backbone.save_pretrained(_ckpt("lora_latest"))
    torch.save(aggregator.state_dict(), _ckpt("agg_latest.pt"))
    torch.save(optimizer.state_dict(),  _ckpt("optimizer_latest.pt"))
    with open(_ckpt("epoch.txt"), "w") as f:
        f.write(str(epoch))
    print(f"Checkpoints saved (epoch {epoch}).")

def load_checkpoints(encoder, aggregator, optimizer, device):
    start_epoch = 0
    epoch_file = _ckpt("epoch.txt")
    if not os.path.exists(epoch_file):
        return start_epoch

    try:
        with open(epoch_file) as f:
            start_epoch = int(f.read().strip()) + 1
        print(f"Resuming from epoch {start_epoch} ...")
        
        encoder.backbone.load_adapter(_ckpt("lora_latest"), "default")
        aggregator.load_state_dict(torch.load(_ckpt("agg_latest.pt"), map_location=device))
        optimizer.load_state_dict(torch.load(_ckpt("optimizer_latest.pt"), map_location=device))
    except Exception as e:
        print(f"Warning: checkpoint loading failed ({e}). Starting from scratch.")
        start_epoch = 0

    return start_epoch

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_cfg     = config["training"]
    wandb_cfg     = config.get("wandb", {})
    contrast_cfg  = config["contrastive"]

    accum_steps  = int(train_cfg.get("gradient_accumulation_steps", 1))
    use_ckpt     = train_cfg.get("gradient_checkpointing", False)

    if WANDB_AVAILABLE:
        wandb.init(
            project=wandb_cfg.get("project", "baukultur_vpr_v2"),
            entity=wandb_cfg.get("entity") or None,
            mode="offline" if wandb_cfg.get("offline", False) else "online",
            config={
                "lr":             train_cfg["learning_rate"],
                "batch_size":     train_cfg["batch_size"],
                "effective_batch": train_cfg["batch_size"] * accum_steps,
                "temperature":     contrast_cfg["temperature"],
                "safe_radius":     contrast_cfg["safe_radius_meters"],
                "grad_accum":     accum_steps,
            },
        )

    print("Initialising Encoder (ViT-H/16+ with DoRA) ...")
    encoder = VisionEncoder(config).to(device)

    if use_ckpt:
        encoder.enable_gradient_checkpointing()

    embed_dim = config["model"]["encoder"]["embed_dim"]
    out_dim   = config["model"]["aggregator"]["output_dim"]

    print("Initialising Aggregator (GeM Pool + MLP) ...")
    aggregator = AggregatorHead(
        embed_dim=embed_dim,
        out_dim=out_dim,
    ).to(device)

    shards_dir = config["data"]["output_shards_dir"]
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "dataset-*.tar")))
    if not shard_files:
        raise RuntimeError(f"No .tar shards found in '{shards_dir}'.")
    print(f"Found {len(shard_files)} shards in '{shards_dir}'")

    print("Connecting WebDataset stream ...")
    dataloader = get_dataloader(
        shards_pattern=shard_files,
        config=config,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    trainable_params = list(encoder.backbone.parameters()) + list(aggregator.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    start_epoch = load_checkpoints(encoder, aggregator, optimizer, device)
    total_epochs = train_cfg["num_epochs"]
    steps_per_epoch = 2000  # Virtual epoch size
    global_step = start_epoch * steps_per_epoch
    
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    temp = contrast_cfg.get("temperature", 0.07)
    safe_radius_meters = contrast_cfg.get("safe_radius_meters", 100.0)

    print(f"\n--- Commencing Contrastive Training on {device} ---")
    data_iter = iter(dataloader)

    for epoch in range(start_epoch, total_epochs):
        encoder.train()
        aggregator.train()

        epoch_loss = 0.0
        loop = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for step in loop:
            optimizer.zero_grad()
            accum_loss = 0.0

            for accum_i in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                imgs1, imgs2, lats, lons = batch
                imgs1 = imgs1.to(device, non_blocking=True)
                imgs2 = imgs2.to(device, non_blocking=True)
                lats = lats.to(device, non_blocking=True)
                lons = lons.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    feats1 = encoder(imgs1)
                    feats2 = encoder(imgs2)
                    
                    q1 = aggregator(feats1)
                    q2 = aggregator(feats2)

                    loss = info_nce_loss(q1, q2, lats, lons, temp, safe_radius_meters)
                    loss = loss / accum_steps
                    
                    loss.backward()

                accum_loss += loss.item()
                del imgs1, imgs2, lats, lons, feats1, feats2, q1, q2, loss

            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            torch.cuda.empty_cache()

            epoch_loss  += accum_loss
            global_step += 1
            
            loop.set_postfix({"loss": f"{accum_loss:.4f}"})

            if WANDB_AVAILABLE:
                wandb.log({
                    "loss": accum_loss,
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} | Avg InfoNCE Loss: {avg_loss:.6f}")

        if WANDB_AVAILABLE:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})

        save_checkpoints(encoder, aggregator, optimizer, epoch)

if __name__ == "__main__":
    main()
