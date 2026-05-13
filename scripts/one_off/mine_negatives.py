"""
Offline Hard-Negative Mining for Baukultur VPR V2.

Extracts 1280-dim L2-normalized signatures from all shards using either the
pretrained DINOv3 ViT-H backbone (round 0) or the trained V2 encoder (round N),
builds a FAISS cosine-similarity index, and saves a versioned parquet mapping.

Usage:
    python mine_negatives.py [--config config_v2.yaml] [--round 0] [--use-trained]
"""

import os
import sys
import json
import glob
import math
import argparse
import tarfile
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import pandas as pd
from PIL import Image
from torchvision.transforms import v2 as T

import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed — mining telemetry disabled.")

# ---------------------------------------------------------------------------
# Lightweight inference encoder (no LoRA, no PEFT dependency)
# ---------------------------------------------------------------------------

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=1)
        x = x.pow(1.0 / self.p)
        return x


def build_inference_encoder(config):
    """Build a frozen DINOv3 backbone + GeM pool for feature extraction (no LoRA)."""
    enc_cfg = config["model"]["encoder"]
    
    local_weights_path = enc_cfg.get("local_weights_path")
    if not local_weights_path or not os.path.exists(local_weights_path):
        raise FileNotFoundError(f"Weights not found at '{local_weights_path}'.")
    
    model_name = enc_cfg["name"]
    hub_repo = "facebookresearch/dinov3"
    
    print(f"Loading {model_name} structure (pretrained=False) ...")
    backbone = torch.hub.load(hub_repo, model_name, pretrained=False)
    print(f"Injecting weights from: {local_weights_path} ...")
    state_dict = torch.load(local_weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict)
    
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    
    embed_dim = enc_cfg.get("embed_dim", 1280)
    pool = GeMPooling(p=3.0)
    
    return backbone.to(torch.bfloat16), pool.to(torch.bfloat16), embed_dim


def extract_features_from_backbone(backbone, pool, images, device):
    """Run images through the frozen backbone and pool to get 1280-dim signatures."""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # DINOv3 forward: prepare_tokens_with_masks → blocks → patch tokens
        features, (H, W) = backbone.prepare_tokens_with_masks(images)
        rope_sincos = backbone.rope_embed(H=H, W=W) if hasattr(backbone, "rope_embed") else None
        for blk in backbone.blocks:
            features = blk(features, rope_sincos)
        
        patch_tokens = features[:, 1:, :]  # Skip [CLS]
        pooled = pool(patch_tokens)  # [B, D]
        normalized = F.normalize(pooled.float(), p=2, dim=-1)  # L2 normalize in float32
    
    return normalized


def build_trained_encoder(config, device):
    """
    Build the TRAINED V2 encoder (LoRA + AggregatorHead) for iterative mining.
    Loads checkpoints from models/weights/checkpoints_v2/.
    """
    import sys
    sys.path.insert(0, "./baukultur_vpr_v2")
    from models.encoder import VisionEncoder
    from models.aggregator import AggregatorHead
    
    ckpt_dir = "models/weights/checkpoints_v2"
    lora_ckpt = os.path.abspath(os.path.join(ckpt_dir, "lora_latest"))
    agg_ckpt = os.path.join(ckpt_dir, "agg_latest.pt")
    
    if not os.path.exists(lora_ckpt) or not os.path.exists(agg_ckpt):
        raise FileNotFoundError(
            f"V2 checkpoints not found in '{ckpt_dir}'. "
            "Cannot use --use-trained without a completed training round."
        )
    
    print("Loading TRAINED V2 encoder for mining ...")
    encoder = VisionEncoder(config).to(device)
    encoder.backbone.load_adapter(lora_ckpt, "default")
    encoder.eval()
    
    embed_dim = config["model"]["encoder"]["embed_dim"]
    out_dim = config["model"]["aggregator"]["output_dim"]
    aggregator = AggregatorHead(embed_dim=embed_dim, out_dim=out_dim).to(device)
    aggregator.load_state_dict(torch.load(agg_ckpt, map_location=device))
    aggregator.eval()
    
    for p in encoder.parameters():
        p.requires_grad = False
    for p in aggregator.parameters():
        p.requires_grad = False
    
    print(f"Loaded trained encoder + aggregator from {ckpt_dir}")
    return encoder, aggregator


def extract_features_trained(encoder, aggregator, images, device):
    """Run images through the trained V2 encoder + aggregator."""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        feats = encoder(images)
        sigs = aggregator(feats)
        normalized = F.normalize(sigs.float(), p=2, dim=-1)
    return normalized


# ---------------------------------------------------------------------------
# Haversine for GPS filtering
# ---------------------------------------------------------------------------

def haversine_meters(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters (numpy)."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# Shard scanning & feature extraction
# ---------------------------------------------------------------------------

def scan_and_extract(config, backbone, pool, device, batch_size=64,
                     trained_encoder=None, trained_aggregator=None,
                     ckpt_path="mining_checkpoint.pt"):
    """
    Stream through all tar shards, extract features and metadata.
    If trained_encoder/trained_aggregator are provided, uses them instead of raw backbone.
    Returns: embeddings [N, D], lats [N], lons [N], sample_ids [N]
    """
    shards_dir = config["data"]["output_shards_dir"]
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "dataset-*.tar")))
    if not shard_files:
        raise RuntimeError(f"No .tar shards found in '{shards_dir}'.")
    
    print(f"Found {len(shard_files)} shards in '{shards_dir}'")
    
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((518, 518), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    ckpt_path_local = ckpt_path
    start_shard_idx = 0
    total_samples = 0
    
    if os.path.exists(ckpt_path_local):
        print(f"Resuming mining from {ckpt_path_local}...")
        ckpt = torch.load(ckpt_path_local)
        all_embeddings = ckpt["all_embeddings"]
        all_lats = ckpt["all_lats"]
        all_lons = ckpt["all_lons"]
        all_sample_ids = ckpt["all_sample_ids"]
        all_shard_names = ckpt["all_shard_names"]
        start_shard_idx = ckpt["shard_idx"] + 1
        total_samples = ckpt["total_samples"]
        print(f"Resumed at shard {start_shard_idx+1} with {total_samples} samples.")
    else:
        all_embeddings = []
        all_lats = []
        all_lons = []
        all_sample_ids = []
        all_shard_names = []
    
    use_trained = trained_encoder is not None and trained_aggregator is not None
    
    for shard_idx, shard_path in enumerate(shard_files):
        if shard_idx < start_shard_idx:
            continue
        
        shard_name = os.path.basename(shard_path)
        print(f"\n[{shard_idx+1}/{len(shard_files)}] Processing {shard_name} ...", flush=True)
        
        # Collect samples from this shard
        batch_images = []
        batch_lats = []
        batch_lons = []
        batch_ids = []
        batch_shards = []
        
        sample_idx = 0
        try:
            with tarfile.open(shard_path, 'r') as tar:
                # Group files by sample prefix
                members = sorted(tar.getnames())
                # Find unique sample prefixes (e.g., "000000" from "000000.img1.jpg")
                prefixes = defaultdict(dict)
                for name in members:
                    parts = name.split(".")
                    if len(parts) >= 2:
                        prefix = parts[0]
                        suffix = ".".join(parts[1:])
                        prefixes[prefix][suffix] = name
                
                for prefix in sorted(prefixes.keys()):
                    files = prefixes[prefix]
                    
                    # We need at least img1.jpg and meta.json
                    img_key = None
                    meta_key = None
                    for k in files:
                        if "img1" in k and k.endswith("jpg"):
                            img_key = k
                        elif k.endswith("json"):
                            meta_key = k
                    
                    if img_key is None:
                        continue
                    
                    try:
                        # Read image
                        img_member = tar.getmember(files[img_key])
                        img_data = tar.extractfile(img_member).read()
                        img = Image.open(io.BytesIO(img_data)).convert("RGB")
                        img_tensor = transform(img)
                        
                        # Read metadata for GPS
                        lat, lon = 0.0, 0.0
                        if meta_key and meta_key in files:
                            meta_member = tar.getmember(files[meta_key])
                            meta_raw = tar.extractfile(meta_member).read()
                            meta = json.loads(meta_raw.decode("utf-8"))
                            lat = float(meta.get("lat", 0.0))
                            lon = float(meta.get("lon", 0.0))
                        
                        sample_id = f"{shard_name}:{prefix}"
                        batch_images.append(img_tensor)
                        batch_lats.append(lat)
                        batch_lons.append(lon)
                        batch_ids.append(sample_id)
                        batch_shards.append(shard_name)
                        sample_idx += 1
                        
                        # Process batch when full
                        if len(batch_images) >= batch_size:
                            imgs = torch.stack(batch_images).to(device)
                            if use_trained:
                                embs = extract_features_trained(trained_encoder, trained_aggregator, imgs, device)
                            else:
                                embs = extract_features_from_backbone(backbone, pool, imgs, device)
                            all_embeddings.append(embs.cpu())
                            all_lats.extend(batch_lats)
                            all_lons.extend(batch_lons)
                            all_sample_ids.extend(batch_ids)
                            all_shard_names.extend(batch_shards)
                            total_samples += len(batch_images)
                            batch_images, batch_lats, batch_lons, batch_ids, batch_shards = [], [], [], [], []
                    
                    except Exception as e:
                        continue
                
        except Exception as e:
            print(f"  [WARNING] Failed to read shard {shard_name}: {e}", flush=True)
            continue
        
        # Process remaining samples in last batch
        if batch_images:
            imgs = torch.stack(batch_images).to(device)
            if use_trained:
                embs = extract_features_trained(trained_encoder, trained_aggregator, imgs, device)
            else:
                embs = extract_features_from_backbone(backbone, pool, imgs, device)
            all_embeddings.append(embs.cpu())
            all_lats.extend(batch_lats)
            all_lons.extend(batch_lons)
            all_sample_ids.extend(batch_ids)
            all_shard_names.extend(batch_shards)
            total_samples += len(batch_images)
        
        print(f"  Extracted {sample_idx} samples (total: {total_samples})", flush=True)
        
        # W&B per-shard logging
        if WANDB_AVAILABLE:
            wandb.log({
                "mining/shard_idx": shard_idx + 1,
                "mining/shard_samples": sample_idx,
                "mining/total_samples": total_samples,
                "mining/progress_pct": (shard_idx + 1) / len(shard_files) * 100,
            }, step=shard_idx + 1)
        
        # Checkpoint every 20 shards
        if (shard_idx + 1) % 20 == 0:
            print(f"  [Checkpoint] Saving progress at shard {shard_idx + 1}...")
            torch.save({
                "all_embeddings": all_embeddings,
                "all_lats": all_lats,
                "all_lons": all_lons,
                "all_sample_ids": all_sample_ids,
                "all_shard_names": all_shard_names,
                "shard_idx": shard_idx,
                "total_samples": total_samples
            }, ckpt_path_local)
    
    if os.path.exists(ckpt_path_local):
        os.remove(ckpt_path_local)  # Clean up intermediate checkpoint after full completion
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    lats = np.array(all_lats, dtype=np.float64)
    lons = np.array(all_lons, dtype=np.float64)
    
    print(f"\nTotal extracted: {len(all_sample_ids)} samples, embeddings shape: {embeddings.shape}")
    return embeddings, lats, lons, all_sample_ids, all_shard_names


# ---------------------------------------------------------------------------
# FAISS index + geographic filtering
# ---------------------------------------------------------------------------

def build_index_and_mine(embeddings, lats, lons, sample_ids, shard_names,
                         top_k=50, num_negatives=8, safe_radius=100.0, output_path="hard_negatives.parquet"):
    """Build FAISS index, find nearest neighbors, filter by GPS, save mapping."""
    N, D = embeddings.shape
    print(f"\nBuilding FAISS IndexFlatIP ({N} vectors, {D} dims) ...")
    
    # Normalize embeddings (should already be, but ensure)
    faiss.normalize_L2(embeddings)
    
    # Try GPU FAISS if available
    index = faiss.IndexFlatIP(D)
    if faiss.get_num_gpus() > 0:
        print("Using GPU FAISS")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embeddings)
    print(f"Index built. Searching top-{top_k} neighbors ...")
    
    # Search in batches to avoid OOM
    search_batch = 4096
    all_distances = []
    all_indices = []
    
    for i in range(0, N, search_batch):
        end = min(i + search_batch, N)
        D_batch, I_batch = index.search(embeddings[i:end], top_k + 1)  # +1 for self-match
        all_distances.append(D_batch)
        all_indices.append(I_batch)
        if (i // search_batch) % 10 == 0:
            print(f"  Searched {end}/{N} ...", flush=True)
    
    distances = np.concatenate(all_distances, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    
    print(f"Filtering by geographic distance (safe_radius={safe_radius}m) ...")
    
    records = []
    skipped = 0
    
    for i in range(N):
        hard_neg_ids = []
        hard_neg_shards = []
        hard_neg_sims = []
        
        for j_rank in range(top_k + 1):
            j = indices[i, j_rank]
            if j == i:  # Skip self
                continue
            
            # Filter by GPS distance
            dist_m = haversine_meters(lats[i], lons[i], lats[j], lons[j])
            if dist_m < safe_radius:
                continue  # Too close geographically = potential false negative
            
            hard_neg_ids.append(sample_ids[j])
            hard_neg_shards.append(shard_names[j])
            hard_neg_sims.append(float(distances[i, j_rank]))
            
            if len(hard_neg_ids) >= num_negatives:
                break
        
        if hard_neg_ids:
            records.append({
                "query_id": sample_ids[i],
                "query_shard": shard_names[i],
                "query_lat": lats[i],
                "query_lon": lons[i],
                "hard_neg_ids": hard_neg_ids,
                "hard_neg_shards": hard_neg_shards,
                "hard_neg_sims": hard_neg_sims,
            })
        else:
            skipped += 1
    
    df = pd.DataFrame(records)
    df.to_parquet(output_path)
    
    coverage = len(records) / N * 100
    avg_negs = df["hard_neg_ids"].apply(len).mean() if len(records) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Mining Complete!")
    print(f"  Total samples:    {N}")
    print(f"  With negatives:   {len(records)} ({coverage:.1f}%)")
    print(f"  Skipped (no neg): {skipped}")
    print(f"  Avg negatives:    {avg_negs:.1f}")
    print(f"  Saved to:         {output_path}")
    print(f"{'='*60}")
    
    # W&B final summary
    if WANDB_AVAILABLE:
        wandb.log({
            "mining/total_samples_final": N,
            "mining/with_negatives": len(records),
            "mining/coverage_pct": coverage,
            "mining/skipped": skipped,
            "mining/avg_negatives": avg_negs,
        })
    
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline Hard-Negative Mining for V2")
    parser.add_argument("--config", default="config_v2.yaml", help="Path to config file")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K FAISS neighbors to consider")
    parser.add_argument("--num-negatives", type=int, default=8, help="Number of hard negatives per query")
    parser.add_argument("--safe-radius", type=float, default=100.0, help="GPS safe radius in meters")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--round", type=int, default=0, help="Mining round (0 = initial, N = iterative)")
    parser.add_argument("--use-trained", action="store_true", help="Use trained V2 encoder instead of raw DINOv3")
    args = parser.parse_args()
    
    # Version the output file by round
    output_path = f"hard_negatives_round_{args.round}.parquet"
    ckpt_path = f"mining_checkpoint_round_{args.round}.pt"
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mining round: {args.round}")
    print(f"Output: {output_path}")
    
    # --- W&B ---
    if WANDB_AVAILABLE:
        wandb_cfg = config.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "baukultur_vpr_v2"),
            entity=wandb_cfg.get("entity") or None,
            mode="offline" if wandb_cfg.get("offline", False) else "online",
            name=f"mining_round_{args.round}",
            config={
                "round": args.round,
                "use_trained": args.use_trained,
                "top_k": args.top_k,
                "num_negatives": args.num_negatives,
                "safe_radius": args.safe_radius,
            },
        )
    
    trained_encoder = None
    trained_aggregator = None
    backbone = None
    pool = None
    
    if args.use_trained and args.round > 0:
        # Use the trained V2 encoder for iterative mining
        trained_encoder, trained_aggregator = build_trained_encoder(config, device)
    else:
        # Use raw DINOv3 backbone for initial mining
        backbone, pool, embed_dim = build_inference_encoder(config)
        backbone = backbone.to(device)
        pool = pool.to(device)
    
    # Extract features from all shards
    embeddings, lats, lons, sample_ids, shard_names = scan_and_extract(
        config, backbone, pool, device, batch_size=args.batch_size,
        trained_encoder=trained_encoder, trained_aggregator=trained_aggregator,
        ckpt_path=ckpt_path,
    )
    
    # Free GPU memory before FAISS
    del backbone, pool, trained_encoder, trained_aggregator
    torch.cuda.empty_cache()
    
    # Build index and mine hard negatives
    build_index_and_mine(
        embeddings, lats, lons, sample_ids, shard_names,
        top_k=args.top_k,
        num_negatives=args.num_negatives,
        safe_radius=args.safe_radius,
        output_path=output_path,
    )
    
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
