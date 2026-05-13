"""
Contrastive Training Loop (InfoNCE) — Baukultur VPR v2

Network:
  Encoder (DoRA on DINOv3) -> Aggregator (GeM + MLP) -> L2 Normalized Signature

Loss:
  InfoNCE with offline-mined hard negatives. No memory bank, no geo-masking at
  train time — the hard negatives are pre-filtered by mine_negatives.py.
"""

import os
import math
import yaml
import argparse
import multiprocessing
import torch
import torch.utils.checkpoint

# CRITICAL: Use 'spawn' instead of 'fork' to prevent SIGSEGV from
# forked workers inheriting the 843M-param ViT-H GPU memory layout.
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

_original_ckpt = torch.utils.checkpoint.checkpoint

def safe_checkpoint(*args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return _original_ckpt(*args, **kwargs)

torch.utils.checkpoint.checkpoint = safe_checkpoint

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed — telemetry disabled.")

import sys
sys.path.insert(0, "./baukultur_vpr_v2")
from models.encoder import VisionEncoder
from models.aggregator import AggregatorHead
from data.mining_dataloader import get_mining_dataloader

def load_config(path: str = "config_v2.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# InfoNCE Loss (clean, hard-negative aware)
# ---------------------------------------------------------------------------

def info_nce_loss(q_query, q_positive, q_negatives, temp):
    """
    Clean InfoNCE loss with pre-mined hard negatives.
    
    Args:
        q_query:     [B, D] L2-normalized query embeddings
        q_positive:  [B, D] L2-normalized positive embeddings
        q_negatives: [B, K, D] L2-normalized hard negative embeddings
        temp:        temperature scalar
    
    Returns:
        scalar loss
    """
    B, D = q_query.shape
    K = q_negatives.shape[1]
    
    # Cast all to float32 for numerical stability
    q_query = q_query.float()
    q_positive = q_positive.float()
    q_negatives = q_negatives.float()
    
    # Positive similarity: [B]
    pos_sim = (q_query * q_positive).sum(dim=-1) / temp
    
    # Negative similarities: [B, K]
    neg_sim = torch.bmm(q_negatives, q_query.unsqueeze(-1)).squeeze(-1) / temp
    
    # Also include in-batch negatives (other queries in the batch)
    # [B, B] — each query against all other queries
    inbatch_sim = torch.matmul(q_query, q_query.T) / temp
    # Mask out self-similarity
    mask_self = torch.eye(B, dtype=torch.bool, device=q_query.device)
    inbatch_sim = inbatch_sim.masked_fill(mask_self, -1e9)
    
    # Concatenate: [positive, hard negatives, in-batch negatives]
    # logits shape: [B, 1 + K + B]
    logits = torch.cat([
        pos_sim.unsqueeze(1),     # [B, 1]
        neg_sim,                   # [B, K]
        inbatch_sim,               # [B, B]
    ], dim=1)
    
    # Labels: the positive is always at index 0
    labels = torch.zeros(B, dtype=torch.long, device=q_query.device)
    
    return F.cross_entropy(logits, labels)

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

CKPT_DIR = "models/weights/checkpoints_v2"

def _ckpt(name: str) -> str:
    return os.path.join(CKPT_DIR, name)

def save_checkpoints(encoder, aggregator, optimizer, epoch: int, global_step: int = -1):
    os.makedirs(CKPT_DIR, exist_ok=True)
    encoder.backbone.save_pretrained(_ckpt("lora_latest"))
    torch.save(aggregator.state_dict(), _ckpt("agg_latest.pt"))
    torch.save(optimizer.state_dict(),  _ckpt("optimizer_latest.pt"))

    with open(_ckpt("epoch.txt"), "w") as f:
        f.write(str(epoch))
    if global_step >= 0:
        with open(_ckpt("step.txt"), "w") as f:
            f.write(str(global_step))
    elif os.path.exists(_ckpt("step.txt")):
        os.remove(_ckpt("step.txt"))
    print(f"Checkpoints saved (epoch {epoch}, step {global_step}).")

    import time
    import shutil
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(CKPT_DIR, "archive", f"run_{timestamp}_epoch_{epoch}")
        os.makedirs(archive_dir, exist_ok=True)
        shutil.copytree(_ckpt("lora_latest"), os.path.join(archive_dir, "lora_latest"))
        shutil.copy2(_ckpt("agg_latest.pt"), archive_dir)
        shutil.copy2(_ckpt("optimizer_latest.pt"), archive_dir)
        shutil.copy2(_ckpt("epoch.txt"), archive_dir)
        if global_step >= 0:
            shutil.copy2(_ckpt("step.txt"), archive_dir)
    except Exception as e:
        print(f"  [WARNING] Archive copy failed ({e}), primary checkpoint is safe.")

def load_checkpoints(encoder, aggregator, optimizer, device, steps_per_epoch=500):
    start_epoch = 0
    start_global_step = 0
    epoch_file = _ckpt("epoch.txt")
    step_file = _ckpt("step.txt")
    if not os.path.exists(epoch_file):
        return start_epoch, start_global_step

    try:
        with open(epoch_file) as f:
            start_epoch = int(f.read().strip())
        if os.path.exists(step_file):
            with open(step_file) as f:
                start_global_step = int(f.read().strip())
            print(f"Resuming from epoch {start_epoch+1} at step {start_global_step} (intra-epoch resume) ...")
        else:
            start_epoch += 1
            start_global_step = start_epoch * steps_per_epoch
            print(f"Resuming from start of epoch {start_epoch+1} ...")
        
        # PEFT expects absolute paths
        lora_ckpt = os.path.abspath(_ckpt("lora_latest"))
        encoder.backbone.load_adapter(lora_ckpt, "default")
        aggregator.load_state_dict(torch.load(_ckpt("agg_latest.pt"), map_location=device))
        optimizer.load_state_dict(torch.load(_ckpt("optimizer_latest.pt"), map_location=device))
            
    except Exception as e:
        print(f"Warning: checkpoint loading failed ({e}). Starting from scratch.")
        start_epoch = 0
        start_global_step = 0

    return start_epoch, start_global_step

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V2 Contrastive Training")
    parser.add_argument("--config", default="config_v2.yaml")
    parser.add_argument("--parquet", default=None, help="Override parquet file path")
    parser.add_argument("--max-epoch", type=int, default=None, help="Stop after this epoch (exclusive)")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers (0 = main process only)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_cfg     = config["training"]
    wandb_cfg     = config.get("wandb", {})
    contrast_cfg  = config["contrastive"]
    mining_cfg    = config.get("mining", {})

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
                "grad_accum":     accum_steps,
                "mining":         "hard_negative",
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

    # --- Mining Dataloader ---
    parquet_path = args.parquet or mining_cfg.get("negatives_file", "hard_negatives.parquet")
    shards_dir = config["data"]["output_shards_dir"]
    num_negatives = mining_cfg.get("negatives_per_query", 8)
    
    print(f"Loading hard-negative mapping from {parquet_path} ...")
    num_workers = args.num_workers if args.num_workers is not None else train_cfg["num_workers"]
    dataloader = get_mining_dataloader(
        parquet_path=parquet_path,
        shards_dir=shards_dir,
        config=config,
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        num_negatives=num_negatives,
    )

    # Split parameters for different learning rates
    lora_params = list(encoder.backbone.parameters())
    agg_params = list(aggregator.parameters())
    trainable_params = lora_params + agg_params
    
    base_lr = float(train_cfg["learning_rate"])
    lora_lr = 5.0e-6
    
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr, "initial_lr": lora_lr},
            {"params": agg_params, "lr": base_lr, "initial_lr": base_lr}
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    steps_per_epoch = 500   # Virtual epoch size
    start_epoch, global_step = load_checkpoints(encoder, aggregator, optimizer, device, steps_per_epoch=steps_per_epoch)
    total_epochs = args.max_epoch if args.max_epoch is not None else train_cfg["num_epochs"]
    
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    temp = contrast_cfg.get("temperature", 0.07)

    print(f"\n--- Commencing Hard-Negative Contrastive Training on {device} ---")

    for epoch in range(start_epoch, total_epochs):
        encoder.train()
        aggregator.train()

        # Warmup configuration
        warmup_steps = 200
        epoch_loss = 0.0
        steps_so_far = global_step % steps_per_epoch
        if steps_so_far > 0:
            print(f"Skipping first {steps_so_far} steps of epoch {epoch+1} due to resumption.", flush=True)
            
        loop = tqdm(range(steps_so_far, steps_per_epoch), desc=f"Epoch {epoch+1}/{total_epochs}", initial=steps_so_far, total=steps_per_epoch)
        
        data_iter = iter(dataloader)
        
        for step in loop:
            optimizer.zero_grad()
            accum_loss = 0.0

            for accum_i in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                query_imgs, pos_imgs, neg_imgs, lats, lons = batch
                # query_imgs: [B, C, H, W]
                # pos_imgs:   [B, C, H, W]
                # neg_imgs:   [B, K, C, H, W]
                
                query_imgs = query_imgs.to(device, non_blocking=True)
                pos_imgs = pos_imgs.to(device, non_blocking=True)
                neg_imgs = neg_imgs.to(device, non_blocking=True)
                
                B, K, C, H, W = neg_imgs.shape

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Encode query and positive
                    q_feats = encoder(query_imgs)
                    p_feats = encoder(pos_imgs)
                    
                    q_query = F.normalize(aggregator(q_feats), p=2, dim=-1)
                    q_positive = F.normalize(aggregator(p_feats), p=2, dim=-1)
                    
                    # Encode hard negatives: reshape [B*K, C, H, W] → encode → reshape [B, K, D]
                    neg_flat = neg_imgs.reshape(B * K, C, H, W)
                    n_feats = encoder(neg_flat)
                    q_negatives = F.normalize(aggregator(n_feats), p=2, dim=-1)
                    q_negatives = q_negatives.reshape(B, K, -1)
                    
                    loss = info_nce_loss(q_query, q_positive, q_negatives, temp)
                    loss = loss / accum_steps
                    
                    if not math.isfinite(loss.item()):
                        print(f"\n[WARNING] NaN/Inf loss detected. Skipping.", flush=True)
                        optimizer.zero_grad()
                        continue
                    
                    loss.backward()

                accum_loss += loss.item()
                del query_imgs, pos_imgs, neg_imgs, q_feats, p_feats, n_feats
                del q_query, q_positive, q_negatives, neg_flat, loss

            # Linear Warmup
            if global_step < warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * pg.get('initial_lr', base_lr)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()

            epoch_loss  += accum_loss
            global_step += 1
            
            loop.set_postfix({"loss": f"{accum_loss:.4f}"})

            if global_step > 0 and global_step % 50 == 0:
                print(f"\n[Checkpoint] Saving at step {global_step}...")
                save_checkpoints(encoder, aggregator, optimizer, epoch, global_step=global_step)

            if WANDB_AVAILABLE:
                wandb.log({
                    "loss": accum_loss,
                    "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "grad_clipped": 1.0 if (torch.is_tensor(grad_norm) and grad_norm.item() > grad_clip) or (not torch.is_tensor(grad_norm) and grad_norm > grad_clip) else 0.0,
                    "lr_agg": optimizer.param_groups[1]['lr'],
                    "lr_lora": optimizer.param_groups[0]['lr'],
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
