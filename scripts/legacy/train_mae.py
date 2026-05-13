"""
Masked Autoencoder (MAE) Training — Landscape Signatures

Architecture:
  Encoder (ViT-L/14): 24 blocks, 1024 dim, 16 heads — processes only unmasked patches
  Decoder: 8 blocks, 512 dim, 16 heads — reconstructs masked patches
  
Loss:
  MSE on masked patches with per-patch normalized pixel targets

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022)
"""

import os
import glob
import math
import yaml
import gc
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2 as T
from tqdm import tqdm

import webdataset as wds

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed — telemetry disabled.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config_mae.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Positional Embeddings (2D sincos, fixed)
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings.
    Returns: [grid_size*grid_size, embed_dim] or [1 + grid_size*grid_size, embed_dim]
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, -1)  # [2, H*W]

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # [H*W, D]

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)

    return pos_embed


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sincos positional embedding from positions."""
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))

    out = pos.unsqueeze(1) * omega.unsqueeze(0)  # [N, D/2]
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)  # [N, D]


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=qkv_bias, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Masked Autoencoder
# ---------------------------------------------------------------------------

class MaskedAutoencoder(nn.Module):
    """
    MAE with ViT-L/14 encoder and lightweight decoder.
    
    Key optimization: encoder processes only unmasked patches (25% of total),
    saving ~75% of encoder FLOPS and memory.
    """
    def __init__(
        self,
        image_size=518,
        patch_size=14,
        # Encoder (ViT-L)
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        # Decoder (lightweight)
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        # Masking
        mask_ratio=0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        
        # Patches
        self.grid_size = image_size // patch_size  # 37
        self.num_patches = self.grid_size ** 2      # 1369
        self.patch_dim = patch_size * patch_size * 3  # 588
        
        # --- Encoder ---
        self.patch_embed = nn.Linear(self.patch_dim, encoder_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        
        # Fixed sincos positional embedding (with CLS)
        pos_embed = get_2d_sincos_pos_embed(encoder_embed_dim, self.grid_size, cls_token=True)
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)  # [1, 1+N, D]
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_embed_dim, encoder_num_heads)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)
        
        # --- Decoder ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embedding (with CLS)
        dec_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, self.grid_size, cls_token=True)
        self.decoder_pos_embed = nn.Parameter(dec_pos_embed.unsqueeze(0), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: project back to raw patch pixels
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform + trunc normal for tokens, matching MAE paper."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def patchify(self, imgs):
        """
        Convert images to patch sequences.
        imgs: [B, 3, H, W]
        Returns: [B, N, patch_dim]   (N = 1369, patch_dim = 588)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.image_size
        
        # Reshape: [B, 3, 37, 14, 37, 14] → [B, 37, 37, 14, 14, 3] → [B, 1369, 588]
        x = imgs.reshape(B, C, self.grid_size, p, self.grid_size, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, G, G, P, P, C]
        x = x.reshape(B, self.num_patches, self.patch_dim)
        return x
    
    def unpatchify(self, x):
        """
        Reconstruct images from patch sequences (for visualization).
        x: [B, N, patch_dim]
        Returns: [B, 3, H, W]
        """
        p = self.patch_size
        B = x.shape[0]
        x = x.reshape(B, self.grid_size, self.grid_size, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, G, P, G, P]
        x = x.reshape(B, 3, self.image_size, self.image_size)
        return x
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking: shuffle patch indices, keep first (1-mask_ratio) fraction.
        
        x: [B, N, D] — patch embeddings
        Returns:
            x_visible: [B, N_vis, D]
            mask: [B, N] — 0 = visible, 1 = masked
            ids_restore: [B, N] — indices to unshuffle
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))
        
        # Random permutation per sample
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first num_keep
        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Binary mask: 0 = visible, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore
    
    def forward_encoder(self, imgs):
        """
        Encode only unmasked patches.
        imgs: [B, 3, H, W]
        Returns: latent [B, N_vis+1, D], mask [B, N], ids_restore [B, N]
        """
        # Patchify
        x = self.patchify(imgs)  # [B, 1369, 588]
        
        # Embed patches
        x = self.patch_embed(x)  # [B, 1369, 1024]
        
        # Add positional embedding (skip CLS position)
        x = x + self.pos_embed[:, 1:, :]
        
        # Random masking → only keep visible patches
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        # x: [B, 342, 1024]
        
        # Prepend CLS token
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 343, 1024]
        
        # Transformer encoder
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decode: project encoder output, append mask tokens, reconstruct full sequence.
        x: [B, N_vis+1, encoder_dim]
        Returns: [B, N, patch_dim]
        """
        # Project to decoder dim
        x = self.decoder_embed(x)  # [B, N_vis+1, 512]
        
        # Append mask tokens to create full sequence
        num_mask = self.num_patches + 1 - x.shape[1]  # 1027
        mask_tokens = self.mask_token.repeat(x.shape[0], num_mask, 1)
        
        # Unshuffle: put visible + mask tokens back in original patch order
        # Separate CLS from patches
        x_no_cls = x[:, 1:, :]  # [B, 342, 512]
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)  # [B, 1369, 512]
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )  # [B, 1369, 512] — unshuffled
        
        # Re-add CLS
        x_full = torch.cat([x[:, :1, :], x_full], dim=1)  # [B, 1370, 512]
        
        # Add decoder positional embedding
        x_full = x_full + self.decoder_pos_embed
        
        # Transformer decoder
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)
        
        # Project to pixel space (skip CLS)
        pred = self.decoder_pred(x_full[:, 1:, :])  # [B, 1369, 588]
        
        return pred
    
    def forward_loss(self, imgs, pred, mask):
        """
        MSE loss on masked patches with per-patch normalized pixel targets.
        
        imgs: [B, 3, H, W]
        pred: [B, N, patch_dim]
        mask: [B, N] — 1 = masked (compute loss here)
        """
        target = self.patchify(imgs)  # [B, 1369, 588]
        
        # Per-patch pixel normalization (zero mean, unit variance)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        
        # MSE loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N] — per-patch MSE
        
        # Average over masked patches only
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, imgs):
        """Full forward pass: encode → decode → loss."""
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ---------------------------------------------------------------------------
# WebDataset Dataloader
# ---------------------------------------------------------------------------

class MAEDataset(IterableDataset):
    """Streams single images from WebDataset shards for MAE pretraining."""
    def __init__(self, dataset: wds.WebDataset, image_size: int = 518):
        super().__init__()
        self.dataset = dataset
        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((image_size, image_size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __iter__(self):
        stream = self.dataset.decode("torchrgb8", handler=wds.warn_and_continue)
        stream = stream.to_tuple("img1.jpg", handler=wds.warn_and_continue)
        
        for (img,) in stream:
            try:
                if img is None:
                    continue
                yield self.transform(img)
            except Exception:
                continue


def _worker_init_fn(worker_id):
    gc.disable()


def get_mae_dataloader(config, batch_size, num_workers):
    """Create WebDataset streaming dataloader for MAE."""
    shards_dir = config["data"]["output_shards_dir"]
    shard_files = sorted(glob.glob(os.path.join(shards_dir, "dataset-*.tar")))
    
    if not shard_files:
        raise RuntimeError(f"No .tar shards found in '{shards_dir}'.")
    print(f"Found {len(shard_files)} shards in '{shards_dir}'")
    
    shuffle_buf = config.get("training", {}).get("shuffle_buffer", 2500)
    image_size = config["model"]["image_size"]
    
    dataset = wds.WebDataset(shard_files, shardshuffle=True, resampled=True, handler=wds.warn_and_continue)
    dataset = dataset.shuffle(shuffle_buf)
    ds = MAEDataset(dataset, image_size=image_size)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )


# ---------------------------------------------------------------------------
# Cosine LR Schedule with Warmup
# ---------------------------------------------------------------------------

def cosine_lr_schedule(optimizer, epoch, num_epochs, warmup_epochs, base_lr):
    """Cosine decay with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

CKPT_DIR = "models/weights/checkpoints_mae"

def _ckpt(name):
    return os.path.join(CKPT_DIR, name)

def save_checkpoint(model, optimizer, epoch, global_step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, _ckpt("mae_latest.pt"))
    
    # Epoch archive
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, _ckpt(f"mae_epoch_{epoch}.pt"))
    
    print(f"Checkpoint saved (epoch {epoch}, step {global_step}).")


def load_checkpoint(model, optimizer, device):
    ckpt_path = _ckpt("mae_latest.pt")
    if not os.path.exists(ckpt_path):
        return 0, 0
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"Resumed from epoch {epoch}, step {global_step}")
        return epoch, global_step
    except Exception as e:
        print(f"Warning: checkpoint loading failed ({e}). Starting fresh.")
        return 0, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAE Pretraining for Landscape Signatures")
    parser.add_argument("--config", default="config_mae.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    train_cfg = config["training"]
    model_cfg = config["model"]
    wandb_cfg = config.get("wandb", {})
    
    batch_size = train_cfg["batch_size"]
    accum_steps = train_cfg["gradient_accumulation_steps"]
    effective_batch = batch_size * accum_steps
    num_epochs = train_cfg["num_epochs"]
    warmup_epochs = train_cfg["warmup_epochs"]
    base_lr = float(train_cfg["base_lr"]) * effective_batch / 256.0
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    
    steps_per_epoch = 2000  # Virtual epoch size
    
    print(f"\n{'='*60}")
    print(f"MAE Pretraining — Landscape Signatures")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  Image size:       {model_cfg['image_size']}")
    print(f"  Patch size:       {model_cfg['patch_size']}")
    print(f"  Mask ratio:       {model_cfg['mask_ratio']}")
    print(f"  Encoder:          ViT-L ({model_cfg['encoder']['depth']} blocks, {model_cfg['encoder']['embed_dim']}d)")
    print(f"  Decoder:          {model_cfg['decoder']['depth']} blocks, {model_cfg['decoder']['embed_dim']}d")
    print(f"  Physical batch:   {batch_size}")
    print(f"  Effective batch:  {effective_batch}")
    print(f"  Scaled LR:        {base_lr:.6f}")
    print(f"  Warmup epochs:    {warmup_epochs}")
    print(f"  Total epochs:     {num_epochs}")
    print(f"{'='*60}\n")
    
    # --- W&B ---
    if WANDB_AVAILABLE:
        wandb.init(
            project=wandb_cfg.get("project", "baukultur_mae"),
            entity=wandb_cfg.get("entity") or None,
            mode="offline" if wandb_cfg.get("offline", False) else "online",
            config={
                "base_lr": base_lr,
                "batch_size": batch_size,
                "effective_batch": effective_batch,
                "mask_ratio": model_cfg["mask_ratio"],
                "encoder_depth": model_cfg["encoder"]["depth"],
                "decoder_depth": model_cfg["decoder"]["depth"],
                "num_epochs": num_epochs,
            },
        )
    
    # --- Model ---
    print("Building MAE model ...")
    model = MaskedAutoencoder(
        image_size=model_cfg["image_size"],
        patch_size=model_cfg["patch_size"],
        encoder_embed_dim=model_cfg["encoder"]["embed_dim"],
        encoder_depth=model_cfg["encoder"]["depth"],
        encoder_num_heads=model_cfg["encoder"]["num_heads"],
        decoder_embed_dim=model_cfg["decoder"]["embed_dim"],
        decoder_depth=model_cfg["decoder"]["depth"],
        decoder_num_heads=model_cfg["decoder"]["num_heads"],
        mask_ratio=model_cfg["mask_ratio"],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MAE parameters: {num_params:.1f}M")
    
    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    
    # --- Dataloader ---
    print("Connecting WebDataset stream ...")
    dataloader = get_mae_dataloader(
        config,
        batch_size=batch_size,
        num_workers=train_cfg["num_workers"],
    )
    
    # --- Resume ---
    start_epoch, global_step = load_checkpoint(model, optimizer, device)
    
    print(f"\n--- Commencing MAE Training on {device} ---\n")
    
    data_iter = iter(dataloader)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        # Update LR using cosine schedule
        current_lr = cosine_lr_schedule(optimizer, epoch, num_epochs, warmup_epochs, base_lr)
        
        epoch_loss = 0.0
        steps_so_far = global_step % steps_per_epoch if epoch == start_epoch else 0
        
        loop = tqdm(
            range(steps_so_far, steps_per_epoch),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            initial=steps_so_far,
            total=steps_per_epoch,
        )
        
        for step in loop:
            optimizer.zero_grad()
            accum_loss = 0.0
            
            for _ in range(accum_steps):
                try:
                    imgs = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    imgs = next(data_iter)
                
                imgs = imgs.to(device, non_blocking=True)
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss, _, _ = model(imgs)
                    loss = loss / accum_steps
                
                if not math.isfinite(loss.item()):
                    print(f"\n[WARNING] NaN/Inf loss. Skipping.", flush=True)
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                accum_loss += loss.item()
                del imgs, loss
            
            # Gradient clipping + step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            epoch_loss += accum_loss
            global_step += 1
            
            loop.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{current_lr:.2e}"})
            
            # Checkpoint every 500 steps
            if global_step > 0 and global_step % 500 == 0:
                print(f"\n[Checkpoint] Saving at step {global_step} ...")
                save_checkpoint(model, optimizer, epoch, global_step)
            
            # W&B logging
            if WANDB_AVAILABLE:
                wandb.log({
                    "loss": accum_loss,
                    "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "lr": current_lr,
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)
        
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} | Avg MAE Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
        
        if WANDB_AVAILABLE:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
        
        save_checkpoint(model, optimizer, epoch, global_step)
    
    print("\nMAE Pretraining Complete.")
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
