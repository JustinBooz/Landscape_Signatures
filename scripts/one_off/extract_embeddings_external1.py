"""
Extract DINOv3 1280D embeddings from External_1 shards → parquet.
Designed to run alongside the ingestion pipeline without starving CPU.
Built-in checkpointing via chunked npz files.
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import webdataset as wds
from torchvision.transforms import v2 as T

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from baukultur_vpr.models.encoder import VisionEncoder
import yaml

CACHE_DIR = "external1_embeddings_cache"
OUTPUT_PARQUET = "training_embeddings_external1.parquet"
SHARDS_DIR = "/media/jubooz/External_1/Shards"
CONFIG_PATH = "config_v1.yaml"
CHUNK_SIZE = 10000
BATCH_SIZE = 64
NUM_WORKERS = 2  # Low to avoid starving ingestion pipeline CPU

EVAL_TRANSFORM = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((518, 518), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def decode_sample(sample):
    img_pil, meta = sample
    try:
        lat = float(meta.get("lat", 0.0))
        lon = float(meta.get("lon", 0.0))
    except Exception:
        lat, lon = 0.0, 0.0
    return EVAL_TRANSFORM(img_pil), lat, lon

@torch.no_grad()
def extract():
    os.makedirs(CACHE_DIR, exist_ok=True)
    flag_file = os.path.join(CACHE_DIR, "extraction_complete.flag")

    if os.path.exists(flag_file):
        print(f"\n[CACHE HIT] Extraction already complete.")
        return

    # Resume from existing chunks
    existing_chunks = sorted(glob.glob(os.path.join(CACHE_DIR, "chunk_*.npz")))
    chunk_idx = len(existing_chunks)
    samples_done = 0
    if chunk_idx > 0:
        for cf in existing_chunks:
            samples_done += len(np.load(cf)['emb'])
        print(f"\n[RESUME] Found {chunk_idx} chunks ({samples_done:,} samples). Skipping ahead...")
    else:
        print(f"\n[FRESH] Starting extraction from scratch...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print(f"\n[Loading Frozen DINOv3 Zero-Shot Backbone]")
    encoder = VisionEncoder(config).to(device).eval()

    shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "dataset-*.tar")))
    print(f"Found {len(shard_files)} shards on External_1")

    dataset = (wds.WebDataset(shard_files, shardshuffle=False, handler=wds.warn_and_continue)
                  .decode("pil", handler=wds.warn_and_continue)
                  .to_tuple("img1.jpg", "meta.json", handler=wds.warn_and_continue)
                  .map(decode_sample))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=2
    )

    # Fast-forward past already-processed samples
    data_iter = iter(dataloader)
    if samples_done > 0:
        batches_to_skip = samples_done // BATCH_SIZE
        print(f"  Fast-forwarding {batches_to_skip} batches...", flush=True)
        import itertools
        data_iter = itertools.islice(data_iter, batches_to_skip, None)

    emb_chunk, coords_chunk = [], []
    collected = samples_done

    print(f"\nExtracting embeddings (resuming from {samples_done:,})...", flush=True)

    for batch in data_iter:
        imgs, lats, lons = batch
        imgs = imgs.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            e = encoder(imgs, use_target=False)
            e = F.normalize(e, p=2, dim=-1)

        emb_chunk.append(e.cpu().float().numpy())
        coords_chunk.append(torch.stack([lats, lons], dim=1).numpy())

        collected += imgs.size(0)
        print(f"  Processed {collected:,} samples...", end='\r', flush=True)

        if sum(len(x) for x in emb_chunk) >= CHUNK_SIZE:
            e_c = np.vstack(emb_chunk)
            cd_c = np.vstack(coords_chunk)
            cfile = os.path.join(CACHE_DIR, f"chunk_{chunk_idx:04d}.npz")
            np.savez_compressed(cfile, emb=e_c, coords=cd_c)
            print(f"\n  [Checkpoint] chunk_{chunk_idx:04d} ({len(e_c):,} samples) → {cfile}", flush=True)
            emb_chunk, coords_chunk = [], []
            chunk_idx += 1

    if emb_chunk:
        e_c = np.vstack(emb_chunk)
        cd_c = np.vstack(coords_chunk)
        cfile = os.path.join(CACHE_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(cfile, emb=e_c, coords=cd_c)
        print(f"\n  [Checkpoint] Final chunk_{chunk_idx:04d} ({len(e_c):,} samples)", flush=True)

    with open(flag_file, "w") as f:
        f.write(f"Total processed: {collected}\n")
    print(f"\nExtraction complete — {collected:,} total samples.", flush=True)

    del encoder
    torch.cuda.empty_cache()

def build_parquet():
    print(f"\nBuilding parquet from cached chunks...")
    chunks = sorted(glob.glob(os.path.join(CACHE_DIR, "chunk_*.npz")))
    embs_all, coords_all = [], []
    for cf in chunks:
        d = np.load(cf)
        embs_all.append(d['emb'])
        coords_all.append(d['coords'])

    embs = np.vstack(embs_all)
    coords = np.vstack(coords_all)

    data = {'lat': coords[:, 0], 'lon': coords[:, 1]}
    for i in range(embs.shape[1]):
        data[f'emb_{i:04d}'] = embs[:, i]

    df = pd.DataFrame(data)
    df.to_parquet(OUTPUT_PARQUET, index=False, engine='pyarrow')
    size_mb = os.path.getsize(OUTPUT_PARQUET) / 1e6
    print(f"Written {OUTPUT_PARQUET}: {len(df):,} rows, {size_mb:.0f} MB")

if __name__ == "__main__":
    extract()
    build_parquet()
