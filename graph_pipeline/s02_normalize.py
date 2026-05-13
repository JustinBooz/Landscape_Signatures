"""
Step 02: Embedding Normalization
==================================
Loads all .pt files, L2-normalizes each embedding vector, and writes
to a float16 numpy memmap for efficient downstream access.

Output: embeddings_normed.mmap + embeddings_normed_shape.json
"""

import os
import sys
import json
import glob
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s02_normalize")


def run():
    mmap_path = config.normed_embeddings_path()
    shape_path = config.normed_embeddings_shape_path()
    progress_path = os.path.join(config.OUTPUT_DIR, "normalize_progress.json")

    # Check if already complete
    if os.path.exists(mmap_path) and os.path.exists(shape_path):
        with open(shape_path, 'r') as f:
            meta = json.load(f)
        if meta.get('complete', False):
            logger.info(f"[CHECKPOINT] Normalized embeddings already exist: "
                        f"{meta['shape']}")
            return

    pt_files = sorted(glob.glob(os.path.join(config.PT_DIR, "*.pt")))
    logger.info(f"Found {len(pt_files)} .pt files")

    t0 = time.time()

    # ---- Pass 1: Count total rows (cached across restarts) ----
    counts_path = os.path.join(config.OUTPUT_DIR, "normalize_row_counts.json")
    row_counts = None
    if os.path.exists(counts_path):
        try:
            with open(counts_path) as f:
                cached = json.load(f)
            if cached.get('n_files') == len(pt_files) and cached.get('total'):
                row_counts = cached['row_counts']
                logger.info(f"  Loaded cached row counts: total={cached['total']:,}")
        except Exception:
            row_counts = None

    if row_counts is None:
        logger.info("Pass 1: Counting rows...")
        row_counts = []
        for i, pt_path in enumerate(pt_files):
            d = torch.load(pt_path, map_location='cpu', weights_only=False)
            row_counts.append(d['embeddings'].shape[0])
            del d
            if (i + 1) % 1000 == 0:
                logger.info(f"  Counted {i + 1}/{len(pt_files)} files")
        total = sum(row_counts)
        logger.info(f"  Total: {total:,} embeddings")
        with open(counts_path, 'w') as f:
            json.dump({'n_files': len(pt_files), 'total': total,
                       'row_counts': row_counts}, f)
    else:
        total = sum(row_counts)

    # ---- Create memmap ----
    shape = (total, config.EMBEDDING_DIM)
    logger.info(f"Creating memmap: {shape} float16 "
                f"({total * config.EMBEDDING_DIM * 2 / 1e9:.1f} GB)")

    # Check for partial progress
    start_file_idx = 0
    start_offset = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            prog = json.load(f)
        start_file_idx = prog.get('last_completed_file', -1) + 1
        start_offset = sum(row_counts[:start_file_idx])
        logger.info(f"  Resuming from file {start_file_idx} "
                     f"(offset {start_offset:,})")

    # Create or open memmap
    if start_file_idx == 0:
        mmap = np.memmap(mmap_path, dtype='float16', mode='w+', shape=shape)
    else:
        mmap = np.memmap(mmap_path, dtype='float16', mode='r+', shape=shape)

    # ---- Pass 2: Normalize and write ----
    logger.info("Pass 2: Normalizing and writing to memmap...")
    offset = start_offset

    for i in range(start_file_idx, len(pt_files)):
        pt_path = pt_files[i]
        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        emb = d['embeddings']  # bfloat16 tensor
        n = emb.shape[0]

        # Cast to float32, L2-normalize, cast to float16
        emb_f32 = emb.float().numpy()  # [n, 4096] float32
        norms = np.linalg.norm(emb_f32, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        emb_normed = (emb_f32 / norms).astype(np.float16)

        mmap[offset:offset + n] = emb_normed
        offset += n

        del d, emb, emb_f32, norms, emb_normed

        # Flush and save progress periodically
        if (i + 1) % 100 == 0:
            mmap.flush()
            with open(progress_path, 'w') as f:
                json.dump({'last_completed_file': i, 'offset': offset}, f)
            logger.info(f"  {i + 1}/{len(pt_files)} files "
                        f"({offset:,}/{total:,} embeddings)")

    # Final flush
    mmap.flush()
    del mmap

    # Save shape metadata
    with open(shape_path, 'w') as f:
        json.dump({'shape': list(shape), 'dtype': 'float16', 'complete': True}, f)

    # Clean up progress file
    if os.path.exists(progress_path):
        os.remove(progress_path)

    elapsed = time.time() - t0
    logger.info(f"Normalization complete: {total:,} × {config.EMBEDDING_DIM} "
                f"float16 → {mmap_path}")
    logger.info(f"  Time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    run()
