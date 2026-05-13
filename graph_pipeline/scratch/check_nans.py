import os
import sys
import numpy as np

sys.path.insert(0, '/home/jubooz/landscape_signatures/graph_pipeline')
import config

print("Loading mmap...")
emb_mmap = config.load_memmap(
    config.normed_embeddings_path(),
    config.normed_embeddings_shape_path(),
    dtype='float16', mode='r'
)

print(f"Shape: {emb_mmap.shape}")

# Let's check for NaNs in batches
batch_size = 50_000
n = emb_mmap.shape[0]

# Since it failed after 5,050,000, let's check the batch from 5,050,000 to 5,100,000 first
start = 5050000
end = min(start + batch_size, n)
print(f"Checking batch {start} to {end}...")
batch = emb_mmap[start:end]
nan_mask = np.isnan(batch)
inf_mask = np.isinf(batch)

if nan_mask.any():
    print(f"Found NaNs in batch {start}:{end}!")
    nan_rows = np.where(nan_mask.any(axis=1))[0]
    print(f"Rows with NaNs: {nan_rows + start}")
if inf_mask.any():
    print(f"Found Infs in batch {start}:{end}!")
    inf_rows = np.where(inf_mask.any(axis=1))[0]
    print(f"Rows with Infs: {inf_rows + start}")

print("Check finished.")
