import sys
import numpy as np
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, '/home/jubooz/landscape_signatures/graph_pipeline')
import config

print("Loading mmap...")
emb_mmap = config.load_memmap(
    config.normed_embeddings_path(),
    config.normed_embeddings_shape_path(),
    dtype='float16', mode='r'
)

start = 5050000
end = min(start + 50000, emb_mmap.shape[0])
print(f"Loading batch {start}:{end}...")
batch = emb_mmap[start:end].astype(np.float32)

print("Initializing IncrementalPCA-512...")
ipca = IncrementalPCA(n_components=512)

print("Running partial_fit...")
try:
    ipca.partial_fit(batch)
    print("partial_fit succeeded!")
except Exception as e:
    print(f"partial_fit failed: {e}")
