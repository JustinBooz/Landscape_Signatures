import webdataset as wds
import json
import glob
print("Starting...")
shard_files = sorted(glob.glob("baukultur_vpr/data/shards/dataset-rectilinear-*.tar"))
print("Found", len(shard_files))
dataset = wds.WebDataset(shard_files[:1], shardshuffle=False)
dataset = dataset.decode("pil").to_tuple("img1.jpg", "meta.json")

print("Iterating...")
for x in dataset:
    print("Yielded")
    break
