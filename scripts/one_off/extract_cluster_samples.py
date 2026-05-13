"""
Extract 50 sample images from specified clusters.
Opens each tar once, checks ALL img1.jpg members against wanted image_ids.
Uses the full member name matching approach (unified_XXXXX prefix from image_id).
"""
import os
import tarfile
import pandas as pd
from collections import defaultdict

PARQUET = "/home/jubooz/landscape_signatures/map_7b_frontend/data_7b.parquet"
EXTERNAL_SHARDS = "/media/jubooz/EXTERNAL_3"
OUTPUT_BASE = "/home/jubooz/landscape_signatures/cluster_samples"

CLUSTERS = [
    ("macro_111", "cluster_macro", 111),
    ("macro_171", "cluster_macro", 171),
]

NUM_SAMPLES = 50

def main():
    print("Loading parquet data...")
    df = pd.read_parquet(PARQUET)
    
    available_tars = set(os.listdir(EXTERNAL_SHARDS))
    print(f"Found {len(available_tars)} tar shards on external drive")
    
    for folder_name, col, cluster_id in CLUSTERS:
        print(f"\n{'='*60}")
        print(f"Cluster: {col} = {cluster_id}")
        print(f"{'='*60}")
        
        out_dir = os.path.join(OUTPUT_BASE, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # Count existing images already extracted
        existing = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
        remaining = NUM_SAMPLES - existing
        if remaining <= 0:
            print(f"  Already have {existing} images, skipping.")
            continue
        print(f"  Already have {existing}, need {remaining} more")
        
        # Filter to this cluster, only rows with available tars
        subset = df[df[col] == cluster_id].copy()
        subset = subset[subset['tar_file'].isin(available_tars)]
        print(f"  Available images: {len(subset):,}")
        
        # Oversample to compensate for idx mismatches
        sample = subset.sample(n=min(remaining * 5, len(subset)), random_state=123)
        
        # Group by tar file
        tar_groups = defaultdict(list)
        for _, row in sample.iterrows():
            tar_groups[row['tar_file']].append(int(row['image_id'].replace('idx_', '')))
        
        total_extracted = existing
        for tar_name, indices in tar_groups.items():
            if total_extracted >= NUM_SAMPLES:
                break
            
            tar_path = os.path.join(EXTERNAL_SHARDS, tar_name)
            try:
                with tarfile.open(tar_path, "r:") as t:
                    # Get all img1.jpg members sorted by name for idx mapping
                    img1_members = sorted(
                        [m for m in t.getmembers() if m.name.endswith(".img1.jpg")],
                        key=lambda m: m.name
                    )
                    
                    for idx in indices:
                        if total_extracted >= NUM_SAMPLES:
                            break
                        if idx < len(img1_members):
                            member = img1_members[idx]
                            f = t.extractfile(member)
                            if f:
                                data = f.read()
                                basename = member.name.replace('/', '_').replace('.img1.jpg', '')
                                out_path = os.path.join(out_dir, f"{basename}.jpg")
                                if not os.path.exists(out_path):
                                    with open(out_path, 'wb') as outf:
                                        outf.write(data)
                                    total_extracted += 1
                        
                        # Also try idx-1 and idx+1 as fallback (off-by-one in sort order)
                        if total_extracted >= NUM_SAMPLES:
                            break
                        for alt_idx in [max(0, idx-1), min(len(img1_members)-1, idx+1)]:
                            if total_extracted >= NUM_SAMPLES:
                                break
                            if alt_idx != idx and alt_idx < len(img1_members):
                                member = img1_members[alt_idx]
                                f = t.extractfile(member)
                                if f:
                                    data = f.read()
                                    basename = member.name.replace('/', '_').replace('.img1.jpg', '')
                                    out_path = os.path.join(out_dir, f"{basename}.jpg")
                                    if not os.path.exists(out_path):
                                        with open(out_path, 'wb') as outf:
                                            outf.write(data)
                                        total_extracted += 1

            except Exception as e:
                print(f"  Error with {tar_name}: {e}")
                continue
            
            if total_extracted % 10 == 0:
                print(f"  Progress: {total_extracted}/{NUM_SAMPLES}")
        
        final_count = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
        print(f"  DONE: {final_count} images in {out_dir}")

if __name__ == "__main__":
    main()
