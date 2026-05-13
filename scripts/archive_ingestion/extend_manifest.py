"""
Append new .pt file entries to the existing manifest.
No rebuild — just scans new files and extends the DataFrame.
"""

import os, sys, glob, time
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'graph_pipeline'))
import config

logger = config.setup_logging("extend_manifest",
    log_file=os.path.join(config.BASE_DIR, "extend_manifest.log"))

MIN_E, MAX_E = config.MIN_EASTING, config.MAX_EASTING
MIN_N, MAX_N = config.MIN_NORTHING, config.MAX_NORTHING


def main():
    manifest_path = config.manifest_path()
    existing = pd.read_parquet(manifest_path)
    existing_files = set(existing['embedding_file'].unique())
    global_idx = int(existing['global_index'].max()) + 1
    file_idx = int(existing['file_index'].max()) + 1

    logger.info(f"Existing manifest: {len(existing):,} rows, "
                f"{len(existing_files)} .pt files, max_index={global_idx - 1}")

    all_pt = sorted(glob.glob(os.path.join(config.PT_DIR, "*.pt")))
    new_pt = [p for p in all_pt if os.path.basename(p) not in existing_files]
    logger.info(f"New .pt files: {len(new_pt)}")

    if not new_pt:
        logger.info("Nothing to append.")
        return

    t0 = time.time()
    records = []
    for fi, pt_path in enumerate(new_pt):
        fname = os.path.basename(pt_path)
        source = 'apple-recovery' if 'apple-recovery' in fname else 'unified'
        file_number = fname.replace('.pt', '').split('-')[-1]

        d = torch.load(pt_path, map_location='cpu', weights_only=False)
        emb = d['embeddings']
        coords = d['coords'].numpy()
        norms = np.linalg.norm(emb.float().numpy(), axis=1)

        for r in range(emb.shape[0]):
            ne, nn = float(coords[r, 0]), float(coords[r, 1])
            records.append({
                'image_id': f"{source}_{file_number}_{r:06d}",
                'global_index': global_idx,
                'embedding_file': fname,
                'file_index': file_idx + fi,
                'row_index_in_file': r,
                'norm_easting': ne,
                'norm_northing': nn,
                'lv95_easting': (ne + 1) / 2 * (MAX_E - MIN_E) + MIN_E,
                'lv95_northing': (nn + 1) / 2 * (MAX_N - MIN_N) + MIN_N,
                'source_dataset': source,
                'embedding_norm': float(norms[r]),
                'is_zero_vector': bool(norms[r] < 1e-6),
                'is_low_norm': bool(norms[r] < 0.1),
                'is_duplicate_coord': False,
            })
            global_idx += 1

        del d, emb, coords, norms
        if (fi + 1) % 50 == 0:
            logger.info(f"  {fi+1}/{len(new_pt)} files, {len(records):,} rows")

    new_df = pd.DataFrame(records)
    merged = pd.concat([existing, new_df], ignore_index=True)

    # Re-check duplicate coords
    dup = merged[['norm_easting', 'norm_northing']].round(8).duplicated(keep=False)
    merged['is_duplicate_coord'] = dup

    tmp = manifest_path + ".tmp"
    merged.to_parquet(tmp, index=False)
    os.rename(tmp, manifest_path)

    elapsed = time.time() - t0
    logger.info(f"Manifest extended: {len(existing):,} → {len(merged):,} "
                f"(+{len(new_df):,}) in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
