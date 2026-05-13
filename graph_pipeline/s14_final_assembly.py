"""
Step 14: Final Assembly & Report
===================================
Merges all outputs into a master parquet and creates the method summary.
Generates cluster_signatures_summary.parquet and Candidate Signature flags.

Outputs:
  - landscape_embedding_analysis_master.parquet
  - cluster_signatures_summary.parquet
  - reports/README_method_summary.md
"""

import os
import sys
import shutil
import time
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from leiden_parallel import REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED

logger = config.setup_logging("s14_final_assembly")


def _require_columns(df, required, source_name):
    """Assert that a dataframe has the columns we're about to read. Catches
    the column-rename-but-forgot-to-update-readers class of bug at the first
    failing step rather than producing silently-wrong output."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{source_name}: missing required columns {missing}. "
                       f"Have: {list(df.columns)}")


def _load_yaml_config():
    path = os.path.join(config.BASE_DIR, "..", "config", "candidate_signature_criteria.yaml")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return None

def _build_master_parquet():
    """Merge manifest + Leiden + microclusters + HDBSCAN/FINCH + metrics."""
    output_path = config.master_parquet_path()
    
    logger.info("Building master parquet...")
    df = pd.read_parquet(config.manifest_path())
    logger.info(f"  Manifest: {len(df):,} rows")

    # Primary clustering config
    ref_label_path = config.leiden_labels_path(REF_PCA, REF_K, REF_GRAPH, REF_RES, REF_SEED)
    if os.path.exists(ref_label_path):
        df['primary_leiden_cluster_id'] = np.load(ref_label_path)
        df['primary_clustering_config'] = f"pca{REF_PCA}_k{REF_K}_{REF_GRAPH}_res{REF_RES}_seed{REF_SEED}"
    else:
        logger.error("Primary clustering not found!")
        return None, None

    # Microclusters
    mc_path = config.microcluster_assignments_path()
    if os.path.exists(mc_path):
        mc_df = pd.read_parquet(mc_path, columns=['global_index', 'microcluster_id'])
        _require_columns(mc_df, ['global_index', 'microcluster_id'], mc_path)
        df = df.merge(mc_df, on='global_index', how='left')

    # Confidence scores (cluster level -> broadcast to points)
    conf_path = config.cluster_confidence_path()
    if os.path.exists(conf_path):
        conf_df = pd.read_parquet(conf_path)
        _require_columns(conf_df, ['cluster_id', 'overall_cluster_confidence',
                                    'seed_stability', 'k_stability',
                                    'graph_stability', 'resolution_stability'],
                          conf_path)
        df = df.merge(conf_df, left_on='primary_leiden_cluster_id', right_on='cluster_id', how='left')
        df.drop(columns=['cluster_id'], inplace=True)

    # Auxiliary comparators
    # s09b writes hdbscan_label (microcluster-lifted); s09 writes per-setting cols
    hdb_path = config.hdbscan_labels_path()
    if os.path.exists(hdb_path):
        hdb_df = pd.read_parquet(hdb_path)
        if 'hdbscan_label' in hdb_df.columns:
            df['hdbscan_microcluster_label'] = hdb_df['hdbscan_label'].values
        else:
            logger.warning(f"  {hdb_path} missing 'hdbscan_label' column")
    else:
        df['hdbscan_microcluster_label'] = np.nan

    hdb_pw_path = config.hdbscan_pointwise_labels_path()
    if os.path.exists(hdb_pw_path):
        hdb_pw = pd.read_parquet(hdb_pw_path)
        # carry over all setting-specific columns
        for col in hdb_pw.columns:
            if col.startswith('hdbscan_mcs'):
                df[col] = hdb_pw[col].values

    finch_path = config.finch_labels_path()
    if os.path.exists(finch_path):
        finch_df = pd.read_parquet(finch_path)
        for col in finch_df.columns:
            df[col] = finch_df[col].values

    # Spatial / Geodata Validation
    spatial_path = config.cluster_spatial_validation_path()
    if os.path.exists(spatial_path):
        spatial_df = pd.read_parquet(spatial_path, columns=['cluster_id', 'spatial_coherence_score', 'geodata_purity_score', 'geodata_entropy'])
        df = df.merge(spatial_df, left_on='primary_leiden_cluster_id', right_on='cluster_id', how='left')
        df.drop(columns=['cluster_id'], inplace=True)

    # Negative Controls
    nc_path = config.negative_control_pvalues_path()
    if os.path.exists(nc_path):
        nc_df = pd.read_parquet(nc_path, columns=['cluster_id', 'negative_control_pvalue'])
        df = df.merge(nc_df, left_on='primary_leiden_cluster_id', right_on='cluster_id', how='left')
        df.drop(columns=['cluster_id'], inplace=True)

    # Medoids
    med_path = config.medoids_path(REF_RES)
    df['medoid_flag'] = False
    if os.path.exists(med_path):
        med_df = pd.read_parquet(med_path)
        if 'medoid_global_index' in med_df.columns:
            med_indices = med_df['medoid_global_index'].values
            df.loc[df['global_index'].isin(med_indices), 'medoid_flag'] = True

    # Candidate Signature Flag
    yaml_config = _load_yaml_config()
    candidate_cluster_ids = set()
    
    if yaml_config and 'candidate_signature_thresholds' in yaml_config:
        thresh = yaml_config['candidate_signature_thresholds']
        # Combine metrics to find candidates
        # We need cluster-level metrics
        if os.path.exists(conf_path) and os.path.exists(spatial_path) and os.path.exists(nc_path):
            summary_df = conf_df.merge(spatial_df, on='cluster_id').merge(nc_df, on='cluster_id')
            
            # Check thresholds
            # Note: TBD thresholds might be strings "TBD". If so, we bypass them or they fail.
            try:
                coh_min = float(thresh.get('spatial_coherence_min', 0))
            except ValueError:
                coh_min = 1e9 # TBD -> bypass
                
            try:
                pur_min = float(thresh.get('geodata_purity_min', 0))
            except ValueError:
                pur_min = -1.0 # TBD -> bypass

            mask = (
                (summary_df['seed_stability'] >= thresh.get('seed_stability_min', 0)) &
                (summary_df['k_stability'] >= thresh.get('k_stability_min', 0)) &
                (summary_df['graph_stability'] >= thresh.get('graph_stability_min', 0)) &
                (summary_df['resolution_stability'] >= thresh.get('resolution_stability_min', 0)) &
                (summary_df['negative_control_pvalue'] <= thresh.get('negative_control_p_max', 1.0))
            )
            
            # Only apply spatial/geodata thresholds if they are not TBD
            if coh_min != 1e9:
                mask = mask & (summary_df['spatial_coherence_score'] <= coh_min)
            if pur_min != -1.0:
                mask = mask & (summary_df['geodata_purity_score'] >= pur_min)
                
            candidate_cluster_ids = set(summary_df[mask]['cluster_id'].values)
            
            summary_df['candidate_signature_flag'] = summary_df['cluster_id'].isin(candidate_cluster_ids)
            
            # Overall score = significance-weighted quality.
            # -log10(p) weights how unlikely the score is under the null
            # (capped to avoid divide-by-zero), times the additive quality of
            # the cluster (confidence + purity). Both quality terms are in
            # [0, 1], so the score is bounded by [0, 2 * (-log10(1/(n+1)))].
            p = summary_df['negative_control_pvalue'].clip(lower=1e-6)
            quality = (summary_df['overall_cluster_confidence']
                        + summary_df['geodata_purity_score'])
            summary_df['overall_candidate_signature_score'] = (-np.log10(p)) * quality
            
            # Add medoids
            if os.path.exists(med_path) and 'medoid_global_index' in med_df.columns:
                summary_df = summary_df.merge(
                    med_df[['cluster_id', 'medoid_global_index']],
                    on='cluster_id', how='left'
                )
                
            # Sizes
            sizes = df['primary_leiden_cluster_id'].value_counts().reset_index()
            sizes.columns = ['cluster_id', 'cluster_size']
            summary_df = summary_df.merge(sizes, on='cluster_id', how='left')
            
            summary_df['method'] = 'Leiden'
            summary_df['config'] = df['primary_clustering_config'].iloc[0]
            
            summ_out = config.cluster_signatures_summary_path()
            summary_df.to_parquet(summ_out, index=False)
            logger.info(f"Saved cluster signatures summary: {summ_out}")
            logger.info(f"  Found {len(candidate_cluster_ids)} Candidate Signatures.")

    df['candidate_signature_flag'] = df['primary_leiden_cluster_id'].isin(candidate_cluster_ids)

    # Save
    df.to_parquet(output_path + ".tmp", index=False)
    os.rename(output_path + ".tmp", output_path)
    logger.info(f"\nMaster parquet saved: {output_path}")
    logger.info(f"  {len(df):,} rows, {len(df.columns)} columns")
    
    return df, yaml_config


def _write_method_summary(yaml_config):
    """Write the method summary README according to requested reporting language."""
    output_path = config.method_summary_path()

    content = """# Method Summary: Graph-Based Visual Landscape Embedding Analysis

## Overview

DINOv3 ViT-7B16 embeddings were treated as a high-dimensional visual similarity
space for street-level landscape images. After L2 normalization and compression
diagnostics, we constructed approximate nearest-neighbor graphs using cosine
similarity.

Leiden identifies visual-embedding communities. Communities that remain stable 
across graph construction choices, clustering parameters, auxiliary methods, 
spatial validation, geodata validation, and negative controls are treated as 
**candidate landscape signatures**. The robustness checks in this pipeline establish 
that, conditional on the DINOv3 embedding, the identified communities are not 
artifacts of clustering parameters. Validation that the embedding itself corresponds 
to human perceptual structure is addressed in WP2 (gaze tracking).

## Pre-Registered Candidate Criteria

The pipeline automatically applied the following robustness thresholds:

```yaml
"""
    if yaml_config:
        content += yaml.dump(yaml_config, default_flow_style=False)
    else:
        content += "No YAML config found.\n"
        
    content += """```

## Pipeline Extensions

1. **Microclustering & Auxiliary Comparators**: We computed 50,000 spherical k-means microclusters and optionally applied HDBSCAN and FINCH.
2. **Graph Variants**: SNN Jaccard and Mutual SNN Jaccard were evaluated to ensure topological robustness.
3. **Fractional Grid**: A targeted 13-run Leiden grid perturbed seeds, k, graphs, and resolutions around the reference configuration.
4. **Spatial Null Models**: Geodata purity and spatial coherence were scored via empirical p-values against 100 random label permutations.

"""

    with open(output_path, 'w') as f:
        f.write(content)
    logger.info(f"Method summary written: {output_path}")


def run():
    logger.info("=" * 60)
    logger.info("STEP 14: Final Assembly & Report")
    logger.info("=" * 60)

    t0 = time.time()

    df, yaml_config = _build_master_parquet()
    if df is not None:
        _write_method_summary(yaml_config)

    elapsed = time.time() - t0
    logger.info(f"\nFinal assembly complete in {elapsed / 60:.1f} min")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
