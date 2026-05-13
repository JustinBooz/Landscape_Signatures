"""
Graph-Based Embedding Analysis Pipeline — Master Runner
=========================================================
Orchestrates all pipeline steps in sequence. Each step checks its
own checkpoints and skips if already complete. Fully resumable.

Usage:
    conda run -n baukultur_vpr python run_pipeline.py
    conda run -n baukultur_vpr python run_pipeline.py --start 3  # start from step 3
    conda run -n baukultur_vpr python run_pipeline.py --only 6   # run only step 6
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("run_pipeline",
    log_file=os.path.join(config.BASE_DIR, "pipeline.log"))


STEPS = [
    (1,  "s01_manifest",          "Manifest Construction"),
    (2,  "s02_normalize",         "Embedding Normalization"),
    (3,  "s03_pca",               "PCA Compression & Diagnostics"),
    (4,  "s04_faiss_knn",         "FAISS kNN Search"),
    (4.5,"s04b_microclusters",    "Microclustering"),
    (5,  "s05_graph",             "Graph Construction"),
    (6,  "s06_leiden",            "Leiden Community Detection"),
    (7,  "s07_stability",         "Cluster Stability Diagnostics"),
    (7.5,"s07b_cluster_confidence","Per-Cluster Confidence"),
    (8,  "s08_medoids",           "Medoid Extraction"),
    (9,  "s09_hdbscan_aux",       "Auxiliary HDBSCAN"),
    (9.5,"s09b_aux_comparators",  "Auxiliary Comparators (Microclusters)"),
    (10, "s10_umap_viz",          "UMAP Visualization"),
    (11, "s11_spatial",           "Spatial Validation"),
    (12, "s12_external",          "External Validation"),
    (13, "s13_negative_controls", "Negative Controls"),
    (14, "s14_final_assembly",    "Final Assembly & Report"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Graph-Based Embedding Analysis Pipeline")
    parser.add_argument("--start", type=float, default=1,
                        help="Start from step N (default: 1)")
    parser.add_argument("--end", type=float, default=14,
                        help="End at step N (default: 14)")
    parser.add_argument("--only", type=float, default=None,
                        help="Run only step N")
    args = parser.parse_args()

    if args.only is not None:
        args.start = args.only
        args.end = args.only

    logger.info("=" * 70)
    logger.info("  GRAPH-BASED EMBEDDING ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Output directory: {config.OUTPUT_DIR}")
    logger.info(f"  Embedding source: {config.PT_DIR}")
    logger.info(f"  Steps: {args.start} → {args.end}")
    logger.info("=" * 70)

    t_pipeline = time.time()
    step_times = {}

    for step_num, module_name, description in STEPS:
        if step_num < args.start or step_num > args.end:
            continue

        logger.info(f"\n{'=' * 70}")
        logger.info(f"  STEP {step_num:02d}: {description}")
        logger.info(f"{'=' * 70}")

        t_step = time.time()

        try:
            module = __import__(module_name)
            module.run()
        except Exception as e:
            logger.error(f"STEP {step_num} FAILED: {e}", exc_info=True)
            logger.error(f"Pipeline stopped at step {step_num}. "
                          f"Fix the issue and restart with --start {step_num}")
            sys.exit(1)

        elapsed = time.time() - t_step
        step_times[step_num] = elapsed
        logger.info(f"\n  Step {step_num} completed in {elapsed / 60:.1f} min")

    # Summary
    total_elapsed = time.time() - t_pipeline
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Total time: {total_elapsed / 3600:.1f} hours "
                f"({total_elapsed / 60:.0f} min)")
    logger.info(f"\n  Step timing:")
    for step_num, module_name, description in STEPS:
        if step_num in step_times:
            t = step_times[step_num]
            logger.info(f"    {step_num:02d}. {description:40s} {t / 60:8.1f} min")
    logger.info(f"\n  Outputs: {config.OUTPUT_DIR}")
    logger.info(f"  Master parquet: {config.master_parquet_path()}")
    logger.info(f"  Method summary: {config.method_summary_path()}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
