"""
Step 12: External Validation
================================
Stub for external metadata comparison. Checks for external datasets
and computes enrichment if found.

Output: external_validation_summary.csv
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = config.setup_logging("s12_external")


def run():
    logger.info("=" * 60)
    logger.info("STEP 12: External Validation")
    logger.info("=" * 60)

    output_path = config.external_validation_path()
    if os.path.exists(output_path):
        logger.info(f"[CHECKPOINT] External validation exists: {output_path}")
        return

    # Check for external data files
    # These would be things like land use maps, elevation data, etc.
    external_data_dir = os.path.join(config.BASE_DIR, "external_data")
    external_files = []

    if os.path.exists(external_data_dir):
        for f in os.listdir(external_data_dir):
            if f.endswith(('.csv', '.parquet', '.gpkg', '.shp', '.geojson')):
                external_files.append(os.path.join(external_data_dir, f))

    if not external_files:
        logger.warning("STEP 12 IS A STUB: external validation is not implemented.")
        logger.warning("No external datasets present in %s/. To enable, drop "
                        "files (.csv, .parquet, .gpkg, .shp, .geojson) there and "
                        "implement the spatial join + enrichment logic.",
                        external_data_dir)

        df = pd.DataFrame([{
            'status': 'stub_not_implemented',
            'message': ('External validation is not implemented. No external '
                        'data was found and no analysis was performed.'),
        }])
        df.to_csv(output_path, index=False)
        return

    logger.warning("STEP 12 IS A STUB: %d external datasets found but the "
                    "enrichment analysis is not implemented.", len(external_files))
    for f in external_files:
        logger.warning(f"  found: {f}")

    df = pd.DataFrame([{
        'status': 'stub_not_implemented_with_data_present',
        'n_files': len(external_files),
        'files': ';'.join(external_files),
    }])
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    run()
