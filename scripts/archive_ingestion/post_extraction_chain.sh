#!/bin/bash
# Waits for extraction to finish, then runs manifest extension + geodata enrichment.
# Usage: nohup bash post_extraction_chain.sh > chain.log 2>&1 &

set -e
cd /home/jubooz/landscape_signatures
EXTRACT_PID=603376  # main python process

echo "$(date) — Waiting for extraction (PID $EXTRACT_PID) to finish..."
while kill -0 $EXTRACT_PID 2>/dev/null; do
    sleep 60
done
echo "$(date) — Extraction finished."

echo ""
echo "$(date) — Step 1: Extending manifest..."
conda run -n baukultur_vpr python -u extend_manifest.py
echo "$(date) — Manifest extended."

echo ""
echo "$(date) — Step 2: Geodata enrichment v2..."
conda run -n baukultur_vpr python -u graph_pipeline/geodata_enrich_v2.py
echo "$(date) — Geodata enrichment done."

echo ""
echo "$(date) — Step 3: DEM fallback..."
conda run -n baukultur_vpr python -u graph_pipeline/fix_terrain_srtm.py
echo "$(date) — DEM fallback done."

echo ""
echo "$(date) — All post-extraction steps complete."
echo "$(date) — Next: run s02 → s06 manually, or continue pipeline."
