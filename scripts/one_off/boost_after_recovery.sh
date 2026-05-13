#!/usr/bin/env bash
# Watchdog: waits for Apple recovery to finish, then boosts main ingestion concurrency.
set -e

RECOVERY_SCRIPT="recover_apple_images.py"
INGESTION_SCRIPT="baukultur_vpr/data/unified_ingestion.py"
PYTHON="/home/jubooz/anaconda3/envs/baukultur_vpr/bin/python"
LOG="unified_ingestion.log"

echo "[$(date)] Watchdog started. Monitoring recovery process..."

# Wait for recovery to finish
while pgrep -f "$RECOVERY_SCRIPT" > /dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] Apple recovery process has exited."

# Apply tuning changes to unified_ingestion.py
echo "[$(date)] Applying concurrency boost to main ingestion..."

# Bump max_concurrent_cells from 24 to 40
sed -i 's/max_concurrent_cells = 24/max_concurrent_cells = 40/' "$INGESTION_SCRIPT"

# Bump api_sem from 150 to 300
sed -i 's/asyncio.Semaphore(150)/asyncio.Semaphore(300)/' "$INGESTION_SCRIPT"

# Bump dl_sem from 400 to 600
sed -i 's/asyncio.Semaphore(400)/asyncio.Semaphore(600)/' "$INGESTION_SCRIPT"

echo "[$(date)] Changes applied. Restarting main ingestion..."

# Kill existing main ingestion
pkill -f "$INGESTION_SCRIPT" || true
sleep 5

# Restart with boosted settings
nohup $PYTHON $INGESTION_SCRIPT >> "$LOG" 2>&1 &

echo "[$(date)] Main ingestion restarted with boosted concurrency. PID: $!"
echo "[$(date)] Watchdog complete."
