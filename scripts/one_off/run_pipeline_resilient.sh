#!/bin/bash
# Resilient three-phase pipeline runner
# Phase 1: Parallel Leiden clustering (leiden_parallel.py)
# Phase 2: Pipeline steps 7-14 (run_pipeline.py --start 7)
# Auto-restarts on crash. All checkpointing is handled by the scripts.

MAX_RETRIES=10
RETRY_DELAY=30
LOGFILE="/home/jubooz/landscape_signatures/graph_pipeline/pipeline_resilient.log"
CD="/home/jubooz/landscape_signatures/graph_pipeline"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [RESILIENT] $1" | tee -a "$LOGFILE"; }

# Phase 1: Parallel Leiden
log "Phase 1: Parallel Leiden clustering (all 5×5)"
for attempt in $(seq 1 $MAX_RETRIES); do
    log "Leiden attempt $attempt/$MAX_RETRIES"
    cd "$CD" && conda run -n baukultur_vpr python -u leiden_parallel.py 2>&1 | tee -a "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        log "Phase 1 complete!"
        break
    fi

    log "Crashed (exit=$EXIT_CODE). Retrying in ${RETRY_DELAY}s..."
    sync
    sleep $RETRY_DELAY
done

# Phase 2: Steps 7-14
log "Phase 2: Pipeline steps 7-14"
for attempt in $(seq 1 $MAX_RETRIES); do
    log "Steps 7-14 attempt $attempt/$MAX_RETRIES"
    cd "$CD" && conda run -n baukultur_vpr python -u run_pipeline.py --start 7 2>&1 | tee -a "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        log "Pipeline complete!"
        exit 0
    fi

    log "Crashed (exit=$EXIT_CODE). Retrying in ${RETRY_DELAY}s..."
    sync
    sleep $RETRY_DELAY
done

log "Exhausted retries."
exit 1
