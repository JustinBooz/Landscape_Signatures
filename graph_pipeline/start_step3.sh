#!/bin/bash
cd /home/jubooz/landscape_signatures/graph_pipeline

LOGFILE="pipeline_full_run.log"
echo "Starting pipeline from step 3..." > $LOGFILE

conda run -n baukultur_vpr python -u run_pipeline.py --start 3 >> $LOGFILE 2>&1
