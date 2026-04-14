#!/bin/bash
###############################################################
# Launch RAFT optical flow preprocessing on all 4 GPUs
# Fully resumable — safe to kill and restart.
#
# Usage:
#   # Foreground (interactive):
#   bash run_flow_preprocess.sh
#
#   # Background (survives logout):
#   nohup bash run_flow_preprocess.sh > ~/stage3/logs/flow_master.log 2>&1 &
#
#   # Check progress:
#   bash check_flow_progress.sh
#   tail -f ~/stage3/logs/compute_flow_gpu0.log
#   cat ~/stage3/logs/flow_progress.csv
###############################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

NUM_GPUS=4

echo "============================================================"
echo "STAGE 3 PREPROCESSING: RAFT OPTICAL FLOW + HAAR HH"
echo "Started: $(date -Iseconds)"
echo "GPUs: $NUM_GPUS"
echo "Logs: $LOGS_DIR/"
echo "============================================================"
echo ""

# Launch one process per GPU
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    echo "[$(date -Iseconds)] Launching GPU $GPU_ID..."

    python3 "$SCRIPT_DIR/compute_optical_flow.py" \
        --num_gpus $NUM_GPUS \
        --gpu_id $GPU_ID \
        --max_frames 64 \
        > "$LOGS_DIR/compute_flow_gpu${GPU_ID}_stdout.log" 2>&1 &

    PID=$!
    echo $PID > "$LOGS_DIR/flow_gpu${GPU_ID}.pid"
    echo "[$(date -Iseconds)] GPU $GPU_ID launched with PID $PID"
done

echo ""
echo "All $NUM_GPUS processes launched."
echo "Monitor:   bash $SCRIPT_DIR/check_flow_progress.sh"
echo "Logs:      tail -f $LOGS_DIR/compute_flow_gpu0.log"
echo "Progress:  cat $LOGS_DIR/flow_progress.csv"
echo ""

# Wait for all to finish
echo "Waiting for all processes to complete..."
wait
echo ""
echo "[$(date -Iseconds)] All processes finished."
echo "Run 'bash $SCRIPT_DIR/check_flow_progress.sh' for final summary."
