#!/bin/bash
# Stage 3: Temporal Motion Analysis — nohup launcher for 4x H100 DDP training
#
# Usage:
#   # Foreground:
#   bash run_stage3.sh
#
#   # Background (survives logout):
#   nohup bash run_stage3.sh > /data/code/runs/stage3/stdout.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="/data/code/runs/stage3"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "[$(date)] Starting Stage 3: Temporal Motion training on 4 GPUs"
echo "Output: $OUTPUT_DIR"
echo "Logs: $OUTPUT_DIR/train.log"
echo "Metrics: $OUTPUT_DIR/metrics.csv"

nohup torchrun \
    --nproc_per_node=4 \
    --master_port=29502 \
    train_stage3.py \
    > "$OUTPUT_DIR/stdout.log" 2>&1 &

PID=$!
echo $PID > "$OUTPUT_DIR/train.pid"
echo "[$(date)] Training launched with PID $PID"
echo "Monitor:  tail -f $OUTPUT_DIR/train.log"
echo "Progress: tail -f $OUTPUT_DIR/metrics.csv"
echo "Stop:     kill $PID"
