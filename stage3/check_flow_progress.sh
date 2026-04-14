#!/bin/bash
###############################################################
# Quick progress checker for optical flow preprocessing
# Usage: bash check_flow_progress.sh
###############################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"

echo "============================================================"
echo "OPTICAL FLOW PREPROCESSING PROGRESS — $(date)"
echo "============================================================"
echo ""

# Check if any flow compute processes are running
RUNNING=0
for pid_file in "$LOGS_DIR"/flow_gpu*.pid 2>/dev/null; do
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        GPU=$(basename "$pid_file" | grep -oP 'gpu\K[0-9]+')
        if kill -0 $PID 2>/dev/null; then
            echo "GPU $GPU: RUNNING (PID=$PID)"
            RUNNING=$((RUNNING+1))
        else
            echo "GPU $GPU: PID file exists but process not running"
        fi
    fi
done
if [ $RUNNING -eq 0 ]; then
    echo "STATUS: No running processes detected"
fi
echo ""

# Show progress CSV
if [ -f "$LOGS_DIR/flow_progress.csv" ]; then
    echo "--- PROGRESS SUMMARY ---"
    column -t -s, "$LOGS_DIR/flow_progress.csv"
    echo ""
fi

# Per-GPU tracker summary
echo "--- PER-GPU TRACKER SUMMARY ---"
for tracker in "$LOGS_DIR"/flow_tracker_gpu*.csv 2>/dev/null; do
    if [ -f "$tracker" ]; then
        GPU=$(basename "$tracker" | grep -oP 'gpu\K[0-9]+')
        TOTAL=$(tail -n +2 "$tracker" | wc -l)
        DONE=$(grep -c ',done,' "$tracker" 2>/dev/null || echo 0)
        FAILED=$(grep -c ',failed,' "$tracker" 2>/dev/null || echo 0)
        SKIPPED=$(grep -c ',skipped,' "$tracker" 2>/dev/null || echo 0)
        echo "GPU $GPU: Total=$TOTAL | Done=$DONE | Failed=$FAILED | Skipped=$SKIPPED"
    fi
done
echo ""

# Count flow_haar_hh.npy files on disk
echo "--- OUTPUT FILES ON DISK ---"
FLOW_COUNT=$(find /data/deepfake_pipeline/processed -name "flow_haar_hh.npy" 2>/dev/null | wc -l)
TOTAL_VIDEOS=$(find /data/deepfake_pipeline/processed -name "frames_rgb.npy" 2>/dev/null | wc -l)
echo "flow_haar_hh.npy files: $FLOW_COUNT / $TOTAL_VIDEOS total videos"
echo ""

# Disk usage
echo "--- DISK USAGE ---"
du -sh /data/deepfake_pipeline/processed/ 2>/dev/null || echo "Processed: N/A"
df -h /data | tail -1
echo ""

# Last 5 log lines per GPU
echo "--- LAST 5 LOG LINES ---"
for logf in "$LOGS_DIR"/compute_flow_gpu*.log 2>/dev/null; do
    if [ -f "$logf" ]; then
        GPU=$(basename "$logf" | grep -oP 'gpu\K[0-9]+')
        echo "  [GPU $GPU]"
        tail -5 "$logf" | sed 's/^/    /'
        echo ""
    fi
done
