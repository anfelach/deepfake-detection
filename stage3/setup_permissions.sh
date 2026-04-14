#!/bin/bash
###############################################################
# PERMISSIONS SETUP FOR STAGE 3
# Run this as Sarra's user (w_sarra_arab_udst_edu_qa)
#
# This gives Anfal write access to the processed/ directory
# so that the optical flow preprocessing can save flow_haar_hh.npy
# files alongside the existing .npy data.
#
# Also creates the Stage 3 output directory on /data for
# training logs, checkpoints, and metrics.
#
# Usage:
#   bash /home/w_anfal_achouri_udst_edu_qa/stage3/setup_permissions.sh
#
# After running, Anfal can launch:
#   bash ~/stage3/run_flow_preprocess.sh
#   bash ~/stage3/run_stage3.sh
###############################################################

set -e

echo "============================================================"
echo "STAGE 3 PERMISSIONS SETUP"
echo "Running as: $(whoami)"
echo "Date: $(date -Iseconds)"
echo "============================================================"
echo ""

# 1. Grant write access to processed/ so flow_haar_hh.npy can be saved
echo "[1/3] Granting write access to /data/deepfake_pipeline/processed/ ..."
chmod -R o+w /data/deepfake_pipeline/processed/
echo "      Done. Anfal can now write flow_haar_hh.npy files."

# 2. Create Stage 3 output directory on /data (for training runs)
echo "[2/3] Creating /data/code/runs/stage3/ ..."
mkdir -p /data/code/runs/stage3/checkpoints
chmod -R o+w /data/code/runs/stage3/
echo "      Done. Training logs/checkpoints will go here."

# 3. Verify
echo "[3/3] Verifying permissions ..."
echo ""

# Test write to processed/
TEST_DIR="/data/deepfake_pipeline/processed"
if [ -w "$TEST_DIR" ]; then
    echo "  [OK] $TEST_DIR is writable"
else
    echo "  [FAIL] $TEST_DIR is NOT writable"
fi

# Test write to runs/stage3
TEST_DIR2="/data/code/runs/stage3"
if [ -w "$TEST_DIR2" ]; then
    echo "  [OK] $TEST_DIR2 is writable"
else
    echo "  [FAIL] $TEST_DIR2 is NOT writable"
fi

echo ""
echo "============================================================"
echo "DONE. Anfal can now run:"
echo "  1. bash ~/stage3/run_flow_preprocess.sh   (preprocessing)"
echo "  2. bash ~/stage3/run_stage3.sh            (training)"
echo "============================================================"
