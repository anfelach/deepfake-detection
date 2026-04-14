#!/usr/bin/env python3
"""
Preprocessing for Stage 3: Compute RAFT optical flow + Haar wavelet HH on flow magnitudes.
Saves flow_haar_hh.npy alongside existing .npy files for each video.

Features:
  - RESUMABLE: Tracks every processed video in CSV; skips already-processed
  - LOGGING: Full log file with timestamps (same pattern as step2_extract_wavelets.py)
  - VERBOSE: Per-video status logged
  - BACKGROUND-SAFE: CSV checkpoints, safe to kill and restart

Output per video:
  {data_dir}/flow_haar_hh.npy    (N-1, 256, 256) float32

Usage:
    # Single GPU:
    python compute_optical_flow.py

    # Specific GPU:
    python compute_optical_flow.py --device cuda:1

    # Multi-GPU parallel (launch one process per GPU):
    python compute_optical_flow.py --num_gpus 4 --gpu_id 0 &
    python compute_optical_flow.py --num_gpus 4 --gpu_id 1 &
    python compute_optical_flow.py --num_gpus 4 --gpu_id 2 &
    python compute_optical_flow.py --num_gpus 4 --gpu_id 3 &

    # Check progress:
    bash check_flow_progress.sh
    tail -f logs/compute_flow.log
    cat logs/flow_progress.csv
"""
import os
import sys
import argparse
import csv
import time
import logging
import numpy as np
import torch
import pywt
from datetime import datetime
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

MANIFESTS = [
    "/data/deepfake_pipeline/splits/manifest_train.csv",
    "/data/deepfake_pipeline/splits/manifest_val.csv",
    "/data/deepfake_pipeline/splits/manifest_test.csv",
]


# ============================================================
# LOGGING (same pattern as step2_extract_wavelets.py)
# ============================================================
def setup_logger(gpu_id):
    """Create logger that writes to both file and stdout."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f"compute_flow_gpu{gpu_id}.log")

    logger = logging.getLogger(f"flow_gpu{gpu_id}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# TRACKER CSV — per-video log (resume from here)
# ============================================================
TRACKER_FIELDS = [
    "video_dir", "generator", "video_id", "n_flow_frames",
    "status", "timestamp", "duration_sec", "gpu_id", "error",
]


def get_tracker_path(gpu_id):
    return os.path.join(LOGS_DIR, f"flow_tracker_gpu{gpu_id}.csv")


def load_tracker(gpu_id):
    """Load set of already-processed video dirs from tracker CSV."""
    completed = set()
    tracker_path = get_tracker_path(gpu_id)
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'done':
                    completed.add(row['video_dir'])
    return completed


def append_tracker(gpu_id, row_dict):
    """Append a row to the tracker CSV."""
    tracker_path = get_tracker_path(gpu_id)
    write_header = not os.path.exists(tracker_path)
    with open(tracker_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


# ============================================================
# PROGRESS CSV — summary across all generators
# ============================================================
PROGRESS_FIELDS = [
    "gpu_id", "total_videos", "processed", "skipped_done",
    "skipped_other", "failed", "total_flow_frames",
    "status", "start_time", "end_time", "duration_sec",
]


def write_progress(gpu_id, stats):
    """Write/update progress summary for this GPU."""
    progress_path = os.path.join(LOGS_DIR, "flow_progress.csv")
    rows = []
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('gpu_id')) != str(gpu_id):
                    rows.append(row)
    rows.append(stats)
    with open(progress_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MANIFEST
# ============================================================
def load_manifest(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


# ============================================================
# CORE: RAFT FLOW + HAAR HH
# ============================================================
def compute_flow_for_video(video_dir, raft_model, device, max_frames=64):
    """
    Compute optical flow + Haar HH wavelet for one video.

    Steps:
        1. Load frames_rgb.npy
        2. Run RAFT on consecutive frame pairs → flow magnitude
        3. Apply Haar wavelet decomposition → HH subband
        4. Save flow_haar_hh.npy

    Returns: (n_flow_frames, status_message)
    """
    rgb_path = os.path.join(video_dir, 'frames_rgb.npy')
    output_path = os.path.join(video_dir, 'flow_haar_hh.npy')

    if not os.path.exists(rgb_path):
        return 0, "missing_rgb"

    # Load RGB frames: (N, 512, 512, 3) uint8
    rgb = np.load(rgb_path, mmap_mode='r')
    n_frames = min(rgb.shape[0], max_frames)

    if n_frames < 2:
        return 0, "too_few_frames"

    n_flow = n_frames - 1
    haar_hh_list = []

    # Process frame pairs in small batches for GPU efficiency
    batch_size = 8
    for start in range(0, n_flow, batch_size):
        end = min(start + batch_size, n_flow)

        # Prepare frame pairs
        frames1 = []
        frames2 = []
        for i in range(start, end):
            f1 = rgb[i].copy().astype(np.float32)      # (512, 512, 3)
            f2 = rgb[i + 1].copy().astype(np.float32)   # (512, 512, 3)

            # RAFT expects (B, 3, H, W) in [0, 255] range
            f1 = torch.from_numpy(f1).permute(2, 0, 1)  # (3, 512, 512)
            f2 = torch.from_numpy(f2).permute(2, 0, 1)  # (3, 512, 512)
            frames1.append(f1)
            frames2.append(f2)

        batch1 = torch.stack(frames1).to(device)  # (batch, 3, 512, 512)
        batch2 = torch.stack(frames2).to(device)  # (batch, 3, 512, 512)

        # RAFT inference
        with torch.no_grad():
            flow_predictions = raft_model(batch1, batch2)
            flow = flow_predictions[-1]  # (batch, 2, 512, 512)

        # Compute flow magnitude and Haar HH on CPU
        flow_np = flow.cpu().numpy()  # (batch, 2, 512, 512)
        for j in range(flow_np.shape[0]):
            dx = flow_np[j, 0]  # (512, 512)
            dy = flow_np[j, 1]  # (512, 512)
            magnitude = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)

            # Haar wavelet decomposition → HH subband (diagonal high-frequency)
            coeffs = pywt.dwt2(magnitude, 'haar')
            _, (_, _, hh) = coeffs  # HH: (256, 256) float32
            haar_hh_list.append(hh)

    # Stack and save
    flow_haar_hh = np.stack(haar_hh_list, axis=0)  # (n_flow, 256, 256)
    np.save(output_path, flow_haar_hh)

    return n_flow, "done"


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute RAFT optical flow + Haar HH for deepfake videos"
    )
    parser.add_argument('--manifests', nargs='+', default=MANIFESTS,
                        help="Manifest CSV files to process")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU device")
    parser.add_argument('--num_gpus', type=int, default=1,
                        help="Total GPUs for parallel processing")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="This GPU's index (for round-robin assignment)")
    parser.add_argument('--max_frames', type=int, default=64,
                        help="Max RGB frames to use per video")
    args = parser.parse_args()

    if args.num_gpus > 1:
        device = f'cuda:{args.gpu_id}'
    else:
        device = args.device

    # Setup logging
    logger = setup_logger(args.gpu_id)

    logger.info("#" * 60)
    logger.info("STAGE 3 PREPROCESSING: RAFT OPTICAL FLOW + HAAR HH")
    logger.info(f"Started at {datetime.now().isoformat()}")
    logger.info(f"GPU: {args.gpu_id} | Device: {device} | Max frames: {args.max_frames}")
    logger.info(f"Num GPUs: {args.num_gpus}")
    logger.info("#" * 60)

    # Load tracker for resume
    completed = load_tracker(args.gpu_id)
    logger.info(f"Loaded tracker: {len(completed)} videos already processed on GPU {args.gpu_id}")

    # Load RAFT model
    logger.info(f"Loading RAFT model on {device}...")
    weights = Raft_Large_Weights.DEFAULT
    raft_model = raft_large(weights=weights).to(device)
    raft_model.eval()
    logger.info("RAFT model loaded successfully")

    # Collect all unique video directories from manifests
    seen_dirs = set()
    all_videos = []
    for manifest_path in args.manifests:
        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest not found: {manifest_path} — skipping")
            continue
        samples = load_manifest(manifest_path)
        for row in samples:
            d = row['data_dir']
            if d not in seen_dirs:
                seen_dirs.add(d)
                all_videos.append(row)
        logger.info(f"Loaded manifest: {manifest_path} ({len(samples)} entries)")

    logger.info(f"Total unique videos across all manifests: {len(all_videos)}")

    # Round-robin assignment for multi-GPU
    my_videos = [v for i, v in enumerate(all_videos) if i % args.num_gpus == args.gpu_id]
    logger.info(f"Assigned to GPU {args.gpu_id}: {len(my_videos)} videos")

    # Process
    stats = {
        "processed": 0,
        "skipped_done": 0,
        "skipped_other": 0,
        "failed": 0,
        "total_flow_frames": 0,
    }
    pipeline_start = time.time()

    for idx, row in enumerate(my_videos):
        video_dir = row['data_dir']
        generator = row.get('generator', 'unknown')
        video_id = row.get('video_id', os.path.basename(video_dir))

        # Skip if already in tracker (resume-safe)
        if video_dir in completed:
            stats["skipped_done"] += 1
            continue

        # Also skip if output file already exists (e.g. from another GPU run)
        output_path = os.path.join(video_dir, 'flow_haar_hh.npy')
        if os.path.exists(output_path):
            stats["skipped_done"] += 1
            # Record in tracker so we don't re-check next time
            append_tracker(args.gpu_id, {
                "video_dir": video_dir, "generator": generator,
                "video_id": video_id, "n_flow_frames": 0,
                "status": "done", "timestamp": datetime.now().isoformat(),
                "duration_sec": "0.00", "gpu_id": args.gpu_id,
                "error": "already_existed",
            })
            completed.add(video_dir)
            continue

        t0 = time.time()
        try:
            n_flow, status = compute_flow_for_video(
                video_dir, raft_model, device, args.max_frames
            )
            dur = time.time() - t0

            if status == "done":
                stats["processed"] += 1
                stats["total_flow_frames"] += n_flow

                append_tracker(args.gpu_id, {
                    "video_dir": video_dir, "generator": generator,
                    "video_id": video_id, "n_flow_frames": n_flow,
                    "status": "done", "timestamp": datetime.now().isoformat(),
                    "duration_sec": f"{dur:.2f}", "gpu_id": args.gpu_id,
                    "error": "",
                })
                completed.add(video_dir)

                if (stats["processed"]) % 10 == 0:
                    logger.info(
                        f"[{generator}/{video_id}] {n_flow} flow frames ({dur:.1f}s) | "
                        f"Progress: {idx+1}/{len(my_videos)} | "
                        f"Done: {stats['processed']} | Skipped: {stats['skipped_done']}"
                    )
            else:
                stats["skipped_other"] += 1
                append_tracker(args.gpu_id, {
                    "video_dir": video_dir, "generator": generator,
                    "video_id": video_id, "n_flow_frames": 0,
                    "status": "skipped", "timestamp": datetime.now().isoformat(),
                    "duration_sec": f"{dur:.2f}", "gpu_id": args.gpu_id,
                    "error": status,
                })
                logger.debug(f"[{generator}/{video_id}] Skipped: {status}")

        except Exception as e:
            dur = time.time() - t0
            stats["failed"] += 1
            err_msg = str(e)[:300]

            append_tracker(args.gpu_id, {
                "video_dir": video_dir, "generator": generator,
                "video_id": video_id, "n_flow_frames": 0,
                "status": "failed", "timestamp": datetime.now().isoformat(),
                "duration_sec": f"{dur:.2f}", "gpu_id": args.gpu_id,
                "error": err_msg,
            })
            logger.warning(f"[{generator}/{video_id}] FAILED ({dur:.1f}s): {err_msg}")

        # Summary progress every 50 videos
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - pipeline_start
            total_handled = stats["processed"] + stats["skipped_done"] + stats["skipped_other"] + stats["failed"]
            remaining = len(my_videos) - idx - 1
            rate = total_handled / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0

            logger.info(
                f"--- GPU {args.gpu_id} PROGRESS [{idx+1}/{len(my_videos)}] ---  "
                f"Processed: {stats['processed']} | Skipped: {stats['skipped_done']} | "
                f"Failed: {stats['failed']} | Flow frames: {stats['total_flow_frames']} | "
                f"Rate: {rate:.1f} vid/s | ETA: {eta/60:.1f} min"
            )

            # Write progress CSV
            write_progress(args.gpu_id, {
                "gpu_id": args.gpu_id,
                "total_videos": len(my_videos),
                "processed": stats["processed"],
                "skipped_done": stats["skipped_done"],
                "skipped_other": stats["skipped_other"],
                "failed": stats["failed"],
                "total_flow_frames": stats["total_flow_frames"],
                "status": "running",
                "start_time": datetime.fromtimestamp(pipeline_start).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_sec": f"{elapsed:.0f}",
            })

    # Final summary
    elapsed = time.time() - pipeline_start

    logger.info("")
    logger.info("#" * 60)
    logger.info(f"GPU {args.gpu_id} COMPLETE")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Processed: {stats['processed']} videos ({stats['total_flow_frames']} flow frames)")
    logger.info(f"Skipped (already done): {stats['skipped_done']}")
    logger.info(f"Skipped (other): {stats['skipped_other']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info("#" * 60)

    # Write final progress
    write_progress(args.gpu_id, {
        "gpu_id": args.gpu_id,
        "total_videos": len(my_videos),
        "processed": stats["processed"],
        "skipped_done": stats["skipped_done"],
        "skipped_other": stats["skipped_other"],
        "failed": stats["failed"],
        "total_flow_frames": stats["total_flow_frames"],
        "status": "done" if stats["failed"] == 0 else "partial",
        "start_time": datetime.fromtimestamp(pipeline_start).isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_sec": f"{elapsed:.0f}",
    })


if __name__ == '__main__':
    main()
