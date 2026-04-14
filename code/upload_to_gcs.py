"""
Upload processed data to Google Cloud Storage.
Supports parallel upload, verification, and optional local deletion.

Usage:
  python upload_to_gcs.py                    # Upload all
  python upload_to_gcs.py --verify-only      # Just verify
  python upload_to_gcs.py --delete-after     # Upload, verify, then delete local
  python upload_to_gcs.py --workers 8        # Parallel uploads
"""
import os
import sys
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL_ROOT = "/data/deepfake_pipeline/processed"
GCS_BUCKET = "gs://udst-deepfake-video-data/processed"


def get_local_file_count(path):
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)
    return count


def upload_generator(gen_name, local_root, gcs_bucket):
    """Upload a single generator directory to GCS."""
    local_dir = os.path.join(local_root, gen_name)
    gcs_dest = f"{gcs_bucket}/{gen_name}/"

    t0 = time.time()
    local_count = get_local_file_count(local_dir)

    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{local_dir}/", gcs_dest],
        capture_output=True, text=True, timeout=7200
    )

    elapsed = time.time() - t0
    success = result.returncode == 0

    return {
        'generator': gen_name,
        'success': success,
        'local_files': local_count,
        'time': elapsed,
        'error': result.stderr[:500] if not success else None,
    }


def verify_generator(gen_name, local_root, gcs_bucket):
    """Verify upload by comparing video counts."""
    local_dir = os.path.join(local_root, gen_name)

    # Count local videos
    if os.path.exists(local_dir):
        local_videos = [d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))]
    else:
        local_videos = []

    # Count GCS videos (accounting for double nesting from cp -r)
    gcs_path = f"{gcs_bucket}/{gen_name}/{gen_name}/"
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True, timeout=300
    )
    gcs_videos = [l for l in result.stdout.strip().split('\n') if l and l.endswith('/')]

    return {
        'generator': gen_name,
        'local_videos': len(local_videos),
        'gcs_videos': len(gcs_videos),
        'match': len(local_videos) == len(gcs_videos) or len(local_videos) == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--delete-after', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(LOCAL_ROOT):
        print(f"Local root not found: {LOCAL_ROOT}")
        if args.verify_only:
            print("Running verify against GCS only...")
        else:
            sys.exit(1)

    generators = sorted(d for d in os.listdir(LOCAL_ROOT)
                        if os.path.isdir(os.path.join(LOCAL_ROOT, d))) if os.path.exists(LOCAL_ROOT) else []

    print(f"Found {len(generators)} generator directories")
    print(f"GCS destination: {GCS_BUCKET}")
    print(f"Workers: {args.workers}\n")

    if not args.verify_only and generators:
        # Upload
        print("Uploading...")
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(upload_generator, g, LOCAL_ROOT, GCS_BUCKET): g for g in generators}
            for future in as_completed(futures):
                r = future.result()
                status = "OK" if r['success'] else "FAILED"
                print(f"  {r['generator']}: {r['local_files']} files, {r['time']:.0f}s [{status}]")
                if r['error']:
                    print(f"    Error: {r['error'][:200]}")
                results.append(r)

        failed = [r for r in results if not r['success']]
        print(f"\nUpload complete: {len(results) - len(failed)}/{len(results)} succeeded")
        if failed:
            print(f"FAILED: {[r['generator'] for r in failed]}")

    # Verify
    if generators:
        print("\nVerifying uploads...")
        for gen in generators:
            v = verify_generator(gen, LOCAL_ROOT, GCS_BUCKET)
            status = "OK" if v['match'] else "MISMATCH"
            print(f"  {v['generator']}: local={v['local_videos']}, gcs={v['gcs_videos']} [{status}]")

    # Delete
    if args.delete_after and generators:
        import shutil
        print("\nDeleting local files...")
        for gen in generators:
            v = verify_generator(gen, LOCAL_ROOT, GCS_BUCKET)
            if v['match']:
                path = os.path.join(LOCAL_ROOT, gen)
                shutil.rmtree(path)
                print(f"  Deleted {gen}")
            else:
                print(f"  SKIPPED {gen} (verification failed)")


if __name__ == '__main__':
    main()
