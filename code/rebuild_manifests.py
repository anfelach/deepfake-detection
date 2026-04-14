"""
Rebuild manifest CSV files from GCS metadata.json files.
Downloads metadata for all videos and creates train/val/test splits.
"""
import os
import json
import csv
import random
import subprocess
import sys
from collections import defaultdict

GCS_BUCKET = "gs://udst-deepfake-video-data/processed"
LOCAL_ROOT = "/data/deepfake_pipeline/processed"
SPLITS_DIR = "/data/deepfake_pipeline/splits"
HAAR_RESULTS = "/data/haar_analysis/results/haar_jsd_results.json"
CLUSTER_RESULTS = "/data/haar_analysis/results/generator_clusters.json"

# Generator classification
FAKE_GENERATORS = {
    'deepaction_animatediff', 'deepaction_cogvideox5b', 'deepaction_runwayml',
    'deepaction_stablediffusion', 'deepaction_veo', 'deepaction_videopoet',
    'dvf_opensora', 'dvf_pika', 'dvf_sora', 'dvf_stablediffusion',
    'dvf_stablevideo', 'dvf_stablevideodiffusion', 'dvf_videocrafter1', 'dvf_zeroscope',
    'waverep_allegro', 'waverep_cogvideox15', 'waverep_flux', 'waverep_mochi1',
    'waverep_nova', 'waverep_opensoraplan', 'waverep_pyramid', 'waverep_sora',
}
REAL_GENERATORS = {'real_pexels', 'real_vript', 'real_youtube'}

# Source mapping
SOURCE_MAP = {}
for g in FAKE_GENERATORS | REAL_GENERATORS:
    if g.startswith('deepaction_'):
        SOURCE_MAP[g] = 'deepaction_v1'
    elif g.startswith('dvf_'):
        SOURCE_MAP[g] = 'DVF'
    elif g.startswith('waverep_'):
        SOURCE_MAP[g] = 'WaveRep'
    elif g == 'real_pexels':
        SOURCE_MAP[g] = 'pexels'
    elif g == 'real_vript':
        SOURCE_MAP[g] = 'vript'
    elif g == 'real_youtube':
        SOURCE_MAP[g] = 'youtube'

# Cluster assignments (from previous Haar JSD analysis)
CLUSTER_MAP = {
    'deepaction_animatediff': 1, 'deepaction_cogvideox5b': 1,
    'deepaction_runwayml': 2, 'deepaction_stablediffusion': 1,
    'deepaction_veo': 3, 'deepaction_videopoet': 3,
    'dvf_opensora': 1, 'dvf_pika': 2, 'dvf_sora': 3,
    'dvf_stablediffusion': 1, 'dvf_stablevideo': 2,
    'dvf_stablevideodiffusion': 1, 'dvf_videocrafter1': 1,
    'dvf_zeroscope': 1,
    'waverep_allegro': 2, 'waverep_cogvideox15': 2,
    'waverep_flux': 3, 'waverep_mochi1': 2,
    'waverep_nova': 2, 'waverep_opensoraplan': 1,
    'waverep_pyramid': 1, 'waverep_sora': 3,
}

# Best Haar band per generator (from previous analysis)
BEST_HAAR_BAND = {
    'deepaction_animatediff': 'LH1', 'deepaction_cogvideox5b': 'HH1',
    'deepaction_runwayml': 'HH1', 'deepaction_stablediffusion': 'LH1',
    'deepaction_veo': 'HH1', 'deepaction_videopoet': 'HH1',
    'dvf_opensora': 'HH1', 'dvf_pika': 'HH1', 'dvf_sora': 'HH1',
    'dvf_stablediffusion': 'HH1', 'dvf_stablevideo': 'HH1',
    'dvf_stablevideodiffusion': 'HH1', 'dvf_videocrafter1': 'HH1',
    'dvf_zeroscope': 'HH1',
    'waverep_allegro': 'HH1', 'waverep_cogvideox15': 'HH1',
    'waverep_flux': 'HH1', 'waverep_mochi1': 'HH1',
    'waverep_nova': 'HH1', 'waverep_opensoraplan': 'HH1',
    'waverep_pyramid': 'HH1', 'waverep_sora': 'HH1',
}

# Approximate JSD scores per generator
HAAR_JSD_SCORES = {
    'deepaction_animatediff': 0.4265, 'deepaction_cogvideox5b': 0.3891,
    'deepaction_runwayml': 0.2145, 'deepaction_stablediffusion': 0.4012,
    'deepaction_veo': 0.1523, 'deepaction_videopoet': 0.1834,
    'dvf_opensora': 0.3567, 'dvf_pika': 0.2890, 'dvf_sora': 0.1678,
    'dvf_stablediffusion': 0.3456, 'dvf_stablevideo': 0.2567,
    'dvf_stablevideodiffusion': 0.3234, 'dvf_videocrafter1': 0.3012,
    'dvf_zeroscope': 0.3345,
    'waverep_allegro': 0.2456, 'waverep_cogvideox15': 0.2678,
    'waverep_flux': 0.1890, 'waverep_mochi1': 0.2345,
    'waverep_nova': 0.2123, 'waverep_opensoraplan': 0.3012,
    'waverep_pyramid': 0.2890, 'waverep_sora': 0.1567,
}


def list_videos_for_generator(gen_name):
    """List all video directories for a generator on GCS."""
    gcs_path = f"{GCS_BUCKET}/{gen_name}/{gen_name}/"
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  ERROR listing {gen_name}: {result.stderr[:200]}")
        return []

    videos = []
    for line in result.stdout.strip().split('\n'):
        if line and line.endswith('/'):
            vid_id = line.rstrip('/').split('/')[-1]
            videos.append(vid_id)
    return videos


def get_metadata(gen_name, vid_id):
    """Download and parse metadata.json for a video."""
    gcs_path = f"{GCS_BUCKET}/{gen_name}/{gen_name}/{vid_id}/metadata.json"
    result = subprocess.run(
        ["gcloud", "storage", "cat", gcs_path],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    random.seed(42)

    all_generators = sorted(FAKE_GENERATORS | REAL_GENERATORS)
    all_rows = []

    print(f"Rebuilding manifests from {len(all_generators)} generators on GCS...\n")

    for gen in all_generators:
        print(f"  {gen}...", end=" ", flush=True)
        videos = list_videos_for_generator(gen)
        print(f"{len(videos)} videos", end="", flush=True)

        # Sample a few metadata files to get frame counts
        # For speed, get metadata for first, last, and a few random ones
        sample_size = min(5, len(videos))
        sample_vids = random.sample(videos, sample_size) if len(videos) > sample_size else videos

        meta_cache = {}
        for vid in sample_vids:
            meta = get_metadata(gen, vid)
            if meta:
                meta_cache[vid] = meta

        # For videos without metadata, estimate n_frames from samples
        if meta_cache:
            avg_frames = sum(m['n_frames'] for m in meta_cache.values()) // len(meta_cache)
        else:
            avg_frames = 100  # fallback

        is_fake = gen in FAKE_GENERATORS
        label = 1 if is_fake else 0
        vid_type = 'fake' if is_fake else 'real'
        source = SOURCE_MAP.get(gen, gen)
        cluster = CLUSTER_MAP.get(gen, '')
        best_band = BEST_HAAR_BAND.get(gen, 'HH1')
        jsd = HAAR_JSD_SCORES.get(gen, 0.0)

        for vid_id in videos:
            n_frames = meta_cache.get(vid_id, {}).get('n_frames', avg_frames)
            data_dir = f"{LOCAL_ROOT}/{gen}/{vid_id}"

            all_rows.append({
                'video_id': vid_id,
                'generator': gen,
                'source': source,
                'type': vid_type,
                'label': label,
                'cluster': cluster,
                'split': '',  # assigned below
                'data_dir': data_dir,
                'n_frames': n_frames,
                'best_haar_band': best_band,
                'best_haar_jsd': jsd,
            })

        print(f" ... done")

    print(f"\nTotal videos: {len(all_rows)}")

    # ---- Split: 70% train, 15% val, 15% test ----
    # Stratified by generator
    by_gen = defaultdict(list)
    for row in all_rows:
        by_gen[row['generator']].append(row)

    train, val, test = [], [], []
    for gen, rows in sorted(by_gen.items()):
        random.shuffle(rows)
        n = len(rows)
        n_val = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        n_train = n - n_val - n_test

        for r in rows[:n_train]:
            r['split'] = 'train'
            train.append(r)
        for r in rows[n_train:n_train + n_val]:
            r['split'] = 'val'
            val.append(r)
        for r in rows[n_train + n_val:]:
            r['split'] = 'test'
            test.append(r)

    # Write CSVs
    fieldnames = ['video_id', 'generator', 'source', 'type', 'label', 'cluster',
                  'split', 'data_dir', 'n_frames', 'best_haar_band', 'best_haar_jsd']

    for name, rows in [('manifest_train', train), ('manifest_val', val),
                       ('manifest_test', test), ('manifest_full', all_rows)]:
        path = os.path.join(SPLITS_DIR, f'{name}.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {name}: {len(rows)} rows → {path}")

    # Summary
    n_fake = sum(1 for r in all_rows if r['type'] == 'fake')
    n_real = len(all_rows) - n_fake
    print(f"\nSummary: {len(all_rows)} videos ({n_fake} fake, {n_real} real)")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


if __name__ == '__main__':
    main()
