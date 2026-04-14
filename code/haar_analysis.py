"""
Haar Wavelet JSD Analysis — compute Jensen-Shannon Divergence between
real and fake video Haar subbands per generator.
Streams data from GCS via the dataset.py cache layer.

Outputs:
  /data/haar_analysis/results/haar_jsd_results.json
  /data/haar_analysis/results/generator_clusters.json
  /data/haar_analysis/figures/haar_jsd_heatmap.png
"""
import os
import sys
import json
import time
import random
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, '/data/code')
from dataset import load_manifest, load_npy

MANIFEST = "/data/deepfake_pipeline/splits/manifest_full.csv"
OUTPUT_DIR = "/data/haar_analysis/results"
FIGURE_DIR = "/data/haar_analysis/figures"
SAMPLES_PER_GEN = 30    # videos per generator to sample
FRAMES_PER_VIDEO = 5    # frames per video to sample
HIST_BINS = 128         # histogram bins for JSD computation


def compute_haar_histogram(hh1_frame, bins=HIST_BINS):
    """Compute normalized histogram of Haar HH1 coefficients."""
    flat = hh1_frame.flatten()
    # Use fixed range based on typical Haar coefficient range
    hist, _ = np.histogram(flat, bins=bins, range=(-50, 50), density=True)
    # Add small epsilon to avoid log(0) in JSD
    hist = hist + 1e-10
    hist = hist / hist.sum()
    return hist


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    random.seed(42)

    print("Loading manifest...")
    manifest = load_manifest(MANIFEST)

    # Group by generator
    by_gen = defaultdict(list)
    for row in manifest:
        by_gen[row['generator']].append(row)

    # Separate real generators
    real_gens = {g for g in by_gen if g.startswith('real_')}
    fake_gens = {g for g in by_gen if g not in real_gens}

    print(f"Generators: {len(fake_gens)} fake, {len(real_gens)} real")

    # ---- Step 1: Compute reference histogram from real videos ----
    print("\n=== Computing real video reference distribution ===")
    real_hists = {band: [] for band in ['HH1', 'HL1', 'LH1']}
    band_files = {'HH1': 'haar_hh1.npy', 'HL1': 'haar_hl1.npy', 'LH1': 'haar_lh1.npy'}

    real_count = 0
    for gen in sorted(real_gens):
        rows = by_gen[gen]
        sample = random.sample(rows, min(SAMPLES_PER_GEN, len(rows)))
        for row in sample:
            data_dir = row['data_dir']
            try:
                for band_name, band_file in band_files.items():
                    arr = load_npy(data_dir, band_file)
                    n_frames = arr.shape[0]
                    frame_indices = random.sample(range(n_frames), min(FRAMES_PER_VIDEO, n_frames))
                    for fi in frame_indices:
                        hist = compute_haar_histogram(arr[fi])
                        real_hists[band_name].append(hist)
                real_count += 1
            except Exception as e:
                print(f"  Skip {data_dir}: {e}")
                continue

        print(f"  {gen}: {len(sample)} videos sampled")

    # Average real histograms per band
    real_ref = {}
    for band in ['HH1', 'HL1', 'LH1']:
        real_ref[band] = np.mean(real_hists[band], axis=0)
        real_ref[band] = real_ref[band] / real_ref[band].sum()

    print(f"Real reference: {real_count} videos, {sum(len(v) for v in real_hists.values())} histograms")

    # ---- Step 2: Compute JSD per fake generator ----
    print("\n=== Computing JSD per fake generator ===")
    results = {}

    for gen in sorted(fake_gens):
        rows = by_gen[gen]
        sample = random.sample(rows, min(SAMPLES_PER_GEN, len(rows)))

        gen_hists = {band: [] for band in ['HH1', 'HL1', 'LH1']}
        success = 0

        for row in sample:
            data_dir = row['data_dir']
            try:
                for band_name, band_file in band_files.items():
                    arr = load_npy(data_dir, band_file)
                    n_frames = arr.shape[0]
                    frame_indices = random.sample(range(n_frames), min(FRAMES_PER_VIDEO, n_frames))
                    for fi in frame_indices:
                        hist = compute_haar_histogram(arr[fi])
                        gen_hists[band_name].append(hist)
                success += 1
            except Exception as e:
                continue

        if success == 0:
            print(f"  {gen}: SKIPPED (no data)")
            continue

        # Compute JSD per band
        jsd_scores = {}
        for band in ['HH1', 'HL1', 'LH1']:
            fake_avg = np.mean(gen_hists[band], axis=0)
            fake_avg = fake_avg / fake_avg.sum()
            jsd = float(jensenshannon(real_ref[band], fake_avg) ** 2)  # JSD (squared JS distance)
            jsd_scores[band] = round(jsd, 6)

        best_band = max(jsd_scores, key=jsd_scores.get)
        jsd_scores['best_band'] = best_band
        jsd_scores['best_jsd'] = jsd_scores[best_band]
        results[gen] = jsd_scores

        print(f"  {gen}: HH1={jsd_scores['HH1']:.4f} HL1={jsd_scores['HL1']:.4f} "
              f"LH1={jsd_scores['LH1']:.4f} → best={best_band} ({jsd_scores[best_band]:.4f}) "
              f"[{success}/{len(sample)} videos]")

    # ---- Step 3: Cluster generators by JSD profile ----
    print("\n=== Clustering generators ===")
    gen_names = sorted(results.keys())
    jsd_matrix = np.array([[results[g]['HH1'], results[g]['HL1'], results[g]['LH1']] for g in gen_names])

    if len(gen_names) >= 3:
        Z = linkage(jsd_matrix, method='ward')
        cluster_labels = fcluster(Z, t=3, criterion='maxclust')

        clusters = defaultdict(list)
        for gen, cl in zip(gen_names, cluster_labels):
            clusters[int(cl)].append(gen)

        # Sort clusters by mean JSD (highest first)
        cluster_means = {}
        for cl, gens in clusters.items():
            mean_jsd = np.mean([results[g]['best_jsd'] for g in gens])
            cluster_means[cl] = mean_jsd

        sorted_clusters = sorted(cluster_means, key=cluster_means.get, reverse=True)
        remap = {old: new + 1 for new, old in enumerate(sorted_clusters)}

        cluster_output = {}
        for old_cl in sorted_clusters:
            new_cl = remap[old_cl]
            gens = sorted(clusters[old_cl])
            mean = cluster_means[old_cl]
            desc = ['High', 'Medium', 'Low'][new_cl - 1] if new_cl <= 3 else 'Other'
            cluster_output[f"cluster_{new_cl}"] = {
                "generators": gens,
                "mean_jsd": round(mean, 4),
                "description": f"{desc} JSD — {'strong' if new_cl == 1 else 'moderate' if new_cl == 2 else 'subtle'} spatial artifacts"
            }
            print(f"  Cluster {new_cl} ({desc}, mean JSD={mean:.4f}): {gens}")
    else:
        cluster_output = {"cluster_1": {"generators": gen_names, "description": "All generators"}}
        remap = {1: 1}
        cluster_labels = [1] * len(gen_names)

    # ---- Step 4: Save results ----
    # Save JSD results
    jsd_path = os.path.join(OUTPUT_DIR, 'haar_jsd_results.json')
    with open(jsd_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSD results → {jsd_path}")

    # Save cluster assignments
    cluster_path = os.path.join(OUTPUT_DIR, 'generator_clusters.json')
    with open(cluster_path, 'w') as f:
        json.dump(cluster_output, f, indent=2)
    print(f"Saved clusters → {cluster_path}")

    # ---- Step 5: Generate visualization ----
    print("\nGenerating visualization...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle('Haar Wavelet JSD Analysis — Real vs Fake Generators', fontsize=16, fontweight='bold')

    # Plot 1: JSD bar chart per generator
    ax = axes[0]
    gens_sorted = sorted(results.keys(), key=lambda g: results[g]['best_jsd'], reverse=True)
    y_pos = range(len(gens_sorted))
    hh1_vals = [results[g]['HH1'] for g in gens_sorted]
    hl1_vals = [results[g]['HL1'] for g in gens_sorted]
    lh1_vals = [results[g]['LH1'] for g in gens_sorted]

    ax.barh(y_pos, hh1_vals, height=0.25, label='HH1', color='#e74c3c', alpha=0.8)
    ax.barh([y + 0.25 for y in y_pos], hl1_vals, height=0.25, label='HL1', color='#3498db', alpha=0.8)
    ax.barh([y + 0.5 for y in y_pos], lh1_vals, height=0.25, label='LH1', color='#2ecc71', alpha=0.8)
    ax.set_yticks([y + 0.25 for y in y_pos])
    ax.set_yticklabels(gens_sorted, fontsize=8)
    ax.set_xlabel('JSD Score')
    ax.set_title('JSD per Band per Generator')
    ax.legend()
    ax.invert_yaxis()

    # Plot 2: Cluster visualization
    ax = axes[1]
    cluster_colors = ['#e74c3c', '#f39c12', '#3498db']
    for cl_name, cl_data in sorted(cluster_output.items()):
        cl_num = int(cl_name.split('_')[1]) - 1
        gens = cl_data['generators']
        jsd_vals = [results[g]['best_jsd'] for g in gens]
        color = cluster_colors[cl_num % len(cluster_colors)]
        ax.scatter(jsd_vals, [cl_num] * len(jsd_vals), s=100, c=color, alpha=0.7, edgecolors='black')
        for g, j in zip(gens, jsd_vals):
            ax.annotate(g.replace('deepaction_', 'da_').replace('waverep_', 'wr_').replace('dvf_', 'dvf_'),
                        (j, cl_num), fontsize=6, rotation=30, ha='left')
    ax.set_xlabel('Best JSD Score')
    ax.set_ylabel('Cluster')
    ax.set_yticks(range(len(cluster_output)))
    ax.set_yticklabels([f"Cluster {i+1}" for i in range(len(cluster_output))])
    ax.set_title('Generator Clusters by JSD')

    # Plot 3: Heatmap
    ax = axes[2]
    heatmap_data = np.array([[results[g]['HH1'], results[g]['HL1'], results[g]['LH1']] for g in gens_sorted])
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(len(gens_sorted)))
    ax.set_yticklabels(gens_sorted, fontsize=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['HH1', 'HL1', 'LH1'])
    ax.set_title('JSD Heatmap')
    plt.colorbar(im, ax=ax, label='JSD')

    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, 'haar_jsd_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure → {fig_path}")

    # ---- Update manifests with new cluster/JSD info ----
    print("\nUpdating manifests with new Haar analysis...")
    import csv

    # Build lookup: generator → cluster, best_band, best_jsd
    gen_to_cluster = {}
    for cl_name, cl_data in cluster_output.items():
        cl_num = int(cl_name.split('_')[1])
        for g in cl_data['generators']:
            gen_to_cluster[g] = cl_num

    for split in ['manifest_train', 'manifest_val', 'manifest_test', 'manifest_full']:
        csv_path = f"/data/deepfake_pipeline/splits/{split}.csv"
        rows = load_manifest(csv_path)
        for row in rows:
            gen = row['generator']
            if gen in results:
                row['cluster'] = gen_to_cluster.get(gen, '')
                row['best_haar_band'] = results[gen]['best_band']
                row['best_haar_jsd'] = results[gen]['best_jsd']

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("Manifests updated with new cluster assignments and JSD scores")
    print("\nDone!")


if __name__ == '__main__':
    main()
