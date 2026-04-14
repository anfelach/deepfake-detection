"""
Dataset visualization: 7 figures covering class balance, generators,
frame distribution, clusters, training curves, and data sources.

Usage: python visualize_dataset.py [--output-dir /data/code/figures]
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter, defaultdict


def load_manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/data/code/figures')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    full = load_manifest('/data/deepfake_pipeline/splits/manifest_full.csv')
    train = load_manifest('/data/deepfake_pipeline/splits/manifest_train.csv')
    val = load_manifest('/data/deepfake_pipeline/splits/manifest_val.csv')
    test = load_manifest('/data/deepfake_pipeline/splits/manifest_test.csv')

    print(f"Full: {len(full)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # ================================================================
    # FIGURE 1: Class Balance
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    n_real = sum(1 for r in full if r['type'] == 'real')
    n_fake = sum(1 for r in full if r['type'] == 'fake')
    bars = ax.bar(['Real', 'Fake'], [n_real, n_fake], color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
    for bar, count in zip(bars, [n_real, n_fake]):
        pct = count / len(full) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Videos', fontsize=13)
    ax.set_title('Class Balance', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(n_real, n_fake) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fig1_class_balance.png'), dpi=150)
    plt.close()
    print("Figure 1: Class Balance")

    # ================================================================
    # FIGURE 2: Generator Distribution
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    gen_counts = Counter(r['generator'] for r in full)
    gens_sorted = sorted(gen_counts.keys(), key=lambda g: gen_counts[g], reverse=True)
    counts = [gen_counts[g] for g in gens_sorted]
    colors = ['#e74c3c' if any(r['generator'] == g and r['type'] == 'fake' for r in full) else '#2ecc71'
              for g in gens_sorted]
    bars = ax.barh(range(len(gens_sorted)), counts, color=colors, edgecolor='white')
    ax.set_yticks(range(len(gens_sorted)))
    ax.set_yticklabels(gens_sorted, fontsize=9)
    ax.set_xlabel('Number of Videos', fontsize=12)
    ax.set_title(f'Generator Distribution ({len(gen_counts)} generators)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(c), va='center', fontsize=8)
    ax.legend([Patch(color='#e74c3c'), Patch(color='#2ecc71')], ['Fake', 'Real'], loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fig2_generator_distribution.png'), dpi=150)
    plt.close()
    print("Figure 2: Generator Distribution")

    # ================================================================
    # FIGURE 3: Frame Distribution
    # ================================================================
    all_frames = [int(r['n_frames']) for r in full]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Frame Count Distribution', fontsize=16, fontweight='bold')

    ax1.hist(all_frames, bins=80, color='#3498db', edgecolor='white', alpha=0.8)
    ax1.axvline(np.median(all_frames), color='red', linestyle='--', linewidth=2, label=f'Median: {int(np.median(all_frames))}')
    ax1.axvline(np.mean(all_frames), color='orange', linestyle='--', linewidth=2, label=f'Mean: {int(np.mean(all_frames))}')
    ax1.axvline(64, color='green', linestyle=':', linewidth=2, label='Stage 1 cap (64)')
    ax1.axvline(256, color='purple', linestyle=':', linewidth=2, label='Stage 2 cap (256)')
    ax1.set_xlabel('Frames per Video')
    ax1.set_ylabel('Count')
    ax1.set_title('Histogram')
    ax1.legend(fontsize=9)
    stats = f'Min: {min(all_frames)}\n25th: {int(np.percentile(all_frames, 25))}\nMedian: {int(np.median(all_frames))}\n75th: {int(np.percentile(all_frames, 75))}\n95th: {int(np.percentile(all_frames, 95))}\nMax: {max(all_frames)}'
    ax1.text(0.95, 0.95, stats, transform=ax1.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), family='monospace')

    sorted_frames = np.sort(all_frames)
    cdf = np.arange(1, len(sorted_frames) + 1) / len(sorted_frames) * 100
    ax2.plot(sorted_frames, cdf, color='#2c3e50', linewidth=2)
    ax2.axvline(64, color='green', linestyle=':', linewidth=2, label='Stage 1 (64)')
    ax2.axvline(256, color='purple', linestyle=':', linewidth=2, label='Stage 2 (256)')
    for p in [50, 75, 95]:
        v = int(np.percentile(all_frames, p))
        ax2.plot(v, p, 'ro', markersize=8)
        ax2.annotate(f'{p}th: {v}', (v, p), textcoords="offset points", xytext=(10, -5), fontsize=9, color='red')
    pct64 = np.mean(np.array(all_frames) <= 64) * 100
    pct256 = np.mean(np.array(all_frames) <= 256) * 100
    ax2.text(0.95, 0.15, f'<= 64: {pct64:.1f}%\n<= 256: {pct256:.1f}%',
             transform=ax2.transAxes, ha='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax2.set_xlabel('Frames per Video')
    ax2.set_ylabel('Cumulative %')
    ax2.set_title('CDF')
    ax2.legend()
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fig3_frame_distribution.png'), dpi=150)
    plt.close()
    print("Figure 3: Frame Distribution")

    # ================================================================
    # FIGURE 4: Cluster Distribution
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = Counter(r['cluster'] for r in full if r['type'] == 'fake' and r['cluster'] not in ('', 'None'))
    clusters = sorted(cluster_counts.keys())
    c_counts = [cluster_counts[c] for c in clusters]
    c_colors = ['#3498db', '#e67e22', '#9b59b6']
    ax.bar([f'Cluster {c}' for c in clusters], c_counts, color=c_colors[:len(clusters)], edgecolor='white')
    for i, (c, cnt) in enumerate(zip(clusters, c_counts)):
        ax.text(i, cnt + 20, str(cnt), ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Fake Videos')
    ax.set_title('Haar JSD Cluster Distribution (Fake Videos)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fig4_cluster_distribution.png'), dpi=150)
    plt.close()
    print("Figure 4: Cluster Distribution")

    # ================================================================
    # FIGURE 5: Stage 1 Training Curves (from CSV if exists)
    # ================================================================
    s1_log = '/data/code/runs/stage1/training_log.csv'
    if os.path.exists(s1_log):
        s1_data = load_manifest(s1_log)
        epochs = [int(r['epoch']) for r in s1_data]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Stage 1: Sentry Gate Training', fontsize=16, fontweight='bold')

        ax1.plot(epochs, [float(r['train_loss']) for r in s1_data], 'b-', label='Train Loss')
        ax1.plot(epochs, [float(r['val_loss']) for r in s1_data], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.set_title('Loss')

        ax2.plot(epochs, [float(r['train_auc']) for r in s1_data], 'b-', label='Train AUC')
        ax2.plot(epochs, [float(r['val_auc']) for r in s1_data], 'r-', label='Val AUC')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC'); ax2.legend(); ax2.set_title('AUC')
        ax2.set_ylim(0.9, 1.01)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'fig5_stage1_curves.png'), dpi=150)
        plt.close()
        print("Figure 5: Stage 1 Training Curves")
    else:
        print("Figure 5: SKIPPED (no Stage 1 training log yet)")

    # ================================================================
    # FIGURE 6: Stage 2 Training Curves (from CSV if exists)
    # ================================================================
    s2_log = '/data/code/runs/stage2/training_log.csv'
    if os.path.exists(s2_log):
        s2_data = load_manifest(s2_log)
        epochs = [int(r['epoch']) for r in s2_data]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Stage 2: Spatial Fingerprint Training', fontsize=16, fontweight='bold')

        ax1.plot(epochs, [float(r['train_loss']) for r in s2_data], 'b-', label='Train Loss')
        ax1.plot(epochs, [float(r['val_loss']) for r in s2_data], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.set_title('Loss')

        ax2.plot(epochs, [float(r['train_auc']) for r in s2_data], 'b-', label='Train AUC')
        ax2.plot(epochs, [float(r['val_auc']) for r in s2_data], 'r-', label='Val AUC')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC'); ax2.legend(); ax2.set_title('AUC')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'fig6_stage2_curves.png'), dpi=150)
        plt.close()
        print("Figure 6: Stage 2 Training Curves")
    else:
        print("Figure 6: SKIPPED (no Stage 2 training log yet)")

    # ================================================================
    # FIGURE 7: Data Source Breakdown
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    source_counts = Counter(r['source'] for r in full)
    sources = sorted(source_counts.keys(), key=lambda s: source_counts[s], reverse=True)
    s_counts = [source_counts[s] for s in sources]
    s_colors = plt.cm.Set3(np.linspace(0, 1, len(sources)))
    bars = ax.barh(range(len(sources)), s_counts, color=s_colors, edgecolor='white')
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=11)
    ax.set_xlabel('Number of Videos')
    ax.set_title('Data Source Breakdown', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, c in zip(bars, s_counts):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, str(c), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fig7_data_sources.png'), dpi=150)
    plt.close()
    print("Figure 7: Data Source Breakdown")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
