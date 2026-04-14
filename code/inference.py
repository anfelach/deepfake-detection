"""
Inference pipeline: Stage 1 (Sentry Gate) + Stage 2 (Spatial Fingerprint).
Input: one or more video files → combined deepfake probability.
Combined score: 30% Stage1 + 70% Stage2.

Usage:
  python inference.py video.mp4
  python inference.py video1.mp4 video2.mp4 --threshold 0.5 --output results.json
"""
import argparse
import json
import sys
import os
import time
import numpy as np
import torch
import cv2
import pywt
from PIL import Image
import torchvision.transforms.functional as TF

from models import SentryGateModel, SpatialFingerprintModel
from config_train import Stage1Config, Stage2Config


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


def extract_frames(video_path, max_frames=None):
    """Extract RGB frames from video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR → RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    frames = np.array(frames)  # (N, H, W, 3)

    if max_frames and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]

    return frames


def compute_haar_hh1(frames):
    """Compute HH1 Haar wavelet subband for each frame on GPU."""
    hh1_list = []
    for frame in frames:
        gray = np.mean(frame, axis=2).astype(np.float32)
        coeffs = pywt.dwt2(gray, 'haar')
        _, (_, _, hh1) = coeffs
        hh1_list.append(hh1)
    return np.array(hh1_list, dtype=np.float32)  # (N, H/2, W/2)


def preprocess_rgb(frames, image_size=224):
    """Convert RGB frames to normalized tensors."""
    tensors = []
    for f in frames:
        img = Image.fromarray(f)
        if img.size[0] != image_size or img.size[1] != image_size:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        t = TF.to_tensor(img).to(DEVICE)
        t = (t - MEAN) / STD
        tensors.append(t)
    return torch.stack(tensors)  # (N, 3, 224, 224)


def preprocess_hh1(hh1_frames, image_size=224):
    """Convert HH1 wavelet frames to normalized 3-channel tensors."""
    tensors = []
    for h in hh1_frames:
        h_min, h_max = h.min(), h.max()
        if h_max - h_min > 1e-8:
            h = (h - h_min) / (h_max - h_min)
        else:
            h = np.zeros_like(h)
        h_img = Image.fromarray((h * 255).astype(np.uint8), mode='L')
        h_img = h_img.resize((image_size, image_size), Image.BILINEAR)
        t = TF.to_tensor(h_img).repeat(3, 1, 1).to(DEVICE)
        t = (t - MEAN) / STD
        tensors.append(t)
    return torch.stack(tensors)  # (N, 3, 224, 224)


def load_models():
    """Load Stage 1 and Stage 2 models from checkpoints."""
    s1_cfg = Stage1Config()
    s2_cfg = Stage2Config()

    # Stage 1
    s1_ckpt = os.path.join(s1_cfg.output_dir, 'checkpoints', 'best_model.pt')
    s1_model = SentryGateModel(s1_cfg.dinov3_model, s1_cfg.hf_token,
                                s1_cfg.hidden_size, s1_cfg.num_clusters)
    if os.path.exists(s1_ckpt):
        state = torch.load(s1_ckpt, map_location=DEVICE)
        s1_model.load_state_dict(state['model'])
        print(f"Stage 1 loaded: {s1_ckpt}")
    else:
        print(f"WARNING: Stage 1 checkpoint not found at {s1_ckpt}")
    s1_model = s1_model.to(DEVICE).eval()

    # Stage 2
    s2_ckpt = os.path.join(s2_cfg.output_dir, 'checkpoints', 'best_model.pt')
    s2_model = SpatialFingerprintModel(s2_cfg.dinov3_model, s2_cfg.hf_token,
                                        s2_cfg.hidden_size)
    if os.path.exists(s2_ckpt):
        state = torch.load(s2_ckpt, map_location=DEVICE)
        s2_model.load_state_dict(state['model'])
        print(f"Stage 2 loaded: {s2_ckpt}")
    else:
        print(f"WARNING: Stage 2 checkpoint not found at {s2_ckpt}")
    s2_model = s2_model.to(DEVICE).eval()

    return s1_model, s2_model


@torch.no_grad()
def classify_video(video_path, s1_model, s2_model, threshold=0.5):
    """Run full inference pipeline on a single video."""
    t0 = time.time()

    # Extract frames
    all_frames = extract_frames(video_path)
    n_total = len(all_frames)

    # Compute Haar wavelets
    hh1_all = compute_haar_hh1(all_frames)

    # ---- Stage 1: 64 frames, quick screening ----
    if n_total >= 64:
        s1_indices = np.linspace(0, n_total - 1, 64, dtype=int)
    else:
        s1_indices = np.arange(n_total)
        s1_indices = np.pad(s1_indices, (0, 64 - n_total), mode='wrap')

    s1_rgb = preprocess_rgb(all_frames[s1_indices[:n_total]] if n_total < 64 else all_frames[s1_indices])
    s1_hh1 = preprocess_hh1(hh1_all[s1_indices[:n_total]] if n_total < 64 else hh1_all[s1_indices])

    # Pad if needed
    if n_total < 64:
        s1_rgb_padded = preprocess_rgb(all_frames)
        s1_hh1_padded = preprocess_hh1(hh1_all)
        pad_size = 64 - n_total
        s1_rgb = torch.cat([s1_rgb_padded, s1_rgb_padded[:pad_size]])
        s1_hh1 = torch.cat([s1_hh1_padded, s1_hh1_padded[:pad_size]])

    s1_rgb = s1_rgb.unsqueeze(0)  # (1, 64, 3, 224, 224)
    s1_hh1 = s1_hh1.unsqueeze(0)

    binary_logits, cluster_logits = s1_model(s1_rgb, s1_hh1)
    s1_prob = torch.sigmoid(binary_logits).item()
    s1_cluster = torch.argmax(cluster_logits, dim=1).item()

    # ---- Stage 2: all frames, deep analysis ----
    max_frames = 256
    if n_total > max_frames:
        s2_indices = np.linspace(0, n_total - 1, max_frames, dtype=int)
    else:
        s2_indices = np.arange(n_total)

    s2_rgb = preprocess_rgb(all_frames[s2_indices])
    s2_hh1 = preprocess_hh1(hh1_all[s2_indices])

    actual = len(s2_indices)
    # Pad to max_frames
    if actual < max_frames:
        pad = max_frames - actual
        s2_rgb = torch.cat([s2_rgb, torch.zeros(pad, 3, 224, 224, device=DEVICE)])
        s2_hh1 = torch.cat([s2_hh1, torch.zeros(pad, 3, 224, 224, device=DEVICE)])

    attention_mask = torch.zeros(max_frames, dtype=torch.bool, device=DEVICE)
    attention_mask[:actual] = True

    s2_rgb = s2_rgb.unsqueeze(0)  # (1, T, 3, 224, 224)
    s2_hh1 = s2_hh1.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

    s2_logits = s2_model(s2_rgb, s2_hh1, attention_mask)
    s2_prob = torch.sigmoid(s2_logits).item()

    # Combined score: 30% Stage1 + 70% Stage2
    combined = 0.3 * s1_prob + 0.7 * s2_prob
    elapsed = time.time() - t0

    result = {
        'video': os.path.basename(video_path),
        'path': video_path,
        'total_frames': n_total,
        'stage1_prob': round(s1_prob, 4),
        'stage1_cluster': s1_cluster,
        'stage2_prob': round(s2_prob, 4),
        'combined_prob': round(combined, 4),
        'prediction': 'FAKE' if combined >= threshold else 'REAL',
        'threshold': threshold,
        'time_seconds': round(elapsed, 2),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Deepfake Video Detection Pipeline')
    parser.add_argument('videos', nargs='+', help='Video file(s) to analyze')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON file')
    args = parser.parse_args()

    print("Loading models...")
    s1_model, s2_model = load_models()

    results = []
    for vpath in args.videos:
        if not os.path.exists(vpath):
            print(f"File not found: {vpath}")
            continue

        print(f"\nAnalyzing: {vpath}")
        try:
            result = classify_video(vpath, s1_model, s2_model, args.threshold)
            results.append(result)

            label = result['prediction']
            prob = result['combined_prob']
            s1 = result['stage1_prob']
            s2 = result['stage2_prob']
            print(f"  Result: {label} (combined={prob:.4f}, S1={s1:.4f}, S2={s2:.4f})")
            print(f"  Frames: {result['total_frames']}, Time: {result['time_seconds']}s")
        except Exception as e:
            print(f"  ERROR: {e}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()