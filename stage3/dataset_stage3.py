"""
Dataset for Stage 3: Temporal Motion Analysis.
Loads pre-computed optical flow Haar HH wavelets (flow_haar_hh.npy).
"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms.functional as TF
from PIL import Image


def load_manifest(csv_path):
    """Load manifest CSV into list of dicts."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))


class TemporalMotionDataset(Dataset):
    """
    Stage 3 dataset: pre-computed optical flow Haar HH wavelets.
    Returns padded sequence + attention mask for BiLSTM.

    Each video's data_dir should contain flow_haar_hh.npy of shape (T-1, 256, 256).
    Videos missing this file or with too few flow frames are filtered out.
    """

    def __init__(self, manifest_path, max_flow_frames=63, min_flow_frames=8,
                 image_size=128, augment=False):
        all_samples = load_manifest(manifest_path)

        # Filter: only keep videos with flow_haar_hh.npy and enough frames
        self.samples = []
        for row in all_samples:
            flow_path = os.path.join(row['data_dir'], 'flow_haar_hh.npy')
            if os.path.exists(flow_path):
                n = int(row['n_frames'])
                if n - 1 >= min_flow_frames:
                    self.samples.append(row)

        self.max_flow_frames = max_flow_frames
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        data_dir = row['data_dir']
        label = int(row['label'])

        # Load pre-computed flow HH wavelet: (T-1, 256, 256) float32
        flow_hh = np.load(os.path.join(data_dir, 'flow_haar_hh.npy'), mmap_mode='r')
        actual_frames = min(flow_hh.shape[0], self.max_flow_frames)

        # Decide augmentation once per video (temporal coherence)
        do_hflip = self.augment and torch.rand(1).item() < 0.5

        frames = []
        for i in range(actual_frames):
            h = flow_hh[i].copy()  # (256, 256)

            # Min-max normalize to [0, 1]
            h_min, h_max = h.min(), h.max()
            if h_max - h_min > 1e-8:
                h = (h - h_min) / (h_max - h_min)
            else:
                h = np.zeros_like(h)

            # Resize to image_size x image_size
            h_img = Image.fromarray((h * 255).astype(np.uint8), mode='L')
            h_img = h_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            h_t = TF.to_tensor(h_img)  # (1, 128, 128) float [0, 1]

            # Consistent horizontal flip across all frames
            if do_hflip:
                h_t = TF.hflip(h_t)

            frames.append(h_t)

        # Pad to max_flow_frames with zeros
        attention_mask = torch.zeros(self.max_flow_frames, dtype=torch.bool)
        attention_mask[:actual_frames] = True

        while len(frames) < self.max_flow_frames:
            frames.append(torch.zeros(1, self.image_size, self.image_size))

        frames_tensor = torch.stack(frames)  # (max_flow_frames, 1, 128, 128)

        return {
            'flow_hh': frames_tensor,                                    # (63, 1, 128, 128)
            'attention_mask': attention_mask,                             # (63,)
            'label': torch.tensor(label, dtype=torch.float32),           # scalar
            'seq_len': torch.tensor(actual_frames, dtype=torch.long),    # scalar
        }


def make_weighted_sampler(manifest_path, min_flow_frames=8):
    """Create WeightedRandomSampler to oversample minority class (reals).
    Applies same filtering as TemporalMotionDataset."""
    all_samples = load_manifest(manifest_path)

    labels = []
    for row in all_samples:
        flow_path = os.path.join(row['data_dir'], 'flow_haar_hh.npy')
        if os.path.exists(flow_path):
            n = int(row['n_frames'])
            if n - 1 >= min_flow_frames:
                labels.append(int(row['label']))

    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    total = n_real + n_fake

    w_real = total / (2 * n_real) if n_real > 0 else 1.0
    w_fake = total / (2 * n_fake) if n_fake > 0 else 1.0

    weights = [w_real if l == 0 else w_fake for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
