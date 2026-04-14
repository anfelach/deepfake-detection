"""
Datasets for Stage 1 (Sentry Gate) and Stage 2 (Spatial Fingerprint).
Both load pre-extracted .npy frames/wavelets from the processed directory.
Supports transparent GCS streaming with local disk cache when local files
have been deleted to save space.
"""
import os
import csv
import hashlib
import threading
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


# ============================================================
# GCS STREAMING CACHE
# ============================================================
LOCAL_ROOT = "/data/deepfake_pipeline/processed"
GCS_BUCKET = "udst-deepfake-video-data"
GCS_PREFIX = "processed"
GCS_CACHE_DIR = "/data/gcs_cache"
GCS_CACHE_MAX_GB = 1000  # Max cache size in GB (1 TB)

_gcs_fs = None
_gcs_lock = threading.Lock()


def _get_gcsfs():
    """Lazy-init gcsfs filesystem (thread-safe)."""
    global _gcs_fs
    if _gcs_fs is None:
        with _gcs_lock:
            if _gcs_fs is None:
                import gcsfs
                _gcs_fs = gcsfs.GCSFileSystem()
    return _gcs_fs


def _local_to_gcs_path(local_path):
    """Convert local data_dir path to GCS path.
    /data/deepfake_pipeline/processed/gen/video →
      udst-deepfake-video-data/processed/gen/gen/video
    (gcloud storage cp -r creates an extra nesting level)
    """
    rel = os.path.relpath(local_path, LOCAL_ROOT)
    # rel = "gen/video" — gcloud cp -r nests as gen/gen/video
    parts = rel.split(os.sep)
    if len(parts) >= 1:
        gen_name = parts[0]
        gcs_rel = os.path.join(gen_name, rel)
    else:
        gcs_rel = rel
    return f"{GCS_BUCKET}/{GCS_PREFIX}/{gcs_rel}"


def _get_cached_path(local_path, filename):
    """Get or download a file from GCS to local cache. Returns cached file path."""
    rel = os.path.relpath(local_path, LOCAL_ROOT)
    cache_path = os.path.join(GCS_CACHE_DIR, rel, filename)

    if os.path.exists(cache_path):
        return cache_path

    # Download from GCS
    gcs_path = f"{_local_to_gcs_path(local_path)}/{filename}"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    fs = _get_gcsfs()
    tmp_path = cache_path + f".tmp.{os.getpid()}"
    try:
        fs.get(gcs_path, tmp_path)
        os.rename(tmp_path, cache_path)  # Atomic on same filesystem
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return cache_path


def load_npy(data_dir, filename, mmap_mode='r'):
    """Load .npy file — from local disk if available, otherwise stream from GCS cache."""
    local_file = os.path.join(data_dir, filename)

    if os.path.exists(local_file):
        return np.load(local_file, mmap_mode=mmap_mode)

    # File not local — fetch from GCS cache
    cached = _get_cached_path(data_dir, filename)
    return np.load(cached, mmap_mode=mmap_mode)


def load_manifest(csv_path):
    """Load manifest CSV into list of dicts."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))


# ============================================================
# STAGE 1: SENTRY GATE DATASET
# ============================================================
class SentryGateDataset(Dataset):
    """
    Stage 1 dataset: 64 sampled frames, RGB + HH1 subband.
    Returns resized 224x224 tensors for DINOv3 input.
    """

    def __init__(self, manifest_path, image_size=224, augment=False):
        self.samples = load_manifest(manifest_path)
        self.image_size = image_size
        self.augment = augment

        # DINOv3 normalization (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        data_dir = row['data_dir']
        label = int(row['label'])
        cluster_raw = row['cluster']
        cluster = int(cluster_raw) - 1 if cluster_raw not in ('', 'None', None) else -1  # 1-indexed → 0-indexed
        n_frames = int(row['n_frames'])

        # Load arrays (local or GCS-cached)
        rgb = load_npy(data_dir, 'frames_rgb.npy')  # (N, 512, 512, 3) uint8
        hh1 = load_npy(data_dir, 'haar_hh1.npy')    # (N, 256, 256) float32

        actual_frames = rgb.shape[0]

        # Sample exactly 64 frame indices (repeat if fewer)
        if actual_frames >= 64:
            indices = np.linspace(0, actual_frames - 1, 64, dtype=int)
        else:
            # Repeat frames to fill 64
            indices = np.arange(actual_frames)
            indices = np.pad(indices, (0, 64 - actual_frames), mode='wrap')

        # Process RGB frames → (64, 3, 224, 224)
        rgb_frames = []
        for i in indices:
            img = Image.fromarray(rgb[i])
            if img.size[0] != self.image_size or img.size[1] != self.image_size:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            t = TF.to_tensor(img)  # (3, 224, 224) float [0,1]

            if self.augment:
                if torch.rand(1).item() < 0.5:
                    t = TF.hflip(t)

            t = (t - self.mean) / self.std  # ImageNet normalize
            rgb_frames.append(t)
        rgb_tensor = torch.stack(rgb_frames)  # (64, 3, 224, 224)

        # Process HH1 frames → replicate to 3ch → (64, 3, 224, 224)
        hh1_frames = []
        for i in indices:
            h = hh1[i]  # (256, 256) float32
            # Normalize HH1 to [0,1] range
            h_min, h_max = h.min(), h.max()
            if h_max - h_min > 1e-8:
                h = (h - h_min) / (h_max - h_min)
            else:
                h = np.zeros_like(h)
            # Resize to 224x224
            h_img = Image.fromarray((h * 255).astype(np.uint8), mode='L')
            h_img = h_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            h_t = TF.to_tensor(h_img)  # (1, 224, 224)
            h_t = h_t.repeat(3, 1, 1)  # (3, 224, 224) replicate to 3ch

            if self.augment:
                if torch.rand(1).item() < 0.5:
                    h_t = TF.hflip(h_t)

            h_t = (h_t - self.mean) / self.std
            hh1_frames.append(h_t)
        hh1_tensor = torch.stack(hh1_frames)  # (64, 3, 224, 224)

        return {
            'rgb': rgb_tensor,       # (64, 3, 224, 224)
            'hh1': hh1_tensor,       # (64, 3, 224, 224)
            'label': torch.tensor(label, dtype=torch.float32),
            'cluster': torch.tensor(cluster, dtype=torch.long),
            'n_frames': n_frames,
        }


# ============================================================
# STAGE 2: SPATIAL FINGERPRINT DATASET
# ============================================================
class SpatialFingerprintDataset(Dataset):
    """
    Stage 2 dataset: ALL frames (up to max_frames), dual RGB + HH1 input.
    RGB for DINOv3's pretrained visual features, HH1 for forensic fingerprints.
    Applies smart Haar augmentations on HH1 stream during training.
    Pads shorter videos with zeros + attention mask.
    """

    def __init__(self, manifest_path, max_frames=64, image_size=224, augment=False):
        self.samples = load_manifest(manifest_path)
        self.max_frames = max_frames
        self.image_size = image_size
        self.augment = augment

        # DINOv3 normalization (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _haar_smart_augment(self, hh1_frame, frame_idx, data_dir):
        """
        Smart Haar augmentation:
        1. 50% chance: blend in HL1 or LH1 band
        2. 30% chance: Gaussian noise
        3. 20% chance: random intensity scaling
        4. Random h/v flip
        """
        h = hh1_frame.copy()

        if torch.rand(1).item() < 0.5:
            try:
                alt_band = 'haar_hl1.npy' if torch.rand(1).item() < 0.5 else 'haar_lh1.npy'
                alt = load_npy(data_dir, alt_band)
                alt_idx = min(frame_idx, alt.shape[0] - 1)
                blend_factor = 0.1 + 0.3 * torch.rand(1).item()
                h = h * (1 - blend_factor) + alt[alt_idx] * blend_factor
            except Exception:
                pass

        if torch.rand(1).item() < 0.3:
            noise_scale = 0.03 * (h.std() if h.std() > 0 else 0.01)
            h = h + np.random.randn(*h.shape).astype(np.float32) * noise_scale

        if torch.rand(1).item() < 0.2:
            scale = 0.8 + 0.4 * torch.rand(1).item()
            h = h * scale

        if torch.rand(1).item() < 0.5:
            h = np.flip(h, axis=1).copy()
        if torch.rand(1).item() < 0.5:
            h = np.flip(h, axis=0).copy()

        return h

    def __getitem__(self, idx):
        row = self.samples[idx]
        data_dir = row['data_dir']
        label = int(row['label'])

        # Load both RGB and HH1 (local or GCS-cached)
        rgb = load_npy(data_dir, 'frames_rgb.npy')  # (N, H, W, 3)
        hh1 = load_npy(data_dir, 'haar_hh1.npy')    # (N, 256, 256)
        total_frames = min(rgb.shape[0], hh1.shape[0])

        # If video has more frames than max_frames, uniformly subsample
        if total_frames > self.max_frames:
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
        actual_frames = len(indices)

        rgb_frames = []
        hh1_frames = []
        for i in indices:
            # --- RGB stream ---
            img = Image.fromarray(rgb[i])
            if img.size[0] != self.image_size or img.size[1] != self.image_size:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            rgb_t = TF.to_tensor(img)  # (3, 224, 224) float [0,1]

            # --- HH1 stream ---
            h = hh1[i].copy()
            if self.augment:
                h = self._haar_smart_augment(h, int(i), data_dir)

            h_min, h_max = h.min(), h.max()
            if h_max - h_min > 1e-8:
                h = (h - h_min) / (h_max - h_min)
            else:
                h = np.zeros_like(h)
            h_img = Image.fromarray((h * 255).astype(np.uint8), mode='L')
            h_img = h_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            hh1_t = TF.to_tensor(h_img).repeat(3, 1, 1)  # (3, 224, 224)

            # Augment: synchronized horizontal flip on both streams
            if self.augment and torch.rand(1).item() < 0.5:
                rgb_t = TF.hflip(rgb_t)
                hh1_t = TF.hflip(hh1_t)

            # Normalize both with ImageNet stats
            rgb_t = (rgb_t - self.mean) / self.std
            hh1_t = (hh1_t - self.mean) / self.std

            rgb_frames.append(rgb_t)
            hh1_frames.append(hh1_t)

        # Pad to max_frames with zeros
        attention_mask = torch.zeros(self.max_frames, dtype=torch.bool)
        attention_mask[:actual_frames] = True

        while len(rgb_frames) < self.max_frames:
            rgb_frames.append(torch.zeros(3, self.image_size, self.image_size))
            hh1_frames.append(torch.zeros(3, self.image_size, self.image_size))

        rgb_tensor = torch.stack(rgb_frames)   # (max_frames, 3, 224, 224)
        hh1_tensor = torch.stack(hh1_frames)   # (max_frames, 3, 224, 224)

        return {
            'rgb': rgb_tensor,               # (max_frames, 3, 224, 224)
            'hh1': hh1_tensor,               # (max_frames, 3, 224, 224)
            'attention_mask': attention_mask,  # (max_frames,)
            'label': torch.tensor(label, dtype=torch.float32),
        }


# ============================================================
# SAMPLER FOR CLASS IMBALANCE
# ============================================================
def make_weighted_sampler(manifest_path):
    """Create WeightedRandomSampler to oversample minority class (reals)."""
    samples = load_manifest(manifest_path)
    labels = [int(s['label']) for s in samples]

    # Count per class
    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    total = n_real + n_fake

    # Weight inversely proportional to class frequency
    w_real = total / (2 * n_real) if n_real > 0 else 1.0
    w_fake = total / (2 * n_fake) if n_fake > 0 else 1.0

    weights = [w_real if l == 0 else w_fake for l in labels]

    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)