"""
Central training configuration for Stage 1 (Sentry Gate) and Stage 2 (Spatial Fingerprint).
"""
from dataclasses import dataclass, field
from typing import Optional, Dict
import json
import os

# ============================================================
# PATHS
# ============================================================
MANIFEST_TRAIN = "/data/deepfake_pipeline/splits/manifest_train.csv"
MANIFEST_VAL = "/data/deepfake_pipeline/splits/manifest_val.csv"
MANIFEST_TEST = "/data/deepfake_pipeline/splits/manifest_test.csv"
HAAR_JSD_RESULTS = "/data/haar_analysis/results/haar_jsd_results.json"
CLUSTER_RESULTS = "/data/haar_analysis/results/generator_clusters.json"

# DINOv3
DINOV3_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_TOKEN = open(os.path.join(os.path.dirname(__file__), '..', 'secrets.txt')).read().split('=', 1)[1].strip()

# GCS Streaming
GCS_BUCKET = "udst-deepfake-video-data"
GCS_PREFIX = "processed"
GCS_CACHE_DIR = "/data/gcs_cache"
GCS_CACHE_MAX_GB = 1000  # 1 TB local cache

# ============================================================
# HAAR JSD PER-GENERATOR LOOKUP
# ============================================================
def load_haar_jsd():
    """Load per-generator Haar JSD results for generator-aware training."""
    if os.path.exists(HAAR_JSD_RESULTS):
        with open(HAAR_JSD_RESULTS) as f:
            return json.load(f)
    return {}

def load_cluster_assignments():
    """Load cluster assignments per generator."""
    if os.path.exists(CLUSTER_RESULTS):
        with open(CLUSTER_RESULTS) as f:
            return json.load(f)
    return {}

# Map from manifest generator names to Haar analysis keys
GENERATOR_TO_HAAR_KEY = {
    "dvf_opensora": "DVF_opensora",
    "dvf_pika": "DVF_pika",
    "dvf_sora": "DVF_sora",
    "dvf_stablediffusion": "DVF_stablediffusion",
    "dvf_stablevideo": "DVF_stablevideo",
    "dvf_stablevideodiffusion": "DVF_stablevideodiffusion",
    "dvf_videocrafter1": "DVF_videocrafter1",
    "dvf_zeroscope": "DVF_zeroscope",
    "deepaction_animatediff": "deepaction_BDAnimateDiffLightning",
    "deepaction_cogvideox5b": "deepaction_CogVideoX5B",
    "deepaction_runwayml": "deepaction_RunwayML",
    "deepaction_stablediffusion": "deepaction_StableDiffusion",
    "deepaction_veo": "deepaction_Veo",
    "deepaction_videopoet": "deepaction_VideoPoet",
    "waverep_allegro": "WaveRep_allegro_h264",
    "waverep_cogvideox15": "WaveRep_cogvideox15_h264",
    "waverep_flux": "WaveRep_flux_web",
    "waverep_mochi1": "WaveRep_mochi1_h264",
    "waverep_nova": "WaveRep_nove_h264",
    "waverep_opensoraplan": "WaveRep_opensoraplan_h264",
    "waverep_pyramid": "WaveRep_pyramid_h264",
    "waverep_sora": "WaveRep_sora_web",
}


@dataclass
class Stage1Config:
    """Stage 1: Sentry Gate — frozen DINOv3 + lightweight heads."""
    # Paths
    output_dir: str = "/data/code/runs/stage1"

    # Data
    frames_per_video: int = 64
    image_size: int = 224
    use_rgb: bool = True
    use_haar_hh1: bool = True  # Stack HH1 as separate input stream

    # Model
    dinov3_model: str = DINOV3_MODEL_NAME
    hf_token: str = HF_TOKEN
    hidden_size: int = 768        # DINOv3 ViT-B output dim
    freeze_backbone: bool = True  # Sentry gate = frozen backbone
    num_clusters: int = 3

    # Training
    epochs: int = 30
    batch_size: int = 8           # per GPU (8 videos * 64 frames each)
    lr: float = 1e-3              # Higher LR since only heads train
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    lambda_cluster: float = 0.3   # Cluster probe loss weight

    # System
    num_workers: int = 4
    use_amp: bool = True
    seed: int = 42
    patience: int = 10
    save_every: int = 1
    gradient_clip: float = 1.0


@dataclass
class Stage2Config:
    """Stage 2: DINOv3 Spatial Fingerprint — fine-tuned with Haar augmentations."""
    # Paths
    output_dir: str = "/data/code/runs/stage2"

    # Data
    max_frames: int = 256         # Cap at 256 frames (covers 75th percentile); pad shorter videos
    image_size: int = 224
    use_haar_hh1: bool = True     # Primary input: HH1 subbands
    use_haar_augmentation: bool = True  # Smart augmentation from methodology

    # Model
    dinov3_model: str = DINOV3_MODEL_NAME
    hf_token: str = HF_TOKEN
    hidden_size: int = 768
    freeze_epochs: int = 10       # Freeze backbone longer — let heads converge first
    unfreeze_last_n_blocks: int = 2  # Only last 2 blocks — less risk of overfitting
    llrd_factor: float = 0.5      # More aggressive decay — protect earlier layers

    # Training
    epochs: int = 40
    batch_size: int = 64          # per GPU — dual RGB+HH1, batch=128 OOMs CPU workers
    lr_head: float = 1e-3
    lr_backbone: float = 1e-6     # 10x lower backbone LR — gentler fine-tuning
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    gradient_accumulation: int = 1  # Effective batch = 64*4GPUs = 256

    # System
    num_workers: int = 4          # Balanced for dual-stream loading
    use_amp: bool = True
    seed: int = 42
    patience: int = 7
    save_every: int = 1
    gradient_clip: float = 1.0