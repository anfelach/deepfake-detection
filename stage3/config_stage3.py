"""
Configuration for Stage 3: Temporal Motion Analysis — CNN + BiLSTM on optical flow wavelets.
"""
from dataclasses import dataclass

# ============================================================
# PATHS
# ============================================================
MANIFEST_TRAIN = "/data/deepfake_pipeline/splits/manifest_train.csv"
MANIFEST_VAL = "/data/deepfake_pipeline/splits/manifest_val.csv"
MANIFEST_TEST = "/data/deepfake_pipeline/splits/manifest_test.csv"


@dataclass
class Stage3Config:
    """Stage 3: Temporal Motion — CNN + BiLSTM on optical flow Haar HH wavelets."""
    # Paths
    output_dir: str = "/data/code/runs/stage3"

    # Data
    max_flow_frames: int = 63        # N-1 flow fields from N RGB frames (max 64 → 63)
    min_flow_frames: int = 8         # Filter out videos with fewer usable flow frames
    flow_image_size: int = 128       # Resize HH flow from 256 → 128 for CNN input

    # CNN feature extractor
    cnn_channels: tuple = (1, 32, 64, 128)
    cnn_feature_dim: int = 256       # Output dim after CNN + adaptive pool + FC

    # BiLSTM
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3

    # Classification head
    classifier_dropout: float = 0.4

    # Training
    epochs: int = 40
    batch_size: int = 16             # per GPU — flow wavelets are small (1ch, 128x128)
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 3

    # System
    num_workers: int = 8
    use_amp: bool = True
    seed: int = 42
    patience: int = 10
    save_every: int = 1
    gradient_clip: float = 1.0
