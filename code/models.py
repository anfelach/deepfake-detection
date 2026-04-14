"""
Models for deepfake detection:
  Stage 1: SentryGateModel — frozen DINOv3 + lightweight binary/cluster heads
  Stage 2: SpatialFingerprintModel — dual RGB+HH1 DINOv3 + temporal attention pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ============================================================
# TEMPORAL ATTENTION POOLING
# ============================================================
class TemporalAttentionPool(nn.Module):
    """Learned attention weights over per-frame features with attention mask."""

    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, 1),
        )

    def forward(self, x, mask=None):
        # x: (B, T, D), mask: (B, T)
        scores = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, D)


# ============================================================
# STAGE 1: SENTRY GATE MODEL
# ============================================================
class SentryGateModel(nn.Module):
    """
    Stage 1: Frozen DINOv3 backbone + lightweight heads.
    Dual-stream: processes RGB and HH1 separately through same backbone,
    concatenates CLS tokens → binary classifier + cluster probe.
    """

    def __init__(self, model_name, hf_token, hidden_size=768,
                 num_clusters=3, dropout=0.1, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, token=hf_token)
        self.hidden_size = hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = 2 * hidden_size  # 1536 (RGB CLS + HH1 CLS)

        # Binary head: real vs fake
        self.binary_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

        # Cluster probe head: predict generator cluster
        self.cluster_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_clusters),
        )

    def _extract_cls(self, frames):
        """Extract CLS token per frame. frames: (B, T, 3, H, W)"""
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)

        with torch.cuda.amp.autocast():
            out = self.backbone(flat).last_hidden_state[:, 0]  # CLS token

        return out.reshape(B, T, -1)  # (B, T, hidden_size)

    def forward(self, rgb, hh1):
        """
        rgb: (B, 64, 3, 224, 224)
        hh1: (B, 64, 3, 224, 224)
        Returns: binary_logits (B, 1), cluster_logits (B, num_clusters)
        """
        rgb_cls = self._extract_cls(rgb)   # (B, 64, 768)
        hh1_cls = self._extract_cls(hh1)   # (B, 64, 768)

        # Mean pool over frames, then concatenate streams
        rgb_pool = rgb_cls.mean(dim=1)  # (B, 768)
        hh1_pool = hh1_cls.mean(dim=1)  # (B, 768)
        combined = torch.cat([rgb_pool, hh1_pool], dim=1)  # (B, 1536)

        binary_logits = self.binary_head(combined)    # (B, 1)
        cluster_logits = self.cluster_head(combined)  # (B, num_clusters)

        return binary_logits, cluster_logits


# ============================================================
# STAGE 2: SPATIAL FINGERPRINT MODEL
# ============================================================
class SpatialFingerprintModel(nn.Module):
    """
    Stage 2: Dual-stream DINOv3 (RGB + HH1) with temporal attention pooling.
    RGB captures pretrained visual features, HH1 captures forensic fingerprints.
    CLS tokens from both streams are concatenated → temporal attention → classifier.
    Supports progressive unfreezing of backbone.
    """

    def __init__(self, model_name, hf_token, hidden_size=768, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, token=hf_token)
        self.hidden_size = hidden_size

        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        feat_dim = 2 * hidden_size  # 1536

        self.temporal_pool = TemporalAttentionPool(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def _get_encoder_layers(self):
        """Find encoder layers — DINOv3 uses model.layer (not encoder.layer or blocks)."""
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            return self.backbone.encoder.layer
        elif hasattr(self.backbone, 'layer'):
            return self.backbone.layer
        elif hasattr(self.backbone, 'blocks'):
            return self.backbone.blocks
        return None

    def unfreeze_last_n_blocks(self, n):
        """Unfreeze the last N transformer blocks of the backbone."""
        encoder_layers = self._get_encoder_layers()
        if encoder_layers is None:
            return 0

        total = len(encoder_layers)
        unfrozen = 0
        for i in range(total - n, total):
            if i >= 0:
                for p in encoder_layers[i].parameters():
                    p.requires_grad = True
                    unfrozen += p.numel()

        # Also unfreeze layernorm
        if hasattr(self.backbone, 'layernorm'):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True
                unfrozen += p.numel()
        elif hasattr(self.backbone, 'norm'):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
                unfrozen += p.numel()

        return unfrozen

    def get_param_groups(self, lr_head, lr_backbone, llrd_factor=0.5):
        """Get parameter groups with layer-wise learning rate decay (LLRD)."""
        head_params = list(self.temporal_pool.parameters()) + list(self.classifier.parameters())

        encoder_layers = self._get_encoder_layers()
        backbone_groups = []
        if encoder_layers is not None:
            n_layers = len(encoder_layers)
            for i, layer in enumerate(encoder_layers):
                trainable = [p for p in layer.parameters() if p.requires_grad]
                if trainable:
                    depth = n_layers - 1 - i  # deeper layers get higher LR
                    lr = lr_backbone * (llrd_factor ** depth)
                    backbone_groups.append({'params': trainable, 'lr': lr})

        # Layernorm at full backbone LR
        for name in ['layernorm', 'norm']:
            if hasattr(self.backbone, name):
                trainable = [p for p in getattr(self.backbone, name).parameters() if p.requires_grad]
                if trainable:
                    backbone_groups.append({'params': trainable, 'lr': lr_backbone})

        return [{'params': head_params, 'lr': lr_head}] + backbone_groups

    def _extract_cls_chunked(self, frames, chunk_size):
        """Extract CLS tokens in chunks to manage GPU memory.
        frames: (B, T, 3, H, W) → (B, T, hidden_size)
        """
        B, T, C, H, W = frames.shape
        cls_list = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = frames[:, start:end].reshape(-1, C, H, W)  # (B*chunk, 3, H, W)
            with torch.cuda.amp.autocast():
                out = self.backbone(chunk).last_hidden_state[:, 0]
            cls_list.append(out.reshape(B, end - start, -1))
        return torch.cat(cls_list, dim=1)  # (B, T, hidden_size)

    def forward(self, rgb, hh1, attention_mask=None):
        """
        rgb: (B, T, 3, 224, 224)
        hh1: (B, T, 3, 224, 224)
        attention_mask: (B, T) bool
        Returns: logits (B, 1)
        """
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        chunk_size = 256 if backbone_frozen else 16

        rgb_cls = self._extract_cls_chunked(rgb, chunk_size)   # (B, T, 768)
        hh1_cls = self._extract_cls_chunked(hh1, chunk_size)   # (B, T, 768)

        # Concatenate streams: (B, T, 1536)
        combined = torch.cat([rgb_cls, hh1_cls], dim=-1)

        # Temporal attention pooling with mask
        pooled = self.temporal_pool(combined, attention_mask)  # (B, 1536)

        return self.classifier(pooled)  # (B, 1)