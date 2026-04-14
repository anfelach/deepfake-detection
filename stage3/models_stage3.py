"""
Model for Stage 3: Temporal Motion Analysis.
CNN feature extractor + BiLSTM sequence model on optical flow Haar HH wavelets.
No DINOv3 backbone — lightweight, trained from scratch (~3M params).
"""
import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    """Learned attention weights over temporal sequence."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, frame_features, attention_mask=None):
        """
        frame_features: (B, T, D)
        attention_mask: (B, T) bool — True for valid frames
        Returns: (B, D)
        """
        attn_scores = self.attention(frame_features).squeeze(-1)  # (B, T)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
        pooled = torch.bmm(attn_weights.unsqueeze(1), frame_features).squeeze(1)  # (B, D)
        return pooled


class TemporalMotionModel(nn.Module):
    """
    CNN + BiLSTM for temporal motion analysis.
    Input: sequence of optical flow Haar HH wavelet frames (single channel).

    Architecture:
        (B, T, 1, 128, 128) → CNN → (B, T, 256) → BiLSTM → (B, T, 512)
        → Attention Pool → (B, 512) → Classifier → (B, 1)
    """

    def __init__(self, cnn_channels=(1, 32, 64, 128), cnn_feature_dim=256,
                 lstm_hidden_size=256, lstm_num_layers=2, lstm_dropout=0.3,
                 classifier_dropout=0.4):
        super().__init__()

        # CNN feature extractor: 3 conv blocks with stride=2
        cnn_layers = []
        for i in range(len(cnn_channels) - 1):
            cnn_layers.extend([
                nn.Conv2d(cnn_channels[i], cnn_channels[i + 1], 3, stride=2, padding=1),
                nn.BatchNorm2d(cnn_channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
        self.cnn = nn.Sequential(*cnn_layers)
        self.pool = nn.AdaptiveAvgPool2d(4)

        # CNN flat dim: last_channel * 4 * 4
        cnn_flat_dim = cnn_channels[-1] * 4 * 4  # 128 * 16 = 2048
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, cnn_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden_size * 2  # bidirectional → 512

        # Temporal attention pooling
        self.temporal_pool = TemporalAttentionPool(lstm_out_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 1),
        )

    def forward(self, flow_hh, attention_mask=None, seq_lengths=None):
        """
        flow_hh:        (B, T, 1, H, W)  e.g. (B, 63, 1, 128, 128)
        attention_mask:  (B, T) bool — True for valid flow frames
        seq_lengths:     (B,) long — actual sequence lengths for packing
        Returns:         logits (B, 1)
        """
        B, T, C, H, W = flow_hh.shape

        # CNN: flatten batch and time dims
        flat = flow_hh.view(B * T, C, H, W)       # (B*63, 1, 128, 128)
        feat = self.cnn(flat)                       # (B*63, 128, 16, 16)
        feat = self.pool(feat)                      # (B*63, 128, 4, 4)
        feat = feat.view(feat.size(0), -1)          # (B*63, 2048)
        feat = self.cnn_fc(feat)                    # (B*63, 256)

        # Reshape back to sequence
        feat = feat.view(B, T, -1)                  # (B, 63, 256)

        # BiLSTM with packing for variable-length sequences
        if seq_lengths is not None:
            seq_lengths_clamped = seq_lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                feat, seq_lengths_clamped.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )  # (B, T, 512)
        else:
            lstm_out, _ = self.lstm(feat)            # (B, T, 512)

        # Temporal attention pooling (masked)
        pooled = self.temporal_pool(lstm_out, attention_mask)  # (B, 512)

        # Classify
        logits = self.classifier(pooled)             # (B, 1)
        return logits
