"""CNN-only PTB-XL baseline: the shared lead-image CNN with a pooled linear readout.

The spatial encoder is constructed exactly as in the CNN-LSTM, temporal-convolution
and hybrid ViT models. The 70 feature columns are averaged with no sequence module,
so this model is the "no temporal/sequence head" reference for those three models.
"""
from __future__ import annotations

import math

from torch import nn

INPUT_SHAPE = (12, 120, 1120)
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
HEAD_CONFIGURATION = {"feature_columns": 70, "feature_dim": 96, "sequence_module": "none",
                      "aggregation": "mean over the 70 pooled feature columns", "classifier": "Linear 96->5"}


def spatial_feature_sequence(feature_map):
    """N,C,R,W -> N,W,C*R: column order, then channels and vertical bands."""
    if feature_map.ndim != 4:
        raise ValueError("Expected feature map N,C,R,W")
    return feature_map.permute(0, 3, 1, 2).contiguous().flatten(2)


class LeadImageCNNPool(nn.Module):
    """Identical lead-image CNN, column mean, dropout, five logits."""
    def __init__(self, dropout=0.2):
        super().__init__()
        if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be finite in [0,1)")
        self.features = nn.Sequential(
            nn.Conv2d(12, 12, 5, stride=(2, 4), padding=2, groups=12, bias=False),
            nn.Conv2d(12, 8, 1, bias=False), nn.GroupNorm(4, 8), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8, bias=False),
            nn.Conv2d(8, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 24, 1, bias=False), nn.GroupNorm(4, 24), nn.ReLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 70))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(96, 5)

    def extract_sequence(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected image batch N x {INPUT_SHAPE}")
        return spatial_feature_sequence(self.spatial_pool(self.features(images)))

    def forward(self, images):
        return self.classifier(self.dropout(self.extract_sequence(images).mean(dim=1)))
