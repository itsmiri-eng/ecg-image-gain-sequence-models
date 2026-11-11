"""Capacity-matched CNN-only head (protocol amendment of 3 Nov 2026): more parameters, still no sequence mixing.

Same shared lead-image CNN as every other network. Each of the 70 feature columns passes through the
same two-layer MLP on its own (96 -> 96 -> 64), the columns are averaged, and a linear layer gives five
logits. No layer sees two columns together before the mean, so the head cannot relate one part of a
row to another (for example the calibration pulse to the waveform). Total 17,065 parameters, between
the temporal-convolution (17,161) and CNN-LSTM (18,025) models; it separates "more capacity" from
"sequence modelling" in the head comparison.
"""
from __future__ import annotations

import math

from torch import nn

from src.models.ptbxl_cnn_pool import INPUT_SHAPE, spatial_feature_sequence

HEAD_CONFIGURATION = {"feature_columns": 70, "feature_dim": 96, "sequence_module": "none",
                      "column_mlp": "Linear 96->96, ReLU, dropout, Linear 96->64, ReLU (shared, per column)",
                      "aggregation": "mean over the 70 columns after the MLP", "classifier": "Linear 64->5"}


class LeadImageCNNPoolWide(nn.Module):
    """Identical lead-image CNN, per-column MLP, column mean, dropout, five logits."""
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
        self.column_mlp = nn.Sequential(nn.Linear(96, 96), nn.ReLU(), nn.Dropout(dropout), nn.Linear(96, 64), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 5)

    def extract_sequence(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected image batch N x {INPUT_SHAPE}")
        return spatial_feature_sequence(self.spatial_pool(self.features(images)))

    def forward(self, images):
        return self.classifier(self.dropout(self.column_mlp(self.extract_sequence(images)).mean(dim=1)))
