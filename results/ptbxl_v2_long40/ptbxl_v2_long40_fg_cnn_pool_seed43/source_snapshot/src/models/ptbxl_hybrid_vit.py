"""Small hybrid vision Transformer over 70 aligned ECG image-feature columns.

The CNN matches the prior image models. Its columns are spatial feature tokens,
not waveform samples or raw-image patches. Global attention is noncausal.
"""
from __future__ import annotations

import math
import torch
from torch import nn

INPUT_SHAPE = (12, 120, 1120)
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
HEAD_CONFIGURATION = {
    "token_count": 70, "feature_dim": 96, "embed_dim": 32,
    "num_heads": 4, "num_layers": 2, "feedforward_dim": 64,
    "activation": "gelu", "norm_first": True, "causal_mask": False,
    "positional_encoding": "learned", "position_initialization_std": 0.02,
    "cls_token": False, "aggregation": "mean of final normalized tokens",
    "independent_layer_initialization": True,
}


def spatial_feature_sequence(feature_map):
    if feature_map.ndim != 4:
        raise ValueError("Expected feature map N,C,R,W")
    return feature_map.permute(0, 3, 1, 2).contiguous().flatten(2)


class LeadImageHybridViT(nn.Module):
    """Identical lead-image CNN, linear feature tokens, two global-attention blocks."""
    def __init__(self, dropout=0.2, embed_dim=32, num_heads=4, num_layers=2, feedforward_dim=64):
        super().__init__()
        if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be finite in [0,1)")
        for key, value in (("embed_dim", embed_dim), ("num_heads", num_heads),
                           ("num_layers", num_layers), ("feedforward_dim", feedforward_dim)):
            if type(value) is not int or value != HEAD_CONFIGURATION[key]:
                raise ValueError(f"Prespecified hybrid head requires {key}={HEAD_CONFIGURATION[key]}")
        self.features = nn.Sequential(
            nn.Conv2d(12, 12, 5, stride=(2, 4), padding=2, groups=12, bias=False),
            nn.Conv2d(12, 8, 1, bias=False), nn.GroupNorm(4, 8), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8, bias=False),
            nn.Conv2d(8, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 24, 1, bias=False), nn.GroupNorm(4, 24), nn.ReLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 70))
        self.projection = nn.Linear(96, embed_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, 70, embed_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.token_dropout = nn.Dropout(dropout)
        # Construct separately: TransformerEncoder clones one layer with identical initialization.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                dim_feedforward=feedforward_dim, dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 5)

    def extract_sequence(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected image batch N x {INPUT_SHAPE}")
        return spatial_feature_sequence(self.spatial_pool(self.features(images)))

    def encode_tokens(self, sequence):
        """Global attention over feature columns, with fixed spatial position indices."""
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (70, 96):
            raise ValueError("Expected feature sequence N x 70 x 96")
        tokens = self.token_dropout(self.projection(sequence) + self.position_embedding)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        return self.final_norm(tokens)

    def forward(self, images):
        tokens = self.encode_tokens(self.extract_sequence(images))
        return self.classifier(self.dropout(tokens.mean(dim=1)))
