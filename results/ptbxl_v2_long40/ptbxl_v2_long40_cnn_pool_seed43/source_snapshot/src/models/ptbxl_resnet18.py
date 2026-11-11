"""ImageNet-pretrained ResNet-18 on the lead-image pages (protocol amendment of 11 Nov 2026, part B).

A standard, large image backbone outside the controlled head comparison. It asks whether the adaptive-gain
cue and the benefit of fixed gain also hold for a conventional pretrained network. Input is the same
12 x 120 x 1120 standardized ink tensor as every other network. The 12 lead rows are stacked vertically into
one 1440 x 1120 single-channel image, without resizing. The ImageNet RGB filters of the first convolution are
summed into one channel, which is the same as feeding the image to all three channels. The layers mirror
torchvision's resnet18, with the same parameter names, so the official weights load with strict key matching.
torchvision itself is not needed. The 1000-class layer is replaced by dropout and a linear 512 -> 5 readout.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from src.models.ptbxl_cnn_pool import INPUT_SHAPE

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models/pretrained/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
HEAD_CONFIGURATION = {"backbone": "ResNet-18 (ImageNet-1k, torchvision resnet18-f37072fd)",
                      "input": "12 lead rows stacked to 1 x 1440 x 1120; conv1 RGB filters summed",
                      "aggregation": "global average pooling", "classifier": "dropout, Linear 512->5"}


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + identity)


def verified_imagenet_state(path=WEIGHTS):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"Unexpected ResNet-18 weights hash {digest}")
    return torch.load(path, map_location="cpu", weights_only=True)


class LeadImageResNet18(nn.Module):
    """ResNet-18 on the stacked lead rows; five logits."""
    pretrained_weights = "ImageNet-1k (torchvision resnet18-f37072fd.pth)"

    def __init__(self, dropout=0.2, pretrained=True):
        super().__init__()
        if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout must be finite in [0,1)")
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, 5)
        if pretrained:
            state = verified_imagenet_state()
            state["conv1.weight"] = state["conv1.weight"].sum(dim=1, keepdim=True)
            state = {key: value for key, value in state.items() if not key.startswith("fc.")}
            missing, unexpected = self.load_state_dict(state, strict=False)
            if unexpected or sorted(missing) != ["classifier.bias", "classifier.weight"]:
                raise ValueError(f"ImageNet weights do not match: missing={missing}, unexpected={unexpected}")

        self.to(memory_format=torch.channels_last)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected image batch N x {INPUT_SHAPE}")
        x = images.reshape(len(images), 1, INPUT_SHAPE[0] * INPUT_SHAPE[1], INPUT_SHAPE[2])
        x = x.contiguous(memory_format=torch.channels_last)
        # bfloat16 mixed precision on the GPU only (amendment of 11 Nov, part B): at 1440 x 1120 pixels a float32
        # epoch takes about 19 min. Weights, optimizer and loss stay float32, and the logits are returned as float32.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            logits = self.classifier(self.dropout(torch.flatten(self.avgpool(x), 1)))
        return logits.float()
