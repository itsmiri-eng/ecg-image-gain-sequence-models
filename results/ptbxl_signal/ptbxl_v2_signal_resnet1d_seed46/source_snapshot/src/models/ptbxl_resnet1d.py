"""Wang-style 1D ResNet for 12-lead 100 Hz waveforms (signal-domain reference, protocol ptbxl_v2_signal_resnet1d).

Wang, Yan and Oates (IJCNN 2017): three residual blocks of 64, 128 and 128 channels; each block is
conv(k=8) -> conv(k=5) -> conv(k=3) with BatchNorm, ReLU between, a 1x1 projection shortcut when the
channel count changes, and ReLU after the sum. Global average pooling, dropout and a linear readout
give five independent logits.
"""
from torch import nn


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        layers = []
        for i, k in enumerate((8, 5, 3)):
            layers += [nn.Conv1d(cin if i == 0 else cout, cout, k, padding="same", bias=False), nn.BatchNorm1d(cout)]
            if i < 2:
                layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        self.shortcut = (nn.Identity() if cin == cout else
                         nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout)))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))


class ResNet1dWang(nn.Module):
    def __init__(self, dropout=0.2, leads=12, classes=5):
        super().__init__()
        self.blocks = nn.Sequential(_Block(leads, 64), _Block(64, 128), _Block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, classes))

    def forward(self, x):
        return self.head(self.blocks(x))
