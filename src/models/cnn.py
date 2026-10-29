"""Model 1a: CNN baseline (interim spec, section 4.2)."""
import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class CNNBaseline(nn.Module):
    """4 x (conv3x3-BN-ReLU-maxpool2) with 32/64/128/256 channels, global average pool,
    dropout, 10-way linear. Input (B, 1, 128, 201)."""

    def __init__(self, n_classes=10, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        chans = (1, *channels)
        self.features = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[-1], n_classes))

    def forward(self, x):
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))
