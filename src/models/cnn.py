"""Model 1 - CNN baseline (Rohan).

Four blocks of [3x3 conv -> BatchNorm -> ReLU -> 2x2 max-pool] with 32/64/128/256
channels, then global average pooling, dropout 0.3 and a 10-way linear layer.
Input (B, 1, 128, 201) -> output (B, 10) logits.
"""
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    def __init__(self, n_classes=10, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        chans = (1,) + tuple(channels)
        self.features = nn.Sequential(*[block(chans[i], chans[i + 1]) for i in range(len(channels))])
        self.pool = nn.AdaptiveAvgPool2d(1)          # global average pooling
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))
