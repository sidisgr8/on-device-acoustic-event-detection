"""Model 2 - CRNN: CNN feature extractor + bidirectional GRU (Akshdeep).

Input (B, 1, 128, 201):
  3 conv blocks (3x3 conv, BatchNorm, ReLU) with 32/64/128 channels and pooling
  (2,2), (2,2), (2,1) over (frequency, time)   ->  (B, 128, 16, 50)
  flatten channels x frequency per time step   ->  (B, 50, 2048)
  linear 2048 -> 128, then 1-layer bidirectional GRU (hidden 128) -> (B, 50, 256)
  mean-pool and max-pool over time, concatenated -> (B, 512)
  dropout 0.3, linear 512 -> 10 logits.
A GRU has fewer parameters than an LSTM (efficiency goal); set rnn="lstm" for the ablation.
"""
import torch
import torch.nn as nn


def conv_block(c_in, c_out, pool):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool),
    )


class CRNN(nn.Module):
    def __init__(self, n_classes=10, n_mels=128, hidden=128, dropout=0.3, rnn="gru"):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(1, 32, (2, 2)),
            conv_block(32, 64, (2, 2)),
            conv_block(64, 128, (2, 1)),     # pool frequency only: keeps 50 time steps
        )
        feat = 128 * (n_mels // 8)           # channels x remaining frequency bins = 2048
        self.proj = nn.Linear(feat, hidden)
        rnn_cls = nn.GRU if rnn == "gru" else nn.LSTM
        self.rnn = rnn_cls(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x):
        x = self.conv(x)                                   # (B, C, F, T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)     # (B, T, C*F)
        x, _ = self.rnn(torch.relu(self.proj(x)))          # (B, T, 2*hidden)
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.fc(self.drop(x))
