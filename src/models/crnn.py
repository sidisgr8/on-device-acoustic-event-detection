"""Model 2: CRNN (interim spec, section 4.3)."""
import torch
from torch import nn


def _block(cin, cout, pool):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class CRNN(nn.Module):
    """3 conv blocks (32/64/128; pool (2,2),(2,2),(2,1) over (freq,time)) -> per-step vector
    (channels x freq) -> linear 128 -> 1-layer BiGRU(128) -> [mean ; max] over time -> dropout ->
    linear. Input (B, 1, 128, 201). The projection is a plain linear map (no activation)."""

    def __init__(self, n_classes=10, channels=(32, 64, 128), pools=((2, 2), (2, 2), (2, 1)),
                 proj=128, hidden=128, dropout=0.3, n_mels=128):
        super().__init__()
        chans = (1, *channels)
        self.convs = nn.Sequential(*[_block(a, b, p) for a, b, p in zip(chans[:-1], chans[1:], pools)])
        f_out = n_mels
        for p in pools:
            f_out //= p[0]
        self.proj = nn.Linear(channels[-1] * f_out, proj)
        self.gru = nn.GRU(proj, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x, return_shapes=False):
        shapes = {"input": tuple(x.shape)}
        x = self.convs(x)
        shapes["after_conv"] = tuple(x.shape)                       # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)               # (B, T', C*F')
        shapes["flattened_per_step"] = tuple(x.shape)
        x = self.proj(x)
        shapes["projected"] = tuple(x.shape)
        x, _ = self.gru(x)
        shapes["gru_out"] = tuple(x.shape)
        x = torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)
        shapes["pooled"] = tuple(x.shape)
        out = self.head(x)
        return (out, shapes) if return_shapes else out
