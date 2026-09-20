"""Model 1b - MobileNetV3-Small, the 'efficient by design' variant (Rohan).

torchvision's MobileNetV3-Small with (a) a 1-channel input layer made by summing the
RGB filters of the first conv, and (b) a new 10-way classifier head.
pretrained=True starts from ImageNet weights (needs internet the first time).
"""
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet(n_classes=10, pretrained=False):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = mobilenet_v3_small(weights=weights)
    old = net.features[0][0]                                   # Conv2d(3, 16, 3, stride 2)
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    new.weight.data = old.weight.data.sum(dim=1, keepdim=True)  # RGB filters -> 1 channel
    net.features[0][0] = new
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, n_classes)
    return net
