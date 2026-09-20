"""Model registry: name -> (constructor, learning rate, pretrained flag)."""
from .cnn import CNN
from .crnn import CRNN
from .mobilenet import build_mobilenet

MODELS = {
    # name:                (builder,                               lr,   pretrained label)
    "cnn":                 (lambda: CNN(),                         1e-3, "No"),
    "crnn":                (lambda: CRNN(),                        1e-3, "No"),
    "mobilenet_scratch":   (lambda: build_mobilenet(pretrained=False), 1e-3, "No"),
    "mobilenet_pretrained": (lambda: build_mobilenet(pretrained=True), 3e-4, "ImageNet"),
}

DISPLAY = {
    "cnn": "CNN baseline", "crnn": "CRNN (CNN + BiGRU)",
    "mobilenet_scratch": "MobileNetV3-Small", "mobilenet_pretrained": "MobileNetV3-Small",
}


def build_model(name):
    return MODELS[name][0]()
