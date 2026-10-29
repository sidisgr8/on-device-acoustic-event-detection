from .cnn import CNNBaseline
from .crnn import CRNN
from .mobilenet import MobileNetV3Small


def build_model(cfg: dict):
    """Instantiate a model from the `model:` section of a config."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name == "cnn_baseline":
        return CNNBaseline(**cfg)
    if name == "mobilenetv3_small":
        return MobileNetV3Small(**cfg)
    if name == "crnn":
        return CRNN(**cfg)
    if name == "ast":
        from .ast import ASTClassifier  # imported lazily: heavy, and not needed by the CNN family
        return ASTClassifier(**cfg)
    raise ValueError(f"unknown model {name!r}")
