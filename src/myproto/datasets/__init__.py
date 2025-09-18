from .fake import build_fake_dataset
from .imagenet import build_imagenet_dataset

def build_dataset(cfg):
    if cfg.name == "fake":
        return build_fake_dataset(**cfg)
    elif cfg.name == "imagenet":
        return build_imagenet_dataset(**cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")
