from .resnet import build_resnet18
from ..vlm.clip import build_clip
from ..pipelines.multimodal import MultiModalPipeline

def build_model(cfg):
    if cfg.name == "resnet18":
        return build_resnet18(num_classes=cfg.num_classes, pretrained=cfg.pretrained)
    elif cfg.name == "clip":
        model, processor = build_clip(model_name=cfg.model_name, device=cfg.device)
        return model
    elif cfg.name == "multimodal":
        return MultiModalPipeline(
            clip_model_name=cfg.clip.model_name,
            clip_device=cfg.clip.device,
            llm_model_name=cfg.llm.model_name,
            llm_device=cfg.llm.device,
            llm_max_length=cfg.llm.max_length,
            llm_do_sample=cfg.llm.do_sample,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.name}")
