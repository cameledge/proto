import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import wandb

from myproto.datasets import build_dataset
from myproto.models import build_model
from myproto.trainers.classifier import ImageClassifier
from myproto.utils.logger import build_logger

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # multimodal pipeline mode
    if cfg.model.name == "multimodal":
        pipeline = build_model(cfg.model)
        candidate_labels = [
            "a cat", "a dog", "a car", "a bicycle", "a person",
            "a building", "an outdoor scene", "a microscopic image"
        ]
        out = pipeline.generate_from_image(
            image_path=cfg.input.image_path,
            candidate_labels=candidate_labels,
            top_k=cfg.model.prompt.top_k,
            prompt_template=cfg.model.prompt.template,
        )

        # logger (e.g., wandb or csv)
        logger = build_logger(cfg.logger)

        if hasattr(logger, "experiment") and isinstance(logger.experiment, wandb.wandb_sdk.wandb_run.Run):
            # log image with caption
            logger.experiment.log({
                "input_image": wandb.Image(out["image"], caption="Multimodal input"),
                "top_labels": {label: prob for label, prob in out["label_probs"]},
                "prompt": out["prompt"],
                "generated_text": out["generated_text"],
            })

        print("Top labels:", out["label_probs"])
        print("\nPrompt:\n", out["prompt"])
        print("\nLLM Output:\n", out["generated_text"])
        return

    # --- normal CV training mode ---
    loaders = build_dataset(cfg.dataset)
    if isinstance(loaders, tuple):
        train_loader, val_loader = loaders
    else:
        train_loader = val_loader = loaders

    model = build_model(cfg.model)

    if cfg.model.name == "resnet18":
        lit_model = ImageClassifier(model, lr=1e-3)
    else:
        raise NotImplementedError("Only ResNet classifier supported in training loop for now")

    logger = build_logger(cfg.logger)
    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    trainer.fit(lit_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
