import os
import hydra
from omegaconf import DictConfig
import wandb
from myproto.models import build_model

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # build multimodal pipeline
    pipeline = build_model(cfg.model)

    # setup W&B logger
    from myproto.utils.logger import build_logger
    logger = build_logger(cfg.logger)

    # list all images in folder
    image_files = [
        os.path.join(cfg.input_batch.image_folder, f)
        for f in os.listdir(cfg.input_batch.image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    print(f"Found {len(image_files)} images in {cfg.input_batch.image_folder}")

    for img_path in image_files:
        out = pipeline.generate_from_image(
            image_path=img_path,
            candidate_labels=cfg.input_batch.candidate_labels,
            top_k=cfg.input_batch.top_k,
            prompt_template=cfg.input_batch.prompt_template
        )

        # log to W&B
        if hasattr(logger, "experiment") and isinstance(logger.experiment, wandb.wandb_sdk.wandb_run.Run):
            logger.experiment.log({
                "input_image": wandb.Image(out["image"], caption=os.path.basename(img_path)),
                "top_labels": {label: prob for label, prob in out["label_probs"]},
                "prompt": out["prompt"],
                "generated_text": out["generated_text"],
            })
        print(f"\nProcessed {os.path.basename(img_path)}")
        print("Top labels:", out["label_probs"])
        print("Generated text:\n", out["generated_text"])

if __name__ == "__main__":
    main()
