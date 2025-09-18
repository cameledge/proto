from pytorch_lightning.loggers import WandbLogger, CSVLogger

def build_logger(cfg):
    if cfg.name == "wandb":
        return WandbLogger(project=cfg.project, log_model=cfg.log_model)
    elif cfg.name == "csv":
        return CSVLogger(save_dir=cfg.save_dir)
    else:
        raise ValueError(f"Unknown logger: {cfg.name}")
