import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import pytorch_lightning as pl
from src.model import WeatherClassifier
from src.data import build_dataloaders
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    _, _, test_loader = build_dataloaders(cfg)
    ckpt = "checkpoints/epoch=29-val_f1=0.9756.ckpt"
    model = WeatherClassifier.load_from_checkpoint(ckpt, cfg=cfg, strict=False)
    trainer = pl.Trainer(precision=cfg.training.precision, logger=False)
    results = trainer.test(model, test_loader)
    print("Test results:", results)

if __name__ == "__main__":
    main()
