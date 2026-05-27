import os
os.environ["PL_WEIGHTS_ONLY_LOAD"] = "0"


import torch
import mlflow
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import typing

from src.data import build_dataloaders, WeatherDataset, get_train_transforms
from src.model import WeatherClassifierMultiHead, WeatherClassifier


torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

torch.set_float32_matmul_precision('high')


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.seed, workers=True)

    print("Загружаю данные...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=list(cfg.data.class_names),
        transform=get_train_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()
    print(f"   Train: {len(train_ds)} изображений")
    print(f"   Веса классов: {[round(w, 3) for w in class_weights.tolist()]}")

    # ── Модель ────────────────────────────────────────────────
    print(f"\n Модель: {cfg.model.name}")
    use_multihead = cfg.training.get("multihead", False)
    model = WeatherClassifierMultiHead(cfg, class_weights=class_weights) \
        if use_multihead else WeatherClassifier(cfg, class_weights=class_weights)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    experiment_name = OmegaConf.select(cfg, "mlflow.experiment_name", default="default-experiment")
    run_name = OmegaConf.select(cfg, "mlflow.run_name", default=f"{cfg.model.name}-run")

    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(cfg.mlflow.experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=cfg.mlflow.run_name,
    )

    mlf_logger._experiment_id = experiment_id

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoints.dirpath,
            filename=cfg.checkpoints.filename,
            monitor=cfg.checkpoints.monitor,
            mode=cfg.checkpoints.mode,
            save_top_k=cfg.checkpoints.save_top_k,
            verbose=True,
        ),
        EarlyStopping(
            monitor=cfg.checkpoints.monitor,
            mode=cfg.checkpoints.mode,
            patience=cfg.training.early_stopping_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=mlf_logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    print("\nНачинаю обучение...")
    trainer.fit(model, train_loader, val_loader)

    print("\nОцениваю на test set...")
    trainer.test(model, test_loader, ckpt_path="best")

    print(f"\nЛучший чекпоинт: {trainer.checkpoint_callback.best_model_path}")

    mlflow.end_run()


if __name__ == "__main__":
    main()