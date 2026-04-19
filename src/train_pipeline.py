import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger
import hydra
from omegaconf import DictConfig

from src.data import build_dataloaders, WeatherDataset, get_train_transforms
from src.model import WeatherClassifier


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:

    # ── Воспроизводимость ─────────────────────────────────────
    pl.seed_everything(cfg.seed, workers=True)

    # ── Данные ───────────────────────────────────────────────
    print("📂 Загружаю данные...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # Веса классов для loss
    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=list(cfg.data.class_names),
        transform=get_train_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()
    print(f"   Train: {len(train_ds)} изображений")
    print(f"   Веса классов: {[round(w, 3) for w in class_weights.tolist()]}")

    # ── Модель ────────────────────────────────────────────────
    print(f"\n🧠 Модель: {cfg.model.name}")
    model = WeatherClassifier(cfg, class_weights=class_weights)

    # ── Логгер MLflow ─────────────────────────────────────────
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    # ── Callbacks ─────────────────────────────────────────────
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

    # ── Trainer ───────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=mlf_logger,
        log_every_n_steps=10,
        deterministic=True,
    )

    # ── Обучение ──────────────────────────────────────────────
    print("\n🚀 Начинаю обучение...")
    trainer.fit(model, train_loader, val_loader)

    # ── Тест ──────────────────────────────────────────────────
    print("\n🧪 Оцениваю на test set...")
    trainer.test(model, test_loader, ckpt_path="best")

    print(f"\n✅ Готово! Лучший чекпоинт: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()