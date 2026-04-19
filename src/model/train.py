import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics import Accuracy, F1Score

from src.model.losses import build_loss


class WeatherClassifier(pl.LightningModule):

    def __init__(self, cfg, class_weights: torch.Tensor = None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["class_weights"])
        num_classes = cfg.model.num_classes

        # ── Backbone ──────────────────────────────────────────
        self.model = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=num_classes,
            drop_rate=cfg.model.dropout,
        )

        # ── Loss ──────────────────────────────────────────────
        self.criterion = build_loss(cfg, class_weights)

        # ── Метрики ───────────────────────────────────────────
        metric_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc   = Accuracy(**metric_kwargs)
        self.val_f1    = F1Score(average="macro", **metric_kwargs)
        self.test_acc  = Accuracy(**metric_kwargs)
        self.test_f1   = F1Score(average="macro", **metric_kwargs)

    # ── Forward ───────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Train ─────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc",  self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    # ── Validation ────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",  self.val_acc, prog_bar=True)
        self.log("val_f1",   self.val_f1,  prog_bar=True)

    # ── Test ──────────────────────────────────────────────────
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).argmax(dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log("test_acc", self.test_acc)
        self.log("test_f1",  self.test_f1)

    # ── Optimizer + Scheduler ─────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )

        if self.cfg.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.training.max_epochs,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        return optimizer