import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
import timm
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix

from src.model.losses import build_loss


class WeatherClassifier(pl.LightningModule):

    def __init__(self, cfg, class_weights: torch.Tensor = None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["class_weights"])
        num_classes = cfg.model.num_classes
        self.class_names = list(cfg.data.class_names)

        # ── Backbone ──────────────────────────────────────────
        self.model = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=num_classes,
            drop_rate=cfg.model.dropout,
        )

        self._freeze_epochs = getattr(cfg.model, "freeze_epochs", 0)
        if self._freeze_epochs > 0:
            self._freeze_backbone()

        # ── Loss ──────────────────────────────────────────────
        self.criterion = build_loss(cfg, class_weights)

        # ── Метрики train/val ─────────────────────────────────
        metric_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc   = Accuracy(**metric_kwargs)
        self.val_f1    = F1Score(average="macro", **metric_kwargs)

        # ── Метрики test ──────────────────────────────────────
        self.test_acc       = Accuracy(**metric_kwargs)
        self.test_f1        = F1Score(average="macro", **metric_kwargs)
        self.test_precision = Precision(average="none", **metric_kwargs)
        self.test_recall    = Recall(average="none", **metric_kwargs)
        self.test_f1_per    = F1Score(average="none", **metric_kwargs)
        self.test_cm        = ConfusionMatrix(**metric_kwargs)

    # ── Freeze helpers ────────────────────────────────────────
    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if "classifier" not in name and "head" not in name and "fc" not in name:
                param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self):
        if self._freeze_epochs > 0 and self.current_epoch == self._freeze_epochs:
            self._unfreeze_backbone()
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.cfg.training.lr * 0.1
            print(f"\n🔓 Epoch {self.current_epoch}: backbone разморожен, lr → {self.cfg.training.lr * 0.1:.2e}")

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
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1_per(preds, y)
        self.test_cm(preds, y)
        self.log("test_acc", self.test_acc)
        self.log("test_f1",  self.test_f1)

    def on_test_epoch_end(self):
        precision = self.test_precision.compute()
        recall    = self.test_recall.compute()
        f1        = self.test_f1_per.compute()
        cm        = self.test_cm.compute().cpu().numpy()

        # ── Per-class метрики в консоль ───────────────────────
        print("\n📊 Per-class metrics:")
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 50)
        for i, name in enumerate(self.class_names):
            print(f"{name:<15} {precision[i]:>10.3f} {recall[i]:>10.3f} {f1[i]:>10.3f}")
        print("-" * 50)
        print(f"{'macro avg':<15} {precision.mean():>10.3f} {recall.mean():>10.3f} {f1.mean():>10.3f}")

        # ── Per-class метрики в MLflow ────────────────────────
        for i, name in enumerate(self.class_names):
            self.log(f"test_f1_{name}",        f1[i])
            self.log(f"test_precision_{name}", precision[i])
            self.log(f"test_recall_{name}",    recall[i])

        # ── Confusion matrix ──────────────────────────────────
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(f"Confusion Matrix — {self.cfg.model.name}", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        cm_path = f"confusion_matrix_{self.cfg.model.name}.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"\n✅ Confusion matrix сохранена: {cm_path}")

        # Логируем в MLflow как артефакт
        if self.logger:
            self.logger.experiment.log_artifact(self.logger.run_id, cm_path)

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