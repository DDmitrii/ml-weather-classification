import os
os.environ["PL_WEIGHTS_ONLY_LOAD"] = "0"

import torch
import mlflow
import numpy as np
import omegaconf
import pytorch_lightning as pl
import typing
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.functional import f1_score, accuracy

import hydra
from omegaconf import DictConfig

from src.data.dataset import WeatherDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.model import WeatherClassifier, WeatherClassifierMultiHead

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])
torch.set_float32_matmul_precision("high")


def collect_all_samples(cfg) -> list:
    """Собираем все сэмплы из train + val."""
    class_names = list(cfg.data.class_names)
    samples = []
    for split in ["train_dir", "val_dir"]:
        root = Path(cfg.data[split])
        for cls in class_names:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    samples.append((img_path, class_names.index(cls)))
    return samples


class SamplesDataset(WeatherDataset):
    """WeatherDataset из готового списка сэмплов (без сканирования папки)."""
    def __init__(self, samples, class_names, transform=None):
        self.root = None
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.samples = samples


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    class_names = list(cfg.data.class_names)
    all_samples = collect_all_samples(cfg)
    labels      = np.array([label for _, label in all_samples])

    print(f"Всего сэмплов для KFold: {len(all_samples)}")
    print(f"   Классы: {class_names}")

    skf     = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.seed)
    n_cls   = cfg.model.num_classes

    # OOF буферы
    oof_preds   = np.full(len(all_samples), -1, dtype=int)
    oof_targets = labels.copy()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_name = cfg.mlflow.get("experiment_name", "kfold-convnext")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/3  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*50}")

        train_samples = [all_samples[i] for i in train_idx]
        val_samples   = [all_samples[i] for i in val_idx]

        train_ds = SamplesDataset(
            train_samples, class_names,
            transform=get_train_transforms(cfg.data.img_size)
        )
        val_ds = SamplesDataset(
            val_samples, class_names,
            transform=get_val_transforms(cfg.data.img_size)
        )

        class_counts = [0] * n_cls
        for _, lbl in train_samples:
            class_counts[lbl] += 1
        sample_weights = [1.0 / class_counts[lbl] for _, lbl in train_samples]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        loader_kwargs = dict(
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
        val_loader   = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        weights = train_ds.get_class_weights()

        use_multihead = cfg.training.get("multihead", False)
        model = (
            WeatherClassifierMultiHead(cfg, class_weights=weights)
            if use_multihead
            else WeatherClassifier(cfg, class_weights=weights)
        )

        from pytorch_lightning.loggers import MLFlowLogger
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            run_name=f"{cfg.model.name}-fold{fold+1}",
        )
        mlf_logger._experiment_id = experiment_id

        callbacks = [
            ModelCheckpoint(
                dirpath=f"checkpoints/fold{fold+1}",
                filename="{epoch:02d}-{val_f1:.4f}",
                monitor="val_f1", mode="max", save_top_k=1,
            ),
            EarlyStopping(
                monitor="val_f1", mode="max",
                patience=cfg.training.early_stopping_patience,
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

        trainer.fit(model, train_loader, val_loader)

        print(f"\nСобираю OOF для fold {fold+1}...")
        model.eval()
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device

        fold_preds = []
        oof_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size,
                                shuffle=False, num_workers=cfg.data.num_workers)

        with torch.no_grad():
            for batch in oof_loader:
                x = batch[0].to(device)
                if use_multihead:
                    logits_dn, logits_wt = model(x)
                    preds = model._combine_preds(logits_dn, logits_wt)
                else:
                    preds = model(x).argmax(dim=1)
                fold_preds.extend(preds.cpu().numpy())

        oof_preds[val_idx] = fold_preds
        mlflow.end_run()

    print("\n" + "="*50)
    print("OOF результаты по всей выборке")
    print("="*50)

    oof_preds_t   = torch.tensor(oof_preds)
    oof_targets_t = torch.tensor(oof_targets)

    oof_acc = accuracy(oof_preds_t, oof_targets_t,
                       task="multiclass", num_classes=n_cls).item()
    oof_f1  = f1_score(oof_preds_t, oof_targets_t,
                       task="multiclass", num_classes=n_cls, average="macro").item()
    f1_per  = f1_score(oof_preds_t, oof_targets_t,
                       task="multiclass", num_classes=n_cls, average="none")

    print(f"   OOF Accuracy : {oof_acc:.4f}")
    print(f"   OOF F1-macro : {oof_f1:.4f}")
    print(f"\n{'Class':<15} {'F1':>8}")
    print("-" * 25)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {f1_per[i]:>8.3f}")

    from torchmetrics.functional import confusion_matrix as cm_fn
    cm = cm_fn(oof_preds_t, oof_targets_t,
               task="multiclass", num_classes=n_cls).numpy()
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    Path("reports").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"OOF Confusion Matrix — {cfg.model.name} (3-fold)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = f"reports/oof_confusion_matrix_{cfg.model.name}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nOOF confusion matrix сохранена: {cm_path}")

    errors = [(all_samples[i][0], oof_targets[i], oof_preds[i])
              for i in range(len(all_samples)) if oof_preds[i] != oof_targets[i]]
    print(f"\nОшибочных предсказаний: {len(errors)} / {len(all_samples)} "
          f"({100*len(errors)/len(all_samples):.1f}%)")

    from collections import Counter
    error_pairs = Counter((class_names[t], class_names[p]) for _, t, p in errors)
    print("\nТоп-10 ошибочных пар (true → predicted):")
    for (true_cls, pred_cls), count in error_pairs.most_common(10):
        print(f"   {true_cls:<15} → {pred_cls:<15} : {count}")

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{cfg.model.name}-oof-summary"):
        mlflow.log_metric("oof_accuracy", oof_acc)
        mlflow.log_metric("oof_f1_macro", oof_f1)
        for i, name in enumerate(class_names):
            mlflow.log_metric(f"oof_f1_{name}", f1_per[i].item())
        mlflow.log_artifact(cm_path)


if __name__ == "__main__":
    main()