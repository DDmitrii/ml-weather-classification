import os
os.environ["PL_WEIGHTS_ONLY_LOAD"] = "0"

import torch
import omegaconf
import typing
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score, accuracy

import hydra

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])
torch.set_float32_matmul_precision("high")


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    from src.data.dataset import WeatherDataset
    from src.data.transforms import get_val_transforms
    from src.model import WeatherClassifierMultiHead, WeatherClassifier

    ckpt_path = "checkpoints/fold1/epoch=20-val_f1=0.9952.ckpt"

    # Загружаем чекпоинт и определяем тип модели по ключам
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = list(state["state_dict"].keys())
    print(f"🔑 Первые 5 ключей в чекпоинте: {keys[:5]}")

    is_multihead = any("backbone." in k or "head_dn." in k for k in keys)
    is_classifier = any("model.stages." in k for k in keys)
    print(f"🔍 multihead: {is_multihead}, plain classifier: {is_classifier}")

    class_names = list(cfg.data.class_names)

    # Инициализируем нужную модель
    if is_multihead:
        model = WeatherClassifierMultiHead(cfg)
    else:
        model = WeatherClassifier(cfg)

    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    print(f"✅ Загружено: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"   ⚠️ Missing keys (первые 3): {missing[:3]}")
    if unexpected:
        print(f"   ⚠️ Unexpected keys (первые 3): {unexpected[:3]}")

    model.eval()

    # Тестовый датасет
    test_ds = WeatherDataset(
        cfg.data.test_dir,
        class_names=class_names,
        transform=get_val_transforms(cfg.data.img_size)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    print(f"\n📂 Тестовых сэмплов: {len(test_ds)}")

    # Запуск теста через Trainer
    trainer = pl.Trainer(
        precision=cfg.training.precision,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.test(model, test_loader)

    # --- Дополнительно: ручной прогон для per-class метрик ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            if is_multihead:
                logits_dn, logits_wt = model(x)
                preds = model._combine_preds(logits_dn, logits_wt)
            else:
                preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    preds_t   = torch.tensor(all_preds)
    targets_t = torch.tensor(all_targets)
    n_cls     = cfg.model.num_classes

    test_acc = accuracy(preds_t, targets_t, task="multiclass", num_classes=n_cls).item()
    test_f1  = f1_score(preds_t, targets_t, task="multiclass", num_classes=n_cls, average="macro").item()
    f1_per   = f1_score(preds_t, targets_t, task="multiclass", num_classes=n_cls, average="none")
    from torchmetrics.functional import precision as prec_fn, recall as rec_fn
    prec_per = prec_fn(preds_t, targets_t, task="multiclass", num_classes=n_cls, average="none")
    rec_per  = rec_fn(preds_t, targets_t, task="multiclass", num_classes=n_cls, average="none")

    print(f"\n📊 Per-class metrics (test):")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {prec_per[i]:>10.3f} {rec_per[i]:>10.3f} {f1_per[i]:>10.3f}")
    print("-" * 50)
    print(f"{'macro avg':<15} {prec_per.mean():>10.3f} {rec_per.mean():>10.3f} {test_f1:>10.3f}")
    print(f"\n✅ Test Accuracy: {test_acc:.4f}  |  Test F1-macro: {test_f1:.4f}")

    # Confusion matrix
    from torchmetrics.functional import confusion_matrix as cm_fn
    cm = cm_fn(preds_t, targets_t, task="multiclass", num_classes=n_cls).numpy()
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    Path("reports").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Test Confusion Matrix — fold1 (val_f1=0.9952)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = f"reports/test_confusion_matrix_fold1.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix сохранена: {cm_path}")


if __name__ == "__main__":
    main()