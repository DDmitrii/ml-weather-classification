import os
os.environ["PL_WEIGHTS_ONLY_LOAD"] = "0"

import torch
import typing
import omegaconf
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
from albumentations.pytorch import ToTensorV2
from hydra import initialize, compose
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data import build_dataloaders, WeatherDataset, get_train_transforms, get_val_transforms
from src.model import WeatherClassifier, WeatherClassifierMultiHead

torch.set_float32_matmul_precision("high")
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

CHECKPOINT    = "checkpoints/epoch=17-val_f1=0.9921.ckpt"  # ← заменить
EXPERIMENT    = "focal_loss_multihead"                        # ← имя конфига
USE_MULTIHEAD = True
TTA_ENABLED   = True
TTA_N         = 5

THRESHOLDS = None


def get_tta_transforms(img_size: int) -> list[A.Compose]:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    base = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    augmentations = [
        [],
        [A.HorizontalFlip(p=1.0)],
        [A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0)],
        [A.GaussianBlur(blur_limit=(3, 5), p=1.0)],
        [A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02), rotate=(-5, 5), p=1.0)],
        [A.RandomGamma(gamma_limit=(85, 115), p=1.0)],
    ]

    return [A.Compose(aug + base) for aug in augmentations[:TTA_N + 1]]


def _forward(model, x: torch.Tensor, use_multihead: bool) -> torch.Tensor:
    """Единый forward pass → softmax probs."""
    if use_multihead:
        logits_dn, logits_wt = model(x)
        return model.predict_probs(logits_dn, logits_wt)
    return torch.softmax(model(x), dim=1)


def collect_probs(model, loader, device, use_multihead) -> tuple[torch.Tensor, list]:
    """Стандартный inference без TTA."""
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            all_probs.append(_forward(model, x, use_multihead).cpu())
            all_labels.extend(y.tolist())
    return torch.cat(all_probs), all_labels


def collect_probs_tta(
    model,
    dataset: WeatherDataset,
    cfg,
    device,
    use_multihead,
) -> tuple[torch.Tensor, list]:
    """
    TTA inference: несколько forward passes с разными аугментациями,
    вероятности усредняются.
    """
    tta_transforms = get_tta_transforms(cfg.data.img_size)
    loader_kwargs  = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    accumulated_probs = None
    all_labels = None

    model.eval()
    for i, transform in enumerate(tta_transforms, start=1):
        tta_ds = WeatherDataset(
            root=dataset.root,
            class_names=dataset.class_names,
            transform=transform,
        )
        loader = DataLoader(tta_ds, **loader_kwargs)

        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1]
                probs_list.append(_forward(model, x, use_multihead).cpu())
                labels_list.extend(y.tolist())

        probs_i = torch.cat(probs_list)
        if accumulated_probs is None:
            accumulated_probs = probs_i
            all_labels = labels_list
        else:
            accumulated_probs += probs_i

        print(f"  TTA pass {i}/{len(tta_transforms)} ✓")

    return accumulated_probs / len(tta_transforms), all_labels


def predict_with_thresholds(probs: torch.Tensor, thresholds: dict | None) -> torch.Tensor:
    """Применяет per-class пороги к softmax вероятностям."""
    if thresholds is None:
        return probs.argmax(dim=1)
    adjusted = probs.clone()
    for cls_idx, thr in thresholds.items():
        adjusted[:, cls_idx] *= (adjusted[:, cls_idx] >= thr).float()
    return adjusted.argmax(dim=1)


def calibrate_thresholds(probs: torch.Tensor, labels: list, class_names: list) -> dict:
    """Подбирает оптимальный порог для каждого класса по F1 на val."""
    best_thresholds = {}
    for cls_idx, cls_name in enumerate(class_names):
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.3, 0.91, 0.05):
            preds = predict_with_thresholds(probs, {cls_idx: thr}).tolist()
            f1 = f1_score(
                labels, preds,
                average=None,
                labels=list(range(len(class_names))),
                zero_division=0,
            )[cls_idx]
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        best_thresholds[cls_idx] = best_thr
        print(f"  {cls_name:15s} → thr={best_thr:.2f}  val_F1={best_f1:.3f}")
    return best_thresholds


def print_and_save_report(
    labels: list,
    preds: list,
    class_names: list,
    title: str,
    cm_path: str,
) -> None:
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    print(f"\n {title}")
    print(classification_report(labels, preds, target_names=class_names, digits=3))

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix: {cm_path}")


if __name__ == "__main__":
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiments={EXPERIMENT}"])

    class_names = list(cfg.data.class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=class_names,
        transform=get_train_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()

    test_ds = WeatherDataset(
        root=cfg.data.test_dir,
        class_names=class_names,
        transform=get_val_transforms(cfg.data.img_size),
    )

    _, val_loader, test_loader = build_dataloaders(cfg)

    ModelClass = WeatherClassifierMultiHead if USE_MULTIHEAD else WeatherClassifier
    model = ModelClass.load_from_checkpoint(
        CHECKPOINT,
        cfg=cfg,
        class_weights=class_weights,
        weights_only=False,
    )
    model.to(device)
    print(f"Модель загружена: {CHECKPOINT}")

    print("\n[1/4] Baseline inference (без TTA)...")
    test_probs, test_labels = collect_probs(model, test_loader, device, USE_MULTIHEAD)
    baseline_preds = predict_with_thresholds(test_probs, None).tolist()
    print_and_save_report(
        test_labels, baseline_preds, class_names,
        "Baseline (no TTA, no calibration)",
        "reports/cm_baseline.png",
    )

    if TTA_ENABLED:
        print(f"\n🔁 [2/4] TTA inference ({TTA_N + 1} passes)...")
        tta_probs, tta_labels = collect_probs_tta(
            model, test_ds, cfg, device, USE_MULTIHEAD
        )
        tta_preds = predict_with_thresholds(tta_probs, None).tolist()
        print_and_save_report(
            tta_labels, tta_preds, class_names,
            f"TTA ({TTA_N + 1} passes, no calibration)",
            "reports/cm_tta.png",
        )

    print("\n [3/4] Подбираю пороги на val...")
    val_probs, val_labels = collect_probs(model, val_loader, device, USE_MULTIHEAD)
    thresholds = calibrate_thresholds(val_probs, val_labels, class_names)
    print(f"\n Найденные пороги: {thresholds}")

    print("\n [4/4] Применяю пороги к test...")

    calibrated_preds = predict_with_thresholds(test_probs, thresholds).tolist()
    print_and_save_report(
        test_labels, calibrated_preds, class_names,
        "Baseline + calibrated thresholds",
        "reports/cm_baseline_calibrated.png",
    )

    if TTA_ENABLED:
        calibrated_tta_preds = predict_with_thresholds(tta_probs, thresholds).tolist()
        print_and_save_report(
            tta_labels, calibrated_tta_preds, class_names,
            f"TTA ({TTA_N + 1} passes) + calibrated thresholds",
            "reports/cm_tta_calibrated.png",
        )

    print("\nРезультаты сохранены в reports/")