import os

os.environ["PL_WEIGHTS_ONLY_LOAD"] = "0"

import torch
import typing
import omegaconf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hydra import initialize, compose
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data import build_dataloaders, WeatherDataset, get_train_transforms
from src.model import WeatherClassifier, WeatherClassifierMultiHead

torch.set_float32_matmul_precision('high')
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

CHECKPOINT = "checkpoints/fold1/epoch=20-val_f1=0.9952.ckpt"
USE_MULTIHEAD = True

# Пороги per-class (None = стандартный argmax, без калибровки)
# Заполняется после подбора на val
THRESHOLDS = None  # например: {4: 0.6, 6: 0.55}  # 4=night_fog, 6=night_snow


def predict_with_thresholds(probs: torch.Tensor, thresholds: dict | None) -> torch.Tensor:
    """Применяет per-class пороги к softmax вероятностям."""
    if thresholds is None:
        return probs.argmax(dim=1)

    adjusted = probs.clone()
    for cls_idx, thr in thresholds.items():
        adjusted[:, cls_idx] *= (adjusted[:, cls_idx] >= thr).float()
    return adjusted.argmax(dim=1)


def collect_probs(model, loader, device, use_multihead):
    """Собирает softmax вероятности и истинные метки."""
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            if use_multihead:
                logits_dn, logits_wt = model(x)
                probs = model._combine_preds(logits_dn, logits_wt, return_probs=True)
            else:
                probs = torch.softmax(model(x), dim=1)
            all_probs.append(probs.cpu())
            all_labels.extend(y.tolist())
    return torch.cat(all_probs), all_labels


def calibrate_thresholds(probs, labels, class_names):
    """Подбирает оптимальный порог для каждого класса по val F1."""
    best_thresholds = {}
    thresholds_range = np.arange(0.3, 0.9, 0.05)

    for cls_idx, cls_name in enumerate(class_names):
        best_f1, best_thr = 0.0, 0.5
        for thr in thresholds_range:
            preds = predict_with_thresholds(probs, {cls_idx: thr}).tolist()
            f1 = f1_score(labels, preds, average=None, labels=list(range(len(class_names))))[cls_idx]
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        best_thresholds[cls_idx] = best_thr
        print(f"  {cls_name:15s} → best_thr={best_thr:.2f}  F1={best_f1:.3f}")

    return best_thresholds


def print_and_save_report(labels, preds, class_names, title, cm_path):
    print(f"\n📊 {title}")
    print(classification_report(labels, preds, target_names=class_names))

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    print(f"✅ Confusion matrix: {cm_path}")


if __name__ == '__main__':
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["experiments=v7_multihead_freeze"])

    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=list(cfg.data.class_names),
        transform=get_train_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()
    class_names = list(cfg.data.class_names)

    _, val_loader, test_loader = build_dataloaders(cfg)

    ModelClass = WeatherClassifierMultiHead if USE_MULTIHEAD else WeatherClassifier
    model = ModelClass.load_from_checkpoint(
        CHECKPOINT,
        cfg=cfg,
        class_weights=class_weights,
        weights_only=False,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ── Baseline (без калибровки) ────────────────────────────────────────
    print("\n🔍 Собираю предсказания на val (для калибровки)...")
    val_probs, val_labels = collect_probs(model, val_loader, device, USE_MULTIHEAD)

    print("\n🔍 Собираю предсказания на test...")
    test_probs, test_labels = collect_probs(model, test_loader, device, USE_MULTIHEAD)

    baseline_preds = predict_with_thresholds(test_probs, None).tolist()
    print_and_save_report(test_labels, baseline_preds, class_names,
                          "Baseline (no threshold calibration)",
                          "reports/cm_baseline.png")

    # ── Калибровка порогов на val ────────────────────────────────────────
    if THRESHOLDS is None:
        print("\n⚙️  Подбираю пороги на val...")
        THRESHOLDS = calibrate_thresholds(val_probs, val_labels, class_names)
        print(f"\n📌 Найденные пороги: {THRESHOLDS}")

    # ── Результат с калибровкой на test ──────────────────────────────────
    calibrated_preds = predict_with_thresholds(test_probs, THRESHOLDS).tolist()
    print_and_save_report(test_labels, calibrated_preds, class_names,
                          "Calibrated thresholds (test)",
                          "reports/cm_calibrated.png")