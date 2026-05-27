import os
os.environ["PL_WEIGHTS_ONLY_LOAD"] = "0"

import torch
import typing
import omegaconf
import numpy as np
import json
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from hydra import initialize, compose
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.data import build_dataloaders, WeatherDataset, get_train_transforms
from src.model import WeatherClassifierMultiHead

torch.set_float32_matmul_precision('high')
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

CHECKPOINT = "checkpoints/fold1/epoch=20-val_f1=0.9952.ckpt"
THRESHOLDS_PATH = "configs/thresholds.json"
MLFLOW_EXPERIMENT = "threshold-calibration"


def collect_probs(model, loader, device):
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            logits_dn, logits_wt = model(x)
            probs = model.predict_probs(logits_dn, logits_wt)
            all_probs.append(probs.cpu())
            all_labels.extend(y.tolist())
    return torch.cat(all_probs), all_labels


def apply_thresholds(probs: torch.Tensor, thresholds: dict) -> list:
    adjusted = probs.clone()
    for cls_idx, thr in thresholds.items():
        mask = adjusted[:, int(cls_idx)] < thr
        adjusted[mask, int(cls_idx)] = 0.0
    return adjusted.argmax(dim=1).tolist()


def calibrate(probs, labels, class_names):
    n_classes = len(class_names)
    best_thresholds = {i: 0.5 for i in range(n_classes)}

    baseline_preds = probs.argmax(dim=1).tolist()
    baseline_f1 = f1_score(labels, baseline_preds, average="macro")
    print(f"\nBaseline val macro-F1: {baseline_f1:.4f}")

    print("\nПодбираю пороги...")
    for cls_idx in range(n_classes):
        best_f1_cls = f1_score(labels, baseline_preds, average=None)[cls_idx]
        best_thr = 0.5

        for thr in np.arange(0.3, 0.85, 0.05):
            test_thresholds = {**best_thresholds, cls_idx: thr}
            preds = apply_thresholds(probs, test_thresholds)
            f1_cls = f1_score(labels, preds, average=None)[cls_idx]
            if f1_cls > best_f1_cls:
                best_f1_cls = f1_cls
                best_thr = thr

        best_thresholds[cls_idx] = round(float(best_thr), 2)
        print(f"  {class_names[cls_idx]:<15} thr={best_thr:.2f}  F1={best_f1_cls:.3f}")

    return best_thresholds


def save_confusion_matrix(labels, preds, class_names, title, path):
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
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["experiments=v7_multihead_freeze"])

    class_names = list(cfg.data.class_names)
    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=class_names,
        transform=get_train_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()
    _, val_loader, test_loader = build_dataloaders(cfg)

    model = WeatherClassifierMultiHead.load_from_checkpoint(
        CHECKPOINT, cfg=cfg, class_weights=class_weights, weights_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Собираю вероятности на val...")
    val_probs, val_labels = collect_probs(model, val_loader, device)

    print("Собираю вероятности на test...")
    test_probs, test_labels = collect_probs(model, test_loader, device)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="threshold-calib-fold1"):

        mlflow.log_param("checkpoint", CHECKPOINT)
        mlflow.log_param("model", cfg.model.name)

        # ── Baseline метрики ─────────────────────────────────
        baseline_preds = test_probs.argmax(dim=1).tolist()
        baseline_f1_macro = f1_score(test_labels, baseline_preds, average="macro")
        baseline_f1_per = f1_score(test_labels, baseline_preds, average=None)

        print("\nBaseline (test):")
        print(classification_report(test_labels, baseline_preds, target_names=class_names))

        mlflow.log_metric("baseline_test_f1_macro", baseline_f1_macro)
        for i, name in enumerate(class_names):
            mlflow.log_metric(f"baseline_test_f1_{name}", baseline_f1_per[i])

        save_confusion_matrix(test_labels, baseline_preds, class_names,
                              "Baseline — no calibration (test)",
                              "reports/cm_baseline.png")
        mlflow.log_artifact("reports/cm_baseline.png")

        thresholds = calibrate(val_probs, val_labels, class_names)
        print(f"\nИтоговые пороги: {thresholds}")

        for i, name in enumerate(class_names):
            mlflow.log_param(f"thr_{name}", thresholds[i])

        calibrated_preds = apply_thresholds(test_probs, thresholds)
        calibrated_f1_macro = f1_score(test_labels, calibrated_preds, average="macro")
        calibrated_f1_per = f1_score(test_labels, calibrated_preds, average=None)

        print("\nПосле калибровки (test):")
        print(classification_report(test_labels, calibrated_preds, target_names=class_names))

        mlflow.log_metric("calibrated_test_f1_macro", calibrated_f1_macro)
        for i, name in enumerate(class_names):
            mlflow.log_metric(f"calibrated_test_f1_{name}", calibrated_f1_per[i])

        delta = calibrated_f1_macro - baseline_f1_macro
        mlflow.log_metric("delta_f1_macro", delta)
        print(f"\nΔ macro-F1: {delta:+.4f}")

        save_confusion_matrix(test_labels, calibrated_preds, class_names,
                              "Calibrated thresholds (test)",
                              "reports/cm_calibrated.png")
        mlflow.log_artifact("reports/cm_calibrated.png")

        with open(THRESHOLDS_PATH, "w") as f:
            json.dump({class_names[i]: v for i, v in thresholds.items()}, f, indent=2)
        mlflow.log_artifact(THRESHOLDS_PATH)
        print(f"\nПороги сохранены: {THRESHOLDS_PATH}")