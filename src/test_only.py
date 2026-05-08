# test_only.py
import torch
import typing
import omegaconf
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from hydra import initialize, compose
from sklearn.metrics import classification_report, confusion_matrix

from src.data import build_dataloaders, WeatherDataset, get_train_transforms
from src.model import WeatherClassifier, WeatherClassifierMultiHead

torch.set_float32_matmul_precision('high')
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

CHECKPOINT = "checkpoints/fold2/epoch=21-val_f1=0.9800.ckpt"
USE_MULTIHEAD = False  # True если мультихед чекпоинт

if __name__ == '__main__':
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["~experiments"])

    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=list(cfg.data.class_names),
        transform=get_train_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()
    _, _, test_loader = build_dataloaders(cfg)

    ModelClass = WeatherClassifierMultiHead if USE_MULTIHEAD else WeatherClassifier
    model = ModelClass.load_from_checkpoint(
        CHECKPOINT,
        cfg=cfg,
        class_weights=class_weights,
        weights_only=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1]
            if USE_MULTIHEAD:
                logits_dn, logits_wt = model(x)
                preds = model._combine_preds(logits_dn, logits_wt)
            else:
                preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())

    class_names = list(cfg.data.class_names)
    print("\n📊 Per-class report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {cfg.model.name} (test)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = f"confusion_matrix_{cfg.model.name}_kfold_test.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\n✅ Confusion matrix сохранена: {cm_path}")