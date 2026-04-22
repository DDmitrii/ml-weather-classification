# test_only.py
import torch
import typing
import omegaconf
import pytorch_lightning as pl
from hydra import initialize, compose
from src.data import build_dataloaders, WeatherDataset, get_train_transforms
from src.model import WeatherClassifier

torch.set_float32_matmul_precision('high')

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

if __name__ == '__main__':
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train")

    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=list(cfg.data.class_names),
        transform=get_train_transforms(cfg.data.img_size),
    )
    class_weights = train_ds.get_class_weights()

    _, _, test_loader = build_dataloaders(cfg)

    ckpt = "checkpoints/epoch=26-val_f1=0.9795.ckpt"
    model = WeatherClassifier.load_from_checkpoint(
        ckpt,
        cfg=cfg,
        class_weights=class_weights,
        weights_only=False,
    )

    trainer = pl.Trainer(logger=False)
    trainer.test(model, test_loader)

    from sklearn.metrics import classification_report

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())

    print("\n📊 Per-class report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=list(cfg.data.class_names)
    ))

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=list(cfg.data.class_names),
        yticklabels=list(cfg.data.class_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (normalized)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()