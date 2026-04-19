import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    preds: list[int],
    targets: list[int],
    class_names: list[str],
) -> dict:
    """Основные метрики классификации."""
    return {
        "accuracy":  round(accuracy_score(targets, preds), 4),
        "f1_macro":  round(f1_score(targets, preds, average="macro"), 4),
        "f1_weighted": round(f1_score(targets, preds, average="weighted"), 4),
    }


def print_report(
    preds: list[int],
    targets: list[int],
    class_names: list[str],
) -> None:
    """Подробный отчёт по каждому классу."""
    print(classification_report(targets, preds, target_names=class_names))


def plot_confusion_matrix(
    preds: list[int],
    targets: list[int],
    class_names: list[str],
    save_path: str = None,
) -> plt.Figure:
    """Нарисовать confusion matrix."""
    cm = confusion_matrix(targets, preds)
    # Нормализуем по строкам — видно % ошибок для каждого класса
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Confusion Matrix (нормализованная)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"💾 Сохранено: {save_path}")

    return fig


def collect_predictions(model, dataloader, device: str = "cpu"):
    """Собрать предсказания модели на всём датасете."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.tolist())
            all_probs.extend(probs.cpu().tolist())

    return all_preds, all_targets, all_probs