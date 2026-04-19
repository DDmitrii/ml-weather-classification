import torch
import pytorch_lightning as pl
from pathlib import Path

from src.data import build_dataloaders
from src.model.train import WeatherClassifier
from src.utils.metrics import (
    compute_metrics,
    print_report,
    plot_confusion_matrix,
    collect_predictions,
)


def evaluate(cfg, checkpoint_path: str) -> dict:
    """Полная оценка модели на test set."""

    # Загружаем модель из чекпоинта
    model = WeatherClassifier.load_from_checkpoint(
        checkpoint_path, cfg=cfg
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Загружаем данные
    _, _, test_loader = build_dataloaders(cfg)

    print(f"🔍 Оцениваю на test set ({device})...")
    preds, targets, probs = collect_predictions(model, test_loader, device)

    class_names = list(cfg.data.class_names)

    # Метрики
    metrics = compute_metrics(preds, targets, class_names)
    print(f"\n📊 Результаты:")
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   F1-macro:    {metrics['f1_macro']:.4f}")
    print(f"   F1-weighted: {metrics['f1_weighted']:.4f}")

    # Подробный отчёт
    print("\n📋 Classification report:")
    print_report(preds, targets, class_names)

    # Confusion matrix
    Path("reports").mkdir(exist_ok=True)
    plot_confusion_matrix(
        preds, targets, class_names,
        save_path="reports/confusion_matrix.png"
    )

    return metrics