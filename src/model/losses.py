import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """CrossEntropy с весами классов для борьбы с дисбалансом."""

    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """
    Focal Loss — усиливает обучение на сложных примерах.
    Полезен при сильном дисбалансе классов.
    gamma=0 → обычный CrossEntropy.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


def build_loss(cfg, class_weights: torch.Tensor = None) -> nn.Module:
    """Выбрать функцию потерь из конфига."""
    weights = class_weights if cfg.training.use_class_weights else None
    loss_name = cfg.training.get("loss", "weighted_ce")

    if loss_name == "focal":
        return FocalLoss(gamma=2.0, weight=weights)
    return WeightedCrossEntropyLoss(weight=weights)