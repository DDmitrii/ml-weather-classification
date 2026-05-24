import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, label_smoothing: float = 0.0):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class MultiHeadLoss(nn.Module):
    """
    loss = loss_dn + lam * loss_wt

    gamma = 0.0  -> CrossEntropy
    gamma > 0.0  -> FocalLoss
    """

    def __init__(
        self,
        lam: float = 1.0,
        gamma: float = 0.0,
        weight_dn: torch.Tensor = None,
        weight_wt: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lam = lam

        if gamma > 0:
            self.criterion_dn = FocalLoss(
                gamma=gamma,
                weight=weight_dn,
                label_smoothing=label_smoothing,
            )
            self.criterion_wt = FocalLoss(
                gamma=gamma,
                weight=weight_wt,
                label_smoothing=label_smoothing,
            )
        else:
            self.criterion_dn = nn.CrossEntropyLoss(
                weight=weight_dn,
                label_smoothing=label_smoothing,
            )
            self.criterion_wt = nn.CrossEntropyLoss(
                weight=weight_wt,
                label_smoothing=label_smoothing,
            )

    def forward(self, logits_dn, logits_wt, y_dn, y_wt):
        loss_dn = self.criterion_dn(logits_dn, y_dn)
        loss_wt = self.criterion_wt(logits_wt, y_wt)
        total = loss_dn + self.lam * loss_wt
        return total, loss_dn, loss_wt


def build_loss(cfg, class_weights: torch.Tensor = None) -> nn.Module:
    loss_name = cfg.training.get("loss", "weighted_ce")
    label_smoothing = cfg.training.get("label_smoothing", 0.0)
    gamma = cfg.training.get("focal_gamma", 2.0)
    use_class_weights = cfg.training.get("use_class_weights", True)

    weights = class_weights if use_class_weights else None

    if loss_name == "multihead":
        lam = cfg.training.get("multihead_lambda", 1.0)
        return MultiHeadLoss(
            lam=lam,
            gamma=0.0,
            label_smoothing=label_smoothing,
        )

    if loss_name == "focal_multihead":
        lam = cfg.training.get("multihead_lambda", 1.0)
        return MultiHeadLoss(
            lam=lam,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

    if loss_name == "focal":
        return FocalLoss(
            gamma=gamma,
            weight=weights,
            label_smoothing=label_smoothing,
        )

    return WeightedCrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing,
    )