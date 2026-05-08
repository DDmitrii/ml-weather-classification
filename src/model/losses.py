import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class MultiHeadLoss(nn.Module):
    """
    Суммирует лоссы двух хедов:
      loss = CE(logits_dn, y_dn) + λ * CE(logits_wt, y_wt)
    """
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(
        self,
        logits_dn: torch.Tensor,   # (B, 2)  — day/night
        logits_wt: torch.Tensor,   # (B, 5)  — weather type
        y_dn: torch.Tensor,        # (B,)
        y_wt: torch.Tensor,        # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_dn = F.cross_entropy(logits_dn, y_dn)
        loss_wt = F.cross_entropy(logits_wt, y_wt)
        total   = loss_dn + self.lam * loss_wt
        return total, loss_dn, loss_wt


def build_loss(cfg, class_weights: torch.Tensor = None) -> nn.Module:
    loss_name = cfg.training.get("loss", "weighted_ce")
    weights   = class_weights if cfg.training.use_class_weights else None

    if loss_name == "multihead":
        lam = cfg.training.get("multihead_lambda", 1.0)
        return MultiHeadLoss(lam=lam)
    if loss_name == "focal":
        return FocalLoss(gamma=2.0, weight=weights)
    return WeightedCrossEntropyLoss(weight=weights)