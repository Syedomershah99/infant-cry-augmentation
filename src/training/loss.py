"""Loss functions for the classifier.

Three options, all selectable via the `loss` block in the YAML config:

  loss:
    type: ce               # cross-entropy (default)
    type: focal            # focal loss (Lin et al., RetinaNet 2017)

Class weighting and label smoothing apply to both. Focal loss uses
gamma (focusing parameter) and the same per-class alpha vector that
the weighted-CE arm would use, so the only knob between recipes is
the (1-p_t)^gamma down-weighting term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with optional per-class alpha weighting and label smoothing.

    L = -alpha_t * (1 - p_t) ** gamma * log p_t
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ls = self.label_smoothing
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Smoothed one-hot
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, ls / max(1, n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - ls)

        p_t = (probs * true_dist).sum(dim=-1).clamp_min(1e-8)
        log_p_t = (log_probs * true_dist).sum(dim=-1)

        focal = (1.0 - p_t) ** self.gamma * (-log_p_t)

        if self.alpha is not None and self.alpha.numel() > 0:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            focal = focal * alpha_t

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def make_loss(cfg: dict, class_w: torch.Tensor) -> nn.Module:
    """Build a loss module from the training config.

    cfg keys honored:
      use_class_weighted_loss: bool — apply class_w to whichever loss is chosen
      label_smoothing: float
      loss.type: ce | focal
      loss.gamma: float (focal only)
    """
    use_w = cfg.get("use_class_weighted_loss", True)
    weight = class_w if use_w else None
    label_smoothing = cfg.get("label_smoothing", 0.0)
    loss_cfg = cfg.get("loss", {}) or {}
    kind = loss_cfg.get("type", "ce")
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if kind == "focal":
        return FocalLoss(
            alpha=weight,
            gamma=loss_cfg.get("gamma", 2.0),
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"unknown loss.type: {kind!r}")
