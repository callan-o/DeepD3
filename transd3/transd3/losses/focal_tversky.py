"""Focal Tversky loss for class-imbalanced segmentation."""

from __future__ import annotations

import torch
from torch import nn


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        tp = torch.sum(prob * target)
        fp = torch.sum(prob * (1 - target))
        fn = torch.sum((1 - prob) * target)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma)
