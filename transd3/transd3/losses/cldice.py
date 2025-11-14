"""Centerline Dice loss implementation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def soft_skeletonize(prob: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    skeleton = prob
    for _ in range(iterations):
        min_pool = -F.max_pool2d(-skeleton, kernel_size=3, stride=1, padding=1)
        skeleton = torch.relu(skeleton - torch.relu(skeleton - min_pool))
    return torch.clamp(skeleton, 0.0, 1.0)


class CLDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_skel = soft_skeletonize(torch.sigmoid(pred))
        target_skel = soft_skeletonize(target)
        tp = torch.sum(pred_skel * target) + self.smooth
        precision = tp / (torch.sum(pred_skel) + self.smooth)
        recall = tp / (torch.sum(target_skel) + self.smooth)
        return 1.0 - (2 * precision * recall) / (precision + recall + self.smooth)
