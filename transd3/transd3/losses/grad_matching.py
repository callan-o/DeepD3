"""Gradient matching regularizer."""

from __future__ import annotations

import torch
from torch import nn


def gradient_map(tensor: torch.Tensor) -> torch.Tensor:
    dx = tensor[..., :, :, 1:] - tensor[..., :, :, :-1]
    dy = tensor[..., :, 1:, :] - tensor[..., :, :-1, :]
    return torch.cat([dx, dy], dim=-1)


class GradientMatchingLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad = gradient_map(pred.unsqueeze(-3))
        target_grad = gradient_map(target.unsqueeze(-3))
        return torch.mean(torch.abs(pred_grad - target_grad))
