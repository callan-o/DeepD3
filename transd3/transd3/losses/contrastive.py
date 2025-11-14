"""Simple contrastive loss helpers."""

from __future__ import annotations

import torch
from torch import nn


class PatchContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_a: torch.Tensor, proj_b: torch.Tensor) -> torch.Tensor:
        proj_a = nn.functional.normalize(proj_a, dim=-1)
        proj_b = nn.functional.normalize(proj_b, dim=-1)
        logits = torch.matmul(proj_a, proj_b.T) / self.temperature
        labels = torch.arange(proj_a.size(0), device=proj_a.device)
        loss = nn.functional.cross_entropy(logits, labels)
        return loss
