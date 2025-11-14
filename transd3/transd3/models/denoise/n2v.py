"""Noise2Void-style denoising head for optional self-supervision."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import pytorch_lightning as pl


class SimpleUNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - pass-through wrapper
        features = self.encoder(x)
        return self.decoder(features)


class LitNoise2Void(pl.LightningModule):
    def __init__(self, in_ch: int = 5, lr: float = 1e-4) -> None:
        super().__init__()
        self.model = SimpleUNet(in_ch)
        self.lr = lr

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        noisy = batch["image"]
        pred = self.model(noisy)
        loss = torch.mean((pred - noisy) ** 2)
        self.log("train/n2v_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):  # pragma: no cover
        return torch.optim.Adam(self.parameters(), lr=self.lr)
