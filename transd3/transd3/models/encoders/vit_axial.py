"""Axial Vision Transformer style encoder."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class AxialAttention(nn.Module):
    """Applies self-attention along height and width axes independently."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm_h = nn.LayerNorm(dim)
        self.attn_h = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm_w = nn.LayerNorm(dim)
        self.attn_w = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        x_norm = self.norm_h(x_flat)
        attn_h, _ = self.attn_h(x_norm, x_norm, x_norm)
        attn_h = attn_h.reshape(b, h, w, c).permute(0, 3, 1, 2)

        y_flat = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        y_norm = self.norm_w(y_flat)
        attn_w, _ = self.attn_w(y_norm, y_norm, y_norm)
        attn_w = attn_w.reshape(b, w, h, c).permute(0, 3, 2, 1)

        out = x + attn_h + attn_w
        out_mlp = self.mlp(out.permute(0, 2, 3, 1))
        out = out + out_mlp.permute(0, 3, 1, 2)
        return out


class ViTAxialEncoder(nn.Module):
    """Stack of axial attention blocks operating on convolutional features."""

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        num_heads: int = 8,
        input_resolution: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.pos_embed = None
        if input_resolution is not None:
            h, w = input_resolution
            self.pos_embed = nn.Parameter(torch.zeros(1, dim, h, w))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([AxialAttention(dim=dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            pos = self.pos_embed
            if pos.shape[-2:] != x.shape[-2:]:
                pos = nn.functional.interpolate(pos, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
