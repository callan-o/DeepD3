"""Segmentation heads composing the TransD3 model."""

from __future__ import annotations

import importlib
from typing import Any, Dict, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transd3.models.encoders.conv_stem import ConvStem as _ConvStem
    from transd3.models.encoders.vit_axial import ViTAxialEncoder as _ViTAxialEncoder
    from transd3.models.seg.mask2former import Mask2FormerDecoder as _Mask2FormerDecoder

ConvStem = getattr(importlib.import_module("transd3.models.encoders.conv_stem"), "ConvStem")
ViTAxialEncoder = getattr(
    importlib.import_module("transd3.models.encoders.vit_axial"), "ViTAxialEncoder"
)
Mask2FormerDecoder = getattr(
    importlib.import_module("transd3.models.seg.mask2former"), "Mask2FormerDecoder"
)


class SpineSegmentationModel(nn.Module):
    """End-to-end model producing shaft, spine, and center outputs."""

    def __init__(
        self,
        in_ch: int,
        dim: int,
        depth: int,
        num_queries: int,
        out_stride: int = 8,
    ) -> None:
        super().__init__()
        self.stem = ConvStem(in_channels=in_ch, embed_dim=dim)
        self.encoder = ViTAxialEncoder(dim=dim, depth=depth)
        self.decoder = Mask2FormerDecoder(dim=dim, num_queries=num_queries)
        self.shaft_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1),
        )
        self.center_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.stem(x)
        feats = self.encoder(feats)
        shaft = self.shaft_head(feats)
        center = self.center_head(feats)
        masks = self.decoder(feats)
        return {
            "logits_shaft": shaft,
            "center_map": center,
            "masks_spine": masks,
        }
