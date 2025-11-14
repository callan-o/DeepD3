"""Simplified Mask2Former-style query decoder."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class Mask2FormerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_queries: int,
        depth: int = 3,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, dim)
        self.layers = nn.ModuleList([
            _DecoderLayer(dim=dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        src = features.flatten(2).permute(0, 2, 1)
        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            query = layer(query, src)
        query = self.output_proj(query)
        mask_logits = torch.einsum("bqc,bchw->bqhw", query, features)
        return mask_logits


class _DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        q = self.norm1(query)
        query = query + self.self_attn(q, q, q)[0]
        q = self.norm2(query)
        query = query + self.cross_attn(q, src, src)[0]
        query = query + self.ffn(self.norm3(query))
        return query
