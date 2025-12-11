"""Masked autoencoder tailored for 2.5D two-photon stacks."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from transd3.models.encoders.conv_stem import ConvStem, load_imagenet_conv_stem
from transd3.models.encoders.vit_axial import ViTAxialEncoder


@dataclass
class MAEConfig:
    in_ch: int = 5
    dim: int = 128
    depth: int = 6
    patch: int = 8
    mask_ratio: float = 0.7
    num_heads: int = 8
    grad_loss_weight: float = 0.1
    vst_eps: float = 1.0e-3
    stem_imagenet_model: Optional[str] = "resnet50.a1_in1k"
    stem_weight_strategy: str = "avg"
    use_voxel_conditioning: bool = True


class MaskedAutoencoder2P(nn.Module):
    def __init__(self, cfg: MAEConfig) -> None:
        super().__init__()
        if cfg.patch % 4 != 0:
            raise ValueError("patch size must be divisible by ConvStem stride (4)")
        if cfg.dim % 4 != 0:
            raise ValueError("Embedding dim must be divisible by 4 for decoder")
        self.cfg = cfg
        self.patch_cells = cfg.patch // 4

        self.stem = ConvStem(cfg.in_ch, cfg.dim)
        if cfg.stem_imagenet_model:
            try:
                load_imagenet_conv_stem(
                    self.stem,
                    model_name=cfg.stem_imagenet_model,
                    in_channels=cfg.in_ch,
                    channel_strategy=cfg.stem_weight_strategy,
                )
            except ImportError as exc:
                warnings.warn(
                    "timm not available; proceeding with randomly initialized ConvStem. "
                    "Install timm>=0.9.16 or set model.stem_imagenet_model=null to disable preload.",
                    RuntimeWarning,
                )

        self.mask_token = nn.Parameter(torch.zeros(1, cfg.dim, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.use_voxel_conditioning = bool(cfg.use_voxel_conditioning)
        if self.use_voxel_conditioning:
            self.cond_norm = nn.LayerNorm(3)
            self.cond_proj = nn.Sequential(
                nn.Linear(3, cfg.dim),
                nn.SiLU(inplace=True),
                nn.Linear(cfg.dim, cfg.dim),
            )
            self.cond_scale = nn.Linear(cfg.dim, cfg.dim)
            self.cond_shift = nn.Linear(cfg.dim, cfg.dim)
            nn.init.zeros_(self.cond_scale.weight)
            nn.init.zeros_(self.cond_scale.bias)
            nn.init.zeros_(self.cond_shift.weight)
            nn.init.zeros_(self.cond_shift.bias)

        self.encoder = ViTAxialEncoder(dim=cfg.dim, depth=cfg.depth, num_heads=cfg.num_heads)
        self.decoder = nn.Sequential(
            nn.Conv2d(cfg.dim, cfg.dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(cfg.dim, cfg.dim // 2, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(cfg.dim // 2, cfg.dim // 4, kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(cfg.dim // 4, cfg.in_ch, kernel_size=3, padding=1),
        )
        self._init_decoder()

    def _init_decoder(self) -> None:
        for module in self.decoder.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, voxel_size: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        stem_feats = self.stem(x)
        b, _, h, w = stem_feats.shape
        if h % self.patch_cells != 0 or w % self.patch_cells != 0:
            raise ValueError("Input resolution must be divisible by patch size")

        hp = h // self.patch_cells
        wp = w // self.patch_cells
        mask = self._generate_mask(b, hp * wp, device=x.device).view(b, hp, wp)
        mask_token = self.mask_token.expand(b, -1, -1, -1)

        if self.use_voxel_conditioning:
            scale, shift = self._encode_condition(voxel_size, batch_size=b, device=x.device)
            stem_feats = self._apply_scale_shift(stem_feats, scale, shift)
            mask_token = self._apply_scale_shift(mask_token, scale, shift)

        masked_feats = self._apply_patch_mask(stem_feats, mask, mask_token)

        encoded = self.encoder(masked_feats)
        recon = self.decoder(encoded)

        mask_full = F.interpolate(mask.unsqueeze(1).float(), scale_factor=self.cfg.patch, mode="nearest")
        aux = {
            "mask": mask_full,
            "mask_ratio": mask.float().mean(),
            "mask_patches": mask,
        }
        return recon, aux

    def _apply_patch_mask(self, feats: torch.Tensor, mask: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feats.shape
        hp, wp = mask.shape[1], mask.shape[2]
        feats_view = feats.view(b, c, hp, self.patch_cells, wp, self.patch_cells)
        feats_view = feats_view.permute(0, 2, 4, 1, 3, 5)  # B, Hp, Wp, C, ph, pw
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        masked = torch.where(
            mask_expanded.bool(),
            mask_token.reshape(b, 1, 1, c, 1, 1),
            feats_view,
        )
        masked = masked.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return masked

    def _encode_condition(
        self, voxel_size: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = torch.zeros(batch_size, 3, device=device, dtype=self.mask_token.dtype)
        if voxel_size is not None:
            cond = voxel_size.to(device=device, dtype=self.mask_token.dtype)
            if cond.dim() == 1:
                cond = cond.view(1, -1).expand(batch_size, -1)
            elif cond.shape[0] == 1 and batch_size > 1:
                cond = cond.expand(batch_size, -1)
        cond = torch.log1p(cond.clamp_min(1.0e-6))
        cond = self.cond_norm(cond)
        cond_embed = self.cond_proj(cond)
        scale = self.cond_scale(cond_embed)
        shift = self.cond_shift(cond_embed)
        return scale, shift

    def _apply_scale_shift(self, feats: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        scale_view = scale.view(feats.shape[0], feats.shape[1], 1, 1)
        shift_view = shift.view(feats.shape[0], feats.shape[1], 1, 1)
        return feats * (1 + scale_view) + shift_view

    def _generate_mask(self, batch: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        num_mask = max(1, int(num_tokens * self.cfg.mask_ratio))
        if num_mask >= num_tokens:
            return torch.ones(batch, num_tokens, dtype=torch.bool, device=device)
        mask = torch.zeros(batch, num_tokens, dtype=torch.bool, device=device)
        for b_idx in range(batch):
            perm = torch.randperm(num_tokens, device=device)
            mask[b_idx, perm[:num_mask]] = True
        return mask


class LitMAE2P(pl.LightningModule):
    def __init__(self, cfg: MAEConfig, optim_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["optim_cfg"])
        self.model = MaskedAutoencoder2P(cfg)
        self.cfg = cfg
        self.optim_cfg = optim_cfg or {"lr": 1.0e-4, "weight_decay": 0.05, "max_epochs": 100}

    def forward(self, x: torch.Tensor, voxel_size: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:  # pragma: no cover
        return self.model(x, voxel_size)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        recon, aux = self(batch["image"], batch.get("voxel_size"))
        mask = aux["mask"]
        loss_vst = self._mae_loss(recon, batch["image"], mask)
        loss_grad = self._grad_loss(recon, batch["image"], mask)
        loss = loss_vst + self.cfg.grad_loss_weight * loss_grad
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_vst", loss_vst, on_epoch=True)
        self.log("train/loss_grad", loss_grad, on_epoch=True)
        self.log("train/mask_ratio", aux["mask_ratio"], on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        recon, aux = self(batch["image"], batch.get("voxel_size"))
        mask = aux["mask"]
        loss_vst = self._mae_loss(recon, batch["image"], mask)
        loss_grad = self._grad_loss(recon, batch["image"], mask)
        loss = loss_vst + self.cfg.grad_loss_weight * loss_grad
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/loss_vst", loss_vst, on_epoch=True)
        self.log("val/loss_grad", loss_grad, on_epoch=True)
        self.log("val/mask_ratio", aux["mask_ratio"], on_epoch=True)

    def configure_optimizers(self):  # pragma: no cover - Lightning handles testing
        lr = self.optim_cfg.get("lr", 1.0e-4)
        weight_decay = self.optim_cfg.get("weight_decay", 0.05)
        t_max = self.optim_cfg.get("max_epochs", 100)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _mae_loss(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transform = self._vst_transform
        diff = (transform(recon) - transform(target)).abs()
        return self._masked_mean(diff, mask)

    def _grad_loss(self, recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        recon_dx = recon[..., :, 1:] - recon[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        recon_dy = recon[..., 1:, :] - recon[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]

        mask_dx = mask[..., :, 1:] * mask[..., :, :-1]
        mask_dy = mask[..., 1:, :] * mask[..., :-1, :]

        loss_x = self._masked_mean((recon_dx - target_dx).abs(), mask_dx)
        loss_y = self._masked_mean((recon_dy - target_dy).abs(), mask_dy)
        return loss_x + loss_y

    def _masked_mean(self, value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return value.mean()
        expanded_mask = mask.to(value.dtype)
        if mask.shape[1] == 1 and value.shape[1] > 1:
            expanded_mask = expanded_mask.expand(-1, value.shape[1], *mask.shape[2:])
        total = expanded_mask.sum().clamp_min(1.0)
        return (value * expanded_mask).sum() / total

    def _vst_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sqrt(torch.clamp(tensor, min=0.0) + self.cfg.vst_eps)
