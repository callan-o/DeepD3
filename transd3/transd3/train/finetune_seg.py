"""Segmentation fine-tuning entry point."""

from __future__ import annotations

import argparse
import importlib
import random
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transd3.data.datamodule import DataConfig, SpineDataModule

if TYPE_CHECKING:  # pragma: no cover - typing only
    from typing import Any as _Any  # placeholder for static analyzers

CLDiceLoss = getattr(importlib.import_module("transd3.losses.cldice"), "CLDiceLoss")
FocalTverskyLoss = getattr(
    importlib.import_module("transd3.losses.focal_tversky"), "FocalTverskyLoss"
)
GradientMatchingLoss = getattr(
    importlib.import_module("transd3.losses.grad_matching"), "GradientMatchingLoss"
)
SpineSegmentationModel = getattr(
    importlib.import_module("transd3.models.seg.heads"), "SpineSegmentationModel"
)


def set_determinism(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    merged = OmegaConf.create(cfg)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_cfg)
    return OmegaConf.to_container(merged, resolve=True)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = torch.sum(prob * target)
    denom = torch.sum(prob) + torch.sum(target) + eps
    return 1.0 - (2 * inter + eps) / denom


class LitSpineSeg(pl.LightningModule):
    def __init__(self, model_cfg: Dict[str, Any], optim_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model = SpineSegmentationModel(**model_cfg)
        self.optim_cfg = optim_cfg
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.focal = FocalTverskyLoss()
        self.cldice = CLDiceLoss()
        self.grad_match = GradientMatchingLoss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch["image"])
        shaft_mask = batch["shaft_mask"]
        spine_instance = batch["spine_instance"]
        center_target = batch["spine_center_heatmap"]

        shaft_logits = outputs["logits_shaft"]
        loss_shaft = (
            self.bce(shaft_logits, shaft_mask)
            + dice_loss(shaft_logits, shaft_mask)
            + 0.3 * self.cldice(shaft_logits, shaft_mask)
        )

        spine_logits = outputs["masks_spine"].amax(dim=1, keepdim=True)
        spine_target = (spine_instance > 0).float().unsqueeze(1)
        loss_spine = self.focal(spine_logits, spine_target)

        center_pred = outputs["center_map"]
        loss_center = F.mse_loss(center_pred, center_target)

        loss = loss_shaft + loss_spine + 0.2 * loss_center

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_shaft", loss_shaft, on_epoch=True)
        self.log("train/loss_spine", loss_spine, on_epoch=True)
        self.log("train/loss_center", loss_center, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch["image"])
        shaft_mask = batch["shaft_mask"]
        spine_instance = batch["spine_instance"]

        shaft_logits = outputs["logits_shaft"]
        prob_shaft = torch.sigmoid(shaft_logits)
        pred_shaft = (prob_shaft > 0.5).float()
        dice_val = 1.0 - dice_loss(shaft_logits, shaft_mask)
        cldice_val = 1.0 - self.cldice(shaft_logits, shaft_mask)

        pred_spine = (outputs["masks_spine"].amax(dim=1) > 0.5).float()
        target_spine = (spine_instance > 0).float()
        intersection = (pred_spine * target_spine).sum(dim=[1, 2])
        union = pred_spine.sum(dim=[1, 2]) + target_spine.sum(dim=[1, 2])
        f1 = (2 * intersection + 1e-6) / (union + 1e-6)
        ap_small = (intersection / (target_spine.sum(dim=[1, 2]) + 1e-6)).mean()

        self.log("val/dice_shaft", dice_val.mean(), prog_bar=True)
        self.log("val/cldice", cldice_val.mean(), prog_bar=True)
        self.log("val/ap_small@0.3", ap_small.mean())
        self.log("val/f1_spine", f1.mean())

    def configure_optimizers(self):  # pragma: no cover
        optim = torch.optim.AdamW(self.parameters(), lr=self.optim_cfg["lr"], weight_decay=self.optim_cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.optim_cfg.get("epochs", 100))
        return {"optimizer": optim, "lr_scheduler": scheduler}


def maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: str | None) -> None:
    if ckpt_path is None:
        return
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Finetune] Missing keys: {missing}")
    if unexpected:
        print(f"[Finetune] Unexpected keys: {unexpected}")


def main() -> None:
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    cfg_dict = apply_overrides(OmegaConf.to_container(base_cfg, resolve=True), args.overrides)
    cfg = OmegaConf.create(cfg_dict)

    set_determinism(1337)
    pl.seed_everything(1337, workers=True)

    data_cfg = DataConfig(
        train_manifest=Path(cfg.data.train_manifest),
        val_manifest=Path(cfg.data.val_manifest),
        batch_size=cfg.data.get("batch_size", 4),
        num_workers=cfg.data.get("num_workers", 8),
        k_neighbors=cfg.data.get("k_neighbors", 2),
        tile_hw=tuple(cfg.data.get("tile_hw", [256, 256])),
        ssl_mode=False,
        use_pseudolabels=cfg.data.get("use_pseudolabels", False),
    )
    datamodule = SpineDataModule(data_cfg)

    model_cfg = {
        "in_ch": cfg.model.in_ch,
        "dim": cfg.model.dim,
        "depth": cfg.model.depth,
        "num_queries": cfg.model.num_queries,
        "out_stride": cfg.model.get("out_stride", 8),
    }
    lit_module = LitSpineSeg(model_cfg, cfg.optim)
    maybe_load_checkpoint(lit_module.model, cfg.init.get("mae_ckpt"))

    ckpt_dir = Path(cfg.log.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="seg-{epoch:03d}-{val_ap_small@0.3:.4f}",
            monitor="val/ap_small@0.3",
            mode="max",
            save_top_k=2,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=str(ckpt_dir), name="tb_logs")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        max_epochs=cfg.optim.epochs,
        default_root_dir=str(ckpt_dir),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
