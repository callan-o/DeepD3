"""Entry point for mixed MAE + Noise2Void pretraining."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transd3.data.datamodule import DataConfig, SpineDataModule
from transd3.models.mae.mae_2p import LitMAE2P, MAEConfig
from transd3.models.denoise.n2v import LitNoise2Void


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


class LitMAEN2VMix(pl.LightningModule):
    def __init__(self, mae_cfg: MAEConfig, lam_n2v: float) -> None:
        super().__init__()
        self.mae = LitMAE2P(mae_cfg)
        self.n2v = LitNoise2Void(in_ch=mae_cfg.in_ch)
        self.lam_n2v = lam_n2v

    def forward(self, x: torch.Tensor):  # pragma: no cover
        return self.mae(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_mae = self.mae.training_step(batch, batch_idx)
        loss_n2v = self.n2v.training_step(batch, batch_idx)
        loss = loss_mae + self.lam_n2v * loss_n2v
        self.log("train/loss_mae", loss_mae, on_epoch=True, prog_bar=True)
        self.log("train/loss_n2v", loss_n2v, on_epoch=True)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self.mae.validation_step(batch, batch_idx)

    def configure_optimizers(self):  # pragma: no cover
        params = list(self.mae.parameters()) + list(self.n2v.parameters())
        optim = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
        return {"optimizer": optim, "lr_scheduler": sched}


def main() -> None:
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    cfg_dict = apply_overrides(OmegaConf.to_container(base_cfg, resolve=True), args.overrides)
    cfg = OmegaConf.create(cfg_dict)

    set_determinism(1337)
    pl.seed_everything(1337, workers=True)

    mae_cfg = MAEConfig(**cfg.model)
    data_cfg = DataConfig(
        train_manifest=Path(cfg.data.train_manifest),
        val_manifest=Path(cfg.data.val_manifest),
        batch_size=cfg.data.get("batch_size", 8),
        num_workers=cfg.data.get("num_workers", 8),
        k_neighbors=cfg.data.get("k_neighbors", 2),
        tile_hw=tuple(cfg.data.get("tile_hw", [256, 256])),
        ssl_mode=True,
    )
    datamodule = SpineDataModule(data_cfg)

    model = LitMAEN2VMix(mae_cfg, lam_n2v=cfg.model.get("lam_n2v", 0.5))

    ckpt_dir = Path(cfg.log.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="pretrain-mix-{epoch:03d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir=str(ckpt_dir), name="tb_logs")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        max_epochs=cfg.optim.max_epochs,
        default_root_dir=str(ckpt_dir),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
