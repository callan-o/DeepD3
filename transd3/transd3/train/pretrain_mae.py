"""Entry point for masked autoencoder pretraining."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from transd3.data.datamodule import DataConfig, SpineDataModule
from transd3.models.mae.mae_2p import LitMAE2P, MAEConfig


def set_determinism(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("overrides", nargs="*", help="Config overrides in key=value form")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    merged = OmegaConf.create(cfg)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_cfg)
    return OmegaConf.to_container(merged, resolve=True)


def main() -> None:
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    cfg_dict = apply_overrides(OmegaConf.to_container(base_cfg, resolve=True), args.overrides)
    cfg = OmegaConf.create(cfg_dict)

    set_determinism(1337)
    pl.seed_everything(1337, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True

    data_cfg = DataConfig(
        train_manifest=Path(cfg.data.train_manifest),
        val_manifest=Path(cfg.data.val_manifest),
        batch_size=cfg.data.get("batch_size", 8),
        num_workers=cfg.data.get("num_workers", 8),
        k_neighbors=cfg.data.get("k_neighbors", 2),
        tile_hw=tuple(cfg.data.get("tile_hw", [256, 256])),
        tile_stride=tuple(cfg.data.tile_stride) if cfg.data.get("tile_stride") else None,
        ssl_mode=True,
        use_pseudolabels=cfg.data.get("use_pseudolabels", False),
        prefetch_factor=cfg.data.get("prefetch_factor", 2),
        max_tiles_per_volume=cfg.data.get("max_tiles_per_volume"),
        max_tiles_total=cfg.data.get("max_tiles_total"),
    )
    datamodule = SpineDataModule(data_cfg)

    mae_cfg = MAEConfig(**cfg.model)
    mae_cfg.in_ch = 2 * data_cfg.k_neighbors + 1

    optim_cfg = dict(OmegaConf.to_container(cfg.optim, resolve=True))
    model = LitMAE2P(mae_cfg, optim_cfg)

    checkpoint_dir = Path(cfg.log.ckpt_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="mae2p_ImageNet-{epoch:03d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(save_dir=str(checkpoint_dir), name="tb_logs")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        max_epochs=cfg.optim.max_epochs,
        default_root_dir=str(checkpoint_dir),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
