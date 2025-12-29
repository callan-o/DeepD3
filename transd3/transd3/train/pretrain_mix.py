"""Entry point for mixed MAE + Noise2Void pretraining."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transd3.data.datamodule import DataConfig, SpineDataModule
from transd3.models.mae.mae_2p import LitMAE2P, MAEConfig
from transd3.models.denoise.n2v import LitNoise2Void


def _count_visible_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        tokens = [tok.strip() for tok in visible.split(",") if tok.strip()]
        if tokens:
            return len(tokens)
    return torch.cuda.device_count() or 0


def _normalize_devices_value(value: Any):
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "auto":
            return "auto"
        if "," in value:
            devices_list = [int(tok.strip()) for tok in value.split(",") if tok.strip()]
            return devices_list or None
        try:
            return int(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unrecognized trainer.devices value: {value}") from exc
    return value


def _detect_world_size() -> int:
    for key in ("WORLD_SIZE", "SLURM_NTASKS", "PMI_SIZE"):
        val = os.environ.get(key)
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                continue
    return 1


def _resolve_devices(trainer_cfg: Dict[str, Any], accelerator: str, world_size: int, num_nodes: int):
    devices_cfg = _normalize_devices_value(trainer_cfg.get("devices"))
    if world_size > 1:
        nodes = max(1, num_nodes)
        per_node = max(1, world_size // nodes)
        if world_size % nodes != 0:
            per_node = world_size
        if devices_cfg not in (None, "auto", per_node):
            print(
                f"Detected torch.distributed launch (WORLD_SIZE={world_size}); overriding trainer.devices to {per_node} "
                f"so that devices*num_nodes matches world size."
            )
        return per_node

    if devices_cfg in (None, "auto"):
        if accelerator == "gpu" and torch.cuda.is_available():
            return max(1, _count_visible_gpus())
        return 1

    return devices_cfg


def set_determinism(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config + apply overrides, build model/datamodule, then exit (no training).",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    merged = OmegaConf.create(cfg)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_cfg)
    return OmegaConf.to_container(merged, resolve=True)


class LitMAEN2VMix(pl.LightningModule):
    def __init__(self, mae_cfg: MAEConfig, lam_n2v: float, optim_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.mae = LitMAE2P(mae_cfg, optim_cfg=optim_cfg)
        self.n2v = LitNoise2Void(in_ch=mae_cfg.in_ch)
        self.lam_n2v = lam_n2v
        self.optim_cfg = dict(optim_cfg)

    def forward(self, x: torch.Tensor):  # pragma: no cover
        return self.mae(x)

    def _mae_forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        recon, aux = self.mae(batch["image"], batch.get("voxel_size"))
        mask = aux["mask"]
        loss_vst = self.mae._mae_loss(recon, batch["image"], mask)
        loss_grad = self.mae._grad_loss(recon, batch["image"], mask)
        loss = loss_vst + self.mae.cfg.grad_loss_weight * loss_grad
        return {
            "loss": loss,
            "loss_vst": loss_vst,
            "loss_grad": loss_grad,
            "mask_ratio": aux["mask_ratio"],
        }

    def _n2v_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        noisy = batch["image"]
        pred = self.n2v.model(noisy)
        return torch.mean((pred - noisy) ** 2)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mae_stats = self._mae_forward(batch)
        loss_n2v = self._n2v_loss(batch)
        loss = mae_stats["loss"] + self.lam_n2v * loss_n2v
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_mae", mae_stats["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_mae_vst", mae_stats["loss_vst"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_mae_grad", mae_stats["loss_grad"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_n2v", loss_n2v, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mask_ratio", mae_stats["mask_ratio"], on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        mae_stats = self._mae_forward(batch)
        self.log("val/loss", mae_stats["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/loss_vst", mae_stats["loss_vst"], on_epoch=True, sync_dist=True)
        self.log("val/loss_grad", mae_stats["loss_grad"], on_epoch=True, sync_dist=True)
        self.log("val/mask_ratio", mae_stats["mask_ratio"], on_epoch=True, sync_dist=True)

    def configure_optimizers(self):  # pragma: no cover
        params = list(self.mae.parameters()) + list(self.n2v.parameters())
        lr = float(self.optim_cfg.get("lr", 1.0e-4))
        weight_decay = float(self.optim_cfg.get("weight_decay", 0.05))
        t_max = int(self.optim_cfg.get("max_epochs", 100))
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max)
        return {"optimizer": optim, "lr_scheduler": sched}


def main() -> None:
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    cfg_dict = apply_overrides(OmegaConf.to_container(base_cfg, resolve=True), args.overrides)
    cfg = OmegaConf.create(cfg_dict)

    if args.dry_run:
        print("[dry_run] Resolved config (after overrides):")
        print(OmegaConf.to_yaml(cfg, resolve=True))

    set_determinism(1337)
    pl.seed_everything(1337, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    model_cfg_raw = cfg.get("model") or {}
    if isinstance(model_cfg_raw, DictConfig):
        model_cfg = OmegaConf.to_container(model_cfg_raw, resolve=True)
    else:
        model_cfg = dict(model_cfg_raw)
    assert isinstance(model_cfg, dict)
    lam_n2v = float(model_cfg.pop("lam_n2v", 0.5))

    mae_cfg = MAEConfig(**model_cfg)

    data_cfg = DataConfig(
        train_manifest=Path(cfg.data.train_manifest),
        val_manifest=Path(cfg.data.val_manifest),
        batch_size=int(cfg.data.get("batch_size", 8)),
        num_workers=int(cfg.data.get("num_workers", 8)),
        k_neighbors=int(cfg.data.get("k_neighbors", 2)),
        tile_hw=tuple(cfg.data.get("tile_hw", [256, 256])),
        tile_stride=tuple(cfg.data.tile_stride) if cfg.data.get("tile_stride") is not None else None,
        ssl_mode=bool(cfg.data.get("ssl_mode", True)),
        use_pseudolabels=bool(cfg.data.get("use_pseudolabels", False)),
        prefetch_factor=int(cfg.data.get("prefetch_factor", 2)),
        max_tiles_per_volume=cfg.data.get("max_tiles_per_volume"),
        max_tiles_total=cfg.data.get("max_tiles_total"),
    )
    datamodule = SpineDataModule(data_cfg)

    expected_in_ch = 2 * int(data_cfg.k_neighbors) + 1
    if int(mae_cfg.in_ch) != expected_in_ch:
        print(
            f"Config model.in_ch={mae_cfg.in_ch} does not match data.k_neighbors={data_cfg.k_neighbors} "
            f"(expected in_ch={expected_in_ch}); overriding model.in_ch to {expected_in_ch}."
        )
        mae_cfg.in_ch = expected_in_ch

    optim_cfg = OmegaConf.to_container(cfg.optim, resolve=True) if cfg.get("optim") is not None else {}
    assert isinstance(optim_cfg, dict)
    model = LitMAEN2VMix(mae_cfg, lam_n2v=lam_n2v, optim_cfg=optim_cfg)

    if args.dry_run:
        print(
            "[dry_run] Effective settings: "
            f"data.train_manifest={data_cfg.train_manifest} "
            f"data.val_manifest={data_cfg.val_manifest} "
            f"data.k_neighbors={data_cfg.k_neighbors} "
            f"data.tile_hw={data_cfg.tile_hw} "
            f"data.tile_stride={data_cfg.tile_stride} "
            f"data.batch_size={data_cfg.batch_size} "
            f"data.num_workers={data_cfg.num_workers} "
            f"data.prefetch_factor={data_cfg.prefetch_factor} "
            f"model.in_ch={mae_cfg.in_ch} "
            f"model.lam_n2v={model.lam_n2v} "
            f"optim.lr={optim_cfg.get('lr')} "
            f"optim.weight_decay={optim_cfg.get('weight_decay')} "
            f"optim.max_epochs={optim_cfg.get('max_epochs')}"
        )
        return

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

    trainer_cfg_raw = cfg.get("trainer", {}) or {}
    if isinstance(trainer_cfg_raw, DictConfig):
        trainer_cfg: Dict[str, Any] = OmegaConf.to_container(trainer_cfg_raw, resolve=True)  # type: ignore[assignment]
    else:
        trainer_cfg = dict(trainer_cfg_raw)

    accelerator = trainer_cfg.get("accelerator") or ("gpu" if torch.cuda.is_available() else "cpu")
    world_size = _detect_world_size()
    num_nodes = int(trainer_cfg.get("num_nodes", 1) or 1)
    devices = _resolve_devices(trainer_cfg, accelerator, world_size, num_nodes)

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "num_nodes": num_nodes,
        "strategy": trainer_cfg.get("strategy", "auto"),
        "precision": trainer_cfg.get("precision", "16-mixed"),
        "max_epochs": cfg.optim.max_epochs,
        "default_root_dir": str(ckpt_dir),
        "callbacks": callbacks,
        "logger": logger,
        "gradient_clip_val": trainer_cfg.get("gradient_clip_val", 1.0),
    }
    log_every = trainer_cfg.get("log_every_n_steps")
    if log_every is not None:
        trainer_kwargs["log_every_n_steps"] = log_every

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
