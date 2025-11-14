"""PyTorch Lightning data module wrapping TransD3 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transd3.spine_data.datasets import TwoPhotonStackDataset


@dataclass
class DataConfig:
    train_manifest: Path
    val_manifest: Path
    batch_size: int = 8
    num_workers: int = 8
    k_neighbors: int = 2
    tile_hw: tuple[int, int] = (256, 256)
    tile_stride: Optional[tuple[int, int]] = None
    ssl_mode: bool = True
    use_pseudolabels: bool = False
    prefetch_factor: int = 2
    max_tiles_per_volume: Optional[int] = None
    max_tiles_total: Optional[int] = None


class SpineDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[TwoPhotonStackDataset] = None
        self.val_ds: Optional[TwoPhotonStackDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            stride = tuple(self.cfg.tile_stride) if self.cfg.tile_stride else None
            self.train_ds = TwoPhotonStackDataset(
                self.cfg.train_manifest,
                k_neighbors=self.cfg.k_neighbors,
                tile_hw=self.cfg.tile_hw,
                tile_stride=stride,
                ssl_mode=self.cfg.ssl_mode,
                use_pseudolabels=self.cfg.use_pseudolabels,
                max_tiles_per_volume=self.cfg.max_tiles_per_volume,
                max_tiles_total=self.cfg.max_tiles_total,
            )
            self.val_ds = TwoPhotonStackDataset(
                self.cfg.val_manifest,
                k_neighbors=self.cfg.k_neighbors,
                tile_hw=self.cfg.tile_hw,
                tile_stride=stride,
                ssl_mode=self.cfg.ssl_mode,
                use_pseudolabels=self.cfg.use_pseudolabels,
                max_tiles_per_volume=self.cfg.max_tiles_per_volume,
                max_tiles_total=self.cfg.max_tiles_total,
            )

    def _build_loader(self, dataset: TwoPhotonStackDataset, *, shuffle: bool) -> DataLoader:
        kwargs = {
            "dataset": dataset,
            "batch_size": self.cfg.batch_size,
            "shuffle": shuffle,
            "num_workers": self.cfg.num_workers,
            "pin_memory": True,
        }
        if self.cfg.num_workers > 0:
            kwargs["prefetch_factor"] = max(1, self.cfg.prefetch_factor)
            kwargs["persistent_workers"] = True
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return self._build_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return self._build_loader(self.val_ds, shuffle=False)
