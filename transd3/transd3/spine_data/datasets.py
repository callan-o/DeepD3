"""Dataset definitions for TransD3 two-photon volumetric stacks."""

from __future__ import annotations

import importlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    zarr: Any
else:  # pragma: no cover
    try:
        zarr = importlib.import_module("zarr")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "zarr is a required dependency for TransD3 datasets; please install it via pip."
        ) from exc


@dataclass(frozen=True)
class TileCoordinate:
    """Index metadata for a deterministic 2.5D tile."""

    volume_path: Path
    z_index: int
    y0: int
    x0: int
    tile_hw: Tuple[int, int]


class TwoPhotonStackDataset(Dataset):
    """Deterministic 2.5D volumetric tiling dataset.

    The dataset reads volumetric Zarr groups produced by ``convert_deepd3_to_zarr.py``
    and tiles them with a fixed stride. Each item returns a stack of ``2 * k_neighbors + 1``
    slices centered at ``z_index`` along with optional supervision depending on ``ssl_mode``.
    """

    def __init__(
        self,
        manifest: Path | str | Sequence[str],
        *,
        k_neighbors: int,
        tile_hw: Tuple[int, int],
        tile_stride: Optional[Tuple[int, int]] = None,
        ssl_mode: bool = True,
        use_pseudolabels: bool = False,
        percentile_bounds: Tuple[float, float] = (1.0, 99.0),
        seed: int = 1337,
        max_tiles_per_volume: Optional[int] = None,
        max_tiles_total: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        if self.k_neighbors < 0:
            raise ValueError("k_neighbors must be >= 0")
        self.tile_hw = tuple(int(v) for v in tile_hw)
        if len(self.tile_hw) != 2:
            raise ValueError("tile_hw must be (H, W)")
        default_stride = (self.tile_hw[0] // 2, self.tile_hw[1] // 2)
        self.tile_stride = tuple(int(v) for v in (tile_stride or default_stride))
        self.ssl_mode = ssl_mode
        self.use_pseudolabels = use_pseudolabels
        self.percentile_bounds = percentile_bounds
        self.max_tiles_per_volume = max_tiles_per_volume
        self.max_tiles_total = max_tiles_total

        rng = random.Random(seed)
        self.volume_paths = self._resolve_manifest(manifest)
        if not self.volume_paths:
            raise ValueError("Manifest did not yield any volume paths")
        rng.shuffle(self.volume_paths)
        self.volume_metadata = {path: load_volume_metadata(path) for path in self.volume_paths}

        self.tile_index: List[TileCoordinate] = []
        for path in self.volume_paths:
            coords = self._enumerate_tiles(path)
            rng.shuffle(coords)
            if self.max_tiles_per_volume is not None and len(coords) > self.max_tiles_per_volume:
                coords = coords[: self.max_tiles_per_volume]
            self.tile_index.extend(coords)
            if self.max_tiles_total is not None and len(self.tile_index) >= self.max_tiles_total:
                self.tile_index = self.tile_index[: self.max_tiles_total]
                break
        if not self.tile_index:
            raise ValueError("No tiles enumerated from the provided volumes")

    @staticmethod
    def _resolve_manifest(manifest: Path | str | Sequence[str]) -> List[Path]:
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if manifest_path.is_file():
                lines = [line.strip() for line in manifest_path.read_text().splitlines()]
            else:
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        else:
            lines = [str(p).strip() for p in manifest]

        volume_paths: List[Path] = []
        for line in lines:
            if not line:
                continue
            path = Path(line)
            if path.suffix != ".zarr":
                raise ValueError(f"Expected Zarr suffix for volume path, got {path}")
            if not path.exists():
                raise FileNotFoundError(f"Volume path does not exist: {path}")
            volume_paths.append(path)
        return volume_paths

    def _enumerate_tiles(self, volume_path: Path) -> List[TileCoordinate]:
        grp = zarr.open_group(str(volume_path), mode="r")
        raw = grp["raw"]
        z, y, x = raw.shape
        tile_h, tile_w = self.tile_hw
        stride_h, stride_w = self.tile_stride

        y_steps = max(1, math.ceil((y - tile_h) / stride_h) + 1)
        x_steps = max(1, math.ceil((x - tile_w) / stride_w) + 1)

        coords: List[TileCoordinate] = []
        for zi in range(z):
            for yi in range(y_steps):
                y0 = min(yi * stride_h, y - tile_h)
                for xi in range(x_steps):
                    x0 = min(xi * stride_w, x - tile_w)
                    coords.append(
                        TileCoordinate(
                            volume_path=volume_path,
                            z_index=zi,
                            y0=y0,
                            x0=x0,
                            tile_hw=self.tile_hw,
                        )
                    )
        return coords

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.tile_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        coord = self.tile_index[index]
        grp = zarr.open_group(str(coord.volume_path), mode="r")

        raw = grp["raw"]
        slices = self._gather_slices(raw, coord.z_index)
        tile = slices[:, coord.y0 : coord.y0 + coord.tile_hw[0], coord.x0 : coord.x0 + coord.tile_hw[1]]
        image = self._normalize(tile)

        item: Dict[str, torch.Tensor] = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "meta": torch.tensor([coord.z_index, coord.y0, coord.x0], dtype=torch.int32),
            "voxel_size": torch.tensor(self._voxel_size_xyz(coord.volume_path), dtype=torch.float32),
        }

        if not self.ssl_mode:
            labels = self._load_labels(grp, coord)
            item.update(labels)
        return item

    def _gather_slices(self, raw: Any, z_index: int) -> np.ndarray:
        num_channels = self.k_neighbors * 2 + 1
        z = raw.shape[0]
        indices = [
            int(np.clip(z_index + offset, 0, z - 1))
            for offset in range(-self.k_neighbors, self.k_neighbors + 1)
        ]
        stack = raw.get_orthogonal_selection((indices, slice(None), slice(None)))
        return np.asarray(stack)

    def _normalize(self, tile: np.ndarray) -> np.ndarray:
        lo, hi = self.percentile_bounds
        flat = tile.reshape(tile.shape[0], -1)
        p_low = np.percentile(flat, lo, axis=1, keepdims=True)
        p_high = np.percentile(flat, hi, axis=1, keepdims=True)
        denom = np.maximum(p_high - p_low, 1e-6)
        norm = (flat - p_low) / denom
        norm = norm.reshape(tile.shape)
        return np.clip(norm, 0.0, 1.0)

    def _load_labels(self, grp: Any, coord: TileCoordinate) -> Dict[str, torch.Tensor]:
        y0, x0 = coord.y0, coord.x0
        h, w = coord.tile_hw
        label_dict: Dict[str, torch.Tensor] = {}

        shaft_key = "labels/shafts_semantic"
        if self.use_pseudolabels and "labels/pseudolabels/shafts_semantic" in grp:
            shaft_key = "labels/pseudolabels/shafts_semantic"
        if shaft_key in grp:
            shaft_arr = grp[shaft_key][coord.z_index, y0 : y0 + h, x0 : x0 + w]
            label_dict["shaft_mask"] = torch.from_numpy(shaft_arr.astype(np.float32, copy=False)).unsqueeze(0)
        else:
            label_dict["shaft_mask"] = torch.zeros((1, h, w), dtype=torch.float32)

        spine_key = "labels/spines_instance"
        if self.use_pseudolabels and "labels/pseudolabels/spines_instance" in grp:
            spine_key = "labels/pseudolabels/spines_instance"
        if spine_key in grp:
            spine_arr = grp[spine_key][coord.z_index, y0 : y0 + h, x0 : x0 + w]
            label_dict["spine_instance"] = torch.from_numpy(spine_arr.astype(np.int32, copy=False))
        else:
            label_dict["spine_instance"] = torch.zeros((h, w), dtype=torch.int32)

        center_key = "labels/spine_center_heatmap"
        if self.use_pseudolabels and "labels/pseudolabels/spine_center_heatmap" in grp:
            center_key = "labels/pseudolabels/spine_center_heatmap"
        if center_key in grp:
            center_arr = grp[center_key][coord.z_index, y0 : y0 + h, x0 : x0 + w]
            label_dict["spine_center_heatmap"] = torch.from_numpy(
                center_arr.astype(np.float32, copy=False)
            ).unsqueeze(0)
        else:
            label_dict["spine_center_heatmap"] = torch.zeros((1, h, w), dtype=torch.float32)

        return label_dict

    def _voxel_size_xyz(self, volume_path: Path) -> Tuple[float, float, float]:
        """Return voxel size in microns ordered as (X, Y, Z) with safe defaults."""
        meta = self.volume_metadata.get(volume_path) or {}
        voxel_size = meta.get("voxel_size_um") if isinstance(meta, dict) else None
        try:
            arr = np.asarray(voxel_size, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        if arr.size != 3:
            arr = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        # Stored order is typically (Z, Y, X); return (X, Y, Z) for consistency.
        x, y, z = float(arr[2]), float(arr[1]), float(arr[0])
        return (x, y, z)


def load_volume_metadata(volume_path: Path) -> Dict[str, object]:
    """Utility to read ``meta.json`` and fall back to Zarr attrs when present."""

    meta: Dict[str, object] = {}
    meta_path = Path(volume_path) / "meta.json"
    if meta_path.exists():
        with meta_path.open("r") as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                meta = {}

    try:
        grp = zarr.open_group(str(volume_path), mode="r")
        raw = grp.get("raw")
        if raw is not None:
            attrs = getattr(raw, "attrs", {})
            attr_voxel = attrs.get("voxel_size_um")
            if attr_voxel is not None and "voxel_size_um" not in meta:
                meta["voxel_size_um"] = attr_voxel
    except Exception:  # pragma: no cover - defensive against missing files
        pass

    return meta
