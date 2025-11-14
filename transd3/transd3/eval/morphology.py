"""Morphological helpers for dendritic spine predictions."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import importlib
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing placeholder
    from typing import Any as _Any

skeletonize = getattr(importlib.import_module("skimage.morphology"), "skeletonize")


def shaft_skeleton(shaft_mask: np.ndarray) -> np.ndarray:
    return skeletonize(shaft_mask.astype(bool)).astype(np.uint8)


def spine_density(spine_mask: np.ndarray, voxel_size_um: tuple[float, float]) -> float:
    area_px = spine_mask.sum()
    area_um2 = area_px * voxel_size_um[0] * voxel_size_um[1]
    return area_um2
