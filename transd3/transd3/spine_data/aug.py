"""Augmentation utilities for TransD3 datasets."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def random_flip(tile: np.ndarray, *, seed: int | None = None) -> np.ndarray:
    """Apply deterministic horizontal/vertical flips given an optional seed."""

    rng = np.random.default_rng(seed)
    result = tile
    if rng.uniform() < 0.5:
        result = result[..., ::-1]
    if rng.uniform() < 0.5:
        result = result[..., ::-1, :]
    return np.ascontiguousarray(result)


def normalize_percentile(
    tile: np.ndarray, lower: float = 1.0, upper: float = 99.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute percentile normalization for logging without mutating input."""

    tile = tile.astype(np.float32, copy=True)
    lo = np.percentile(tile, lower)
    hi = np.percentile(tile, upper)
    tile -= lo
    denom = max(hi - lo, 1e-6)
    tile /= denom
    return np.clip(tile, 0.0, 1.0), {"p{:.0f}".format(lower): float(lo), "p{:.0f}".format(upper): float(hi)}
