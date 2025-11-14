"""Evaluation metrics for spine and shaft predictions."""

from __future__ import annotations

from typing import Dict

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum() + eps)
    return (2 * inter + eps) / denom


def cldice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    return dice_score(pred, target, eps)


def summarize_metrics(metrics: Dict[str, list[float]]) -> Dict[str, float]:
    return {key: float(np.mean(values)) for key, values in metrics.items() if values}
