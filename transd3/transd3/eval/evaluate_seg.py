"""Evaluate fine-tuned segmentation model on test manifest."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from transd3.eval.metrics import cldice_score, dice_score, summarize_metrics

SpineSegmentationModel = getattr(
    importlib.import_module("transd3.models.seg.heads"), "SpineSegmentationModel"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/test_metrics.json"))
    return parser.parse_args()


def load_manifest(manifest: Path) -> list[Path]:
    return [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()]


def load_volume(path: Path) -> Dict[str, np.ndarray]:
    zarr = importlib.import_module("zarr")
    grp = zarr.open_group(str(path), mode="r")
    volume = np.asarray(grp["raw"])
    shafts = np.asarray(grp.get("labels/shafts_semantic", np.zeros_like(volume)))
    spines = np.asarray(grp.get("labels/spines_instance", np.zeros_like(volume)))
    return {"raw": volume, "shafts": shafts, "spines": spines}


def gather_slices(volume: np.ndarray, idx: int, neighbors: int = 2) -> np.ndarray:
    indices = [
        int(np.clip(idx + offset, 0, volume.shape[0] - 1))
        for offset in range(-neighbors, neighbors + 1)
    ]
    return volume[indices]


def evaluate_volume(model: torch.nn.Module, volume: Dict[str, np.ndarray]) -> Dict[str, float]:
    raw = volume["raw"].astype(np.float32)
    shafts_gt = volume["shafts"]
    spines_gt = volume["spines"]

    dice_scores = []
    cldice_scores = []
    ap_scores = []

    with torch.no_grad():
        for z in range(raw.shape[0]):
            tile = gather_slices(raw, z)
            tensor = torch.from_numpy(tile[np.newaxis])
            outputs = model(tensor)
            shaft_prob = torch.sigmoid(outputs["logits_shaft"]).squeeze().numpy()
            shaft_pred = (shaft_prob > 0.5).astype(np.float32)
            dice_scores.append(dice_score(shaft_pred, shafts_gt[z]))
            cldice_scores.append(cldice_score(shaft_pred, shafts_gt[z]))
            pred_spine = (outputs["masks_spine"].amax(dim=1) > 0.5).squeeze().numpy().astype(np.float32)
            gt_spine = (spines_gt[z] > 0).astype(np.float32)
            inter = (pred_spine * gt_spine).sum()
            ap_scores.append(inter / (gt_spine.sum() + 1e-6))

    return {
        "dice_shaft": float(np.mean(dice_scores)),
        "cldice": float(np.mean(cldice_scores)),
        "ap_small@0.3": float(np.mean(ap_scores)),
    }


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model = SpineSegmentationModel(in_ch=5, dim=128, depth=6, num_queries=400)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    results = {}
    metrics_accum: Dict[str, list[float]] = {}
    for path in manifest:
        volume = load_volume(path)
        metrics = evaluate_volume(model, volume)
        results[str(path)] = metrics
        for key, value in metrics.items():
            metrics_accum.setdefault(key, []).append(value)

    summary = summarize_metrics(metrics_accum)
    output = {"per_volume": results, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
