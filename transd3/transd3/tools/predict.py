"""Inference CLI for TransD3 segmentation models."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


SpineSegmentationModel = getattr(
    importlib.import_module("transd3.models.seg.heads"), "SpineSegmentationModel"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=2)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    return parser.parse_args()


def load_volume(path: Path) -> np.ndarray:
    if path.suffix == ".zarr":
        zarr = importlib.import_module("zarr")
        grp = zarr.open_group(str(path), mode="r")
        return np.asarray(grp["raw"])  # type: ignore[index]
    tifffile = importlib.import_module("tifffile")
    return tifffile.imread(path)


def save_zarr(path: Path, name: str, array: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    store = importlib.import_module("zarr").DirectoryStore(str(path / f"{name}.zarr"))
    root = importlib.import_module("zarr").group(store=store, overwrite=True)
    root.create_dataset("data", data=array, chunks=(1, 256, 256), overwrite=True)


def center_slices(volume: np.ndarray, idx: int, neighbors: int) -> np.ndarray:
    indices = [np.clip(idx + offset, 0, volume.shape[0] - 1) for offset in range(-neighbors, neighbors + 1)]
    return volume[indices]


def main() -> None:
    args = parse_args()
    volume = load_volume(args.input).astype(np.float32)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    in_ch = args.neighbors * 2 + 1
    model = SpineSegmentationModel(in_ch=in_ch, dim=128, depth=6, num_queries=400)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    shaft_probs = np.zeros_like(volume)
    spine_instances = np.zeros_like(volume, dtype=np.int32)
    center_heatmaps = np.zeros_like(volume)

    with torch.no_grad():
        for z in range(volume.shape[0]):
            tile = center_slices(volume, z, args.neighbors)
            tile = tile[np.newaxis]
            tensor = torch.from_numpy(tile)
            outputs = model(tensor)
            shaft_probs[z] = torch.sigmoid(outputs["logits_shaft"]).squeeze().numpy()
            center_heatmaps[z] = outputs["center_map"].squeeze().numpy()
            spine_instances[z] = (outputs["masks_spine"].amax(dim=1) > 0.5).squeeze().numpy().astype(np.int32)

    save_zarr(args.output, "shaft_prob", shaft_probs)
    save_zarr(args.output, "spine_instances", spine_instances)
    save_zarr(args.output, "center_heatmap", center_heatmaps)


if __name__ == "__main__":
    main()
