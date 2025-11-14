"""Evaluate a pretrained MAE model on TransD3 two-photon data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None
    HAS_MATPLOTLIB = False
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from transd3.data.datamodule import DataConfig, SpineDataModule
from transd3.models.mae.mae_2p import LitMAE2P, MAEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAE checkpoints.")
    parser.add_argument("checkpoint", type=Path, help="Path to Lightning checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("transd3/configs/mae_2p.yaml"),
        help="Model/data config used for training",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val"),
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/mae_eval"),
        help="Directory for metrics and visualizations",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Optional OmegaConf overrides in key=value form",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit the number of batches to evaluate",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=12,
        help="Number of example tiles to save for qualitative inspection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device (cuda or cpu)",
    )
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    merged = OmegaConf.create(cfg)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        merged = OmegaConf.merge(merged, override_cfg)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


@torch.no_grad()
def collect_examples(
    images: torch.Tensor,
    recon: torch.Tensor,
    masks: torch.Tensor,
    metas: torch.Tensor,
    limit: int,
) -> Dict[str, np.ndarray]:
    qty = min(limit, images.size(0))
    idx = torch.arange(qty)
    return {
        "images": images[idx].cpu().numpy(),
        "reconstructions": recon[idx].cpu().numpy(),
        "masks": masks[idx].cpu().numpy(),
        "meta": metas[idx].cpu().numpy(),
    }


def append_metrics(store: Dict[str, List[float]], batch_metrics: Dict[str, torch.Tensor]) -> None:
    for key, tensor in batch_metrics.items():
        store.setdefault(key, []).extend(tensor.detach().cpu().tolist())


def summarize_metrics(metric_store: Dict[str, List[float]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, values in metric_store.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": int(arr.size),
        }
    return summary


@torch.no_grad()
def evaluate(model: LitMAE2P, dataloader: torch.utils.data.DataLoader, max_batches: int | None) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray]]:
    metric_store: Dict[str, List[float]] = {}
    example_store: Dict[str, np.ndarray] | None = None

    progress = tqdm(enumerate(dataloader, start=1), total=len(dataloader) if hasattr(dataloader, "__len__") else None, desc="Evaluating", leave=False)
    for batch_idx, batch in progress:
        images = batch["image"].to(model.device)
        recon, aux = model(images)
        mask = aux.get("mask")
        if mask is None:
            mask = torch.zeros_like(images[:, 0:1, ...], dtype=torch.bool, device=images.device)

        mse = F.mse_loss(recon, images, reduction="none").flatten(1).mean(1)
        mae = F.l1_loss(recon, images, reduction="none").flatten(1).mean(1)
        psnr = 10.0 * torch.log10(torch.clamp(1.0 / mse, min=1e-12))
        append_metrics(metric_store, {"mse": mse, "mae": mae, "psnr": psnr})

        if example_store is None:
            example_store = collect_examples(images, recon, mask, batch["meta"], limit=64)

        if max_batches is not None and batch_idx >= max_batches:
            break

    if example_store is None:
        raise RuntimeError("No batches were processed during evaluation.")

    return metric_store, example_store


def save_examples(example_store: Dict[str, np.ndarray], output_dir: Path, num_examples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "examples.npz"
    np.savez_compressed(npz_path, **example_store)

    if not HAS_MATPLOTLIB:
        print("matplotlib not available: saved examples as a compressed .npz but not PNGs")
        return

    center_idx = example_store["images"].shape[1] // 2
    for idx in range(min(num_examples, example_store["images"].shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        target = example_store["images"][idx, center_idx]
        recon = example_store["reconstructions"][idx, center_idx]
        residual = np.abs(recon - target)
        for ax, data, title in zip(
            axes,
            (target, recon, residual),
            ("Target", "Reconstruction", "|Residual|"),
        ):
            ax.imshow(data, cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"example_{idx:03d}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg_dict = apply_overrides(OmegaConf.to_container(cfg, resolve=True), args.overrides)
    cfg = OmegaConf.create(cfg_dict)

    data_cfg = DataConfig(
        train_manifest=Path(cfg.data.train_manifest),
        val_manifest=Path(cfg.data.val_manifest),
        batch_size=cfg.data.get("batch_size", 8),
        num_workers=cfg.data.get("num_workers", 0),
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
    datamodule.setup("fit")
    dataloader = (
        datamodule.val_dataloader() if args.split == "val" else datamodule.train_dataloader()
    )

    mae_cfg = MAEConfig(**cfg.model)
    # Attempt to load Lightning checkpoint; if checkpoint was saved from a
    # different MAE architecture, fall back to a non-strict state_dict load
    # so we can still run inference with compatible parameters.
    try:
        model = LitMAE2P.load_from_checkpoint(
            args.checkpoint,
            cfg=mae_cfg,
            map_location="cpu",
        )
    except Exception as exc:  # pragma: no cover - compatibility fallback
        print(f"Checkpoint load_with_strict failed ({exc}), attempting non-strict state_dict load.")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        model = LitMAE2P(mae_cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Warning] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys in checkpoint: {list(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}")
    model.eval()
    model.to(args.device)

    metric_store, example_store = evaluate(model, dataloader, args.max_batches)
    summary = summarize_metrics(metric_store)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    save_examples(example_store, args.output, args.num_examples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
