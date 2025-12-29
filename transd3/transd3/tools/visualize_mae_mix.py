"""Visualize MAE-mix reconstructions on validation tiles.

This tool loads the same config used for training (OmegaConf YAML + dotlist overrides),
instantiates the mixed MAE+N2V LightningModule, loads the checkpoint state dict,
then runs the MAE forward pass on validation tiles and writes PNG grids.

Example:
  python -m transd3.tools.visualize_mae_mix \
    --config transd3/configs/mae_2p.yaml \
    --ckpt ../../checkpoints/mae_mix/last.ckpt \
    --outdir ../../checkpoints/mae_mix/vis_last \
    data.val_manifest=/path/to/val.txt data.batch_size=2 data.num_workers=4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from transd3.data.datamodule import DataConfig, SpineDataModule
from transd3.models.mae.mae_2p import MAEConfig
from transd3.train.pretrain_mix import LitMAEN2VMix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Base YAML config (same schema as training).")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path (e.g., last.ckpt).")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for PNGs/CSV.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of validation tiles to visualize.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference.",
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=0,
        help="Optional hard limit on number of batches to iterate (0 = no limit).",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args()


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    merged = OmegaConf.create(cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(merged, resolve=True)


def _choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for visualization. Install it (e.g., pip install matplotlib)."
        ) from exc


def _to_image(x: torch.Tensor) -> torch.Tensor:
    # Ensure [H, W] float32 on CPU.
    if x.is_cuda:
        x = x.detach().cpu()
    else:
        x = x.detach()
    return x.float().clamp(0.0, 1.0)


def _masked_metrics(target: torch.Tensor, recon: torch.Tensor, mask01: torch.Tensor) -> Dict[str, float]:
    # target/recon/mask01 are [H,W] on CPU.
    diff = (recon - target).abs()
    mask = (mask01 > 0.5).float()
    inv = 1.0 - mask
    masked_mae = float((diff * mask).sum() / (mask.sum().clamp_min(1.0)))
    unmasked_mae = float((diff * inv).sum() / (inv.sum().clamp_min(1.0)))
    mae = float(diff.mean())
    mask_ratio = float(mask.mean())
    return {
        "mae": mae,
        "masked_mae": masked_mae,
        "unmasked_mae": unmasked_mae,
        "mask_ratio": mask_ratio,
    }


def _plot_tile(
    *,
    plt,
    target: torch.Tensor,
    masked_input: torch.Tensor,
    recon: torch.Tensor,
    mask01: torch.Tensor,
    metrics: Dict[str, float],
    title_prefix: str,
    outpath: Path,
) -> None:
    import numpy as np

    target_np = target.numpy()
    masked_np = masked_input.numpy()
    recon_np = recon.numpy()
    err_np = np.abs(recon_np - target_np)
    mask_np = mask01.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(target_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("target")
    axes[1].imshow(masked_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("masked input")
    axes[2].imshow(recon_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("recon")
    im = axes[3].imshow(err_np, cmap="magma")
    axes[3].set_title("|recon-target|")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{title_prefix}  mask={metrics['mask_ratio']:.3f}  "
        f"mae={metrics['mae']:.4f}  masked={metrics['masked_mae']:.4f}  unmasked={metrics['unmasked_mae']:.4f}",
        fontsize=10,
    )

    # Save mask overlay separately (cheap + helpful).
    overlay_path = outpath.with_name(outpath.stem + "_mask.png")
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    ax2.imshow(target_np, cmap="gray", vmin=0.0, vmax=1.0)
    ax2.imshow(mask_np, cmap="jet", alpha=0.35, vmin=0.0, vmax=1.0)
    ax2.set_title("mask overlay")
    ax2.set_xticks([])
    ax2.set_yticks([])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    fig2.savefig(overlay_path, dpi=150)
    plt.close(fig)
    plt.close(fig2)


def main() -> None:
    args = _parse_args()

    base_cfg = OmegaConf.load(args.config)
    cfg_dict = _apply_overrides(OmegaConf.to_container(base_cfg, resolve=True), args.overrides)
    cfg = OmegaConf.create(cfg_dict)

    device = _choose_device(args.device)
    plt = _ensure_matplotlib()

    model_cfg_raw = cfg.get("model") or {}
    model_cfg = OmegaConf.to_container(model_cfg_raw, resolve=True) if model_cfg_raw is not None else {}
    assert isinstance(model_cfg, dict)
    lam_n2v = float(model_cfg.pop("lam_n2v", 0.5))

    mae_cfg = MAEConfig(**model_cfg)

    data_cfg = DataConfig(
        train_manifest=Path(cfg.data.train_manifest),
        val_manifest=Path(cfg.data.val_manifest),
        batch_size=int(cfg.data.get("batch_size", 1)),
        num_workers=int(cfg.data.get("num_workers", 0)),
        k_neighbors=int(cfg.data.get("k_neighbors", 2)),
        tile_hw=tuple(cfg.data.get("tile_hw", [256, 256])),
        tile_stride=tuple(cfg.data.tile_stride) if cfg.data.get("tile_stride") is not None else None,
        ssl_mode=bool(cfg.data.get("ssl_mode", True)),
        use_pseudolabels=bool(cfg.data.get("use_pseudolabels", False)),
        prefetch_factor=int(cfg.data.get("prefetch_factor", 2)),
        max_tiles_per_volume=cfg.data.get("max_tiles_per_volume"),
        max_tiles_total=cfg.data.get("max_tiles_total"),
    )

    expected_in_ch = 2 * int(data_cfg.k_neighbors) + 1
    if int(mae_cfg.in_ch) != expected_in_ch:
        mae_cfg.in_ch = expected_in_ch

    optim_cfg = OmegaConf.to_container(cfg.optim, resolve=True) if cfg.get("optim") is not None else {}
    assert isinstance(optim_cfg, dict)

    model = LitMAEN2VMix(mae_cfg, lam_n2v=lam_n2v, optim_cfg=optim_cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    model.to(device)

    datamodule = SpineDataModule(data_cfg)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "metrics.csv"

    center_ch = int(data_cfg.k_neighbors)

    rows: List[Dict[str, Any]] = []
    seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break

            image = batch["image"].to(device=device)
            voxel_size = batch.get("voxel_size")
            if voxel_size is not None:
                voxel_size = voxel_size.to(device=device)

            recon, aux = model.mae(image, voxel_size)
            mask_full = aux.get("mask")
            if mask_full is None:
                raise RuntimeError("MAE forward did not return aux['mask']; cannot visualize masked recon.")

            for b in range(image.shape[0]):
                if seen >= args.num_samples:
                    break

                target = _to_image(image[b, center_ch])
                recon_c = _to_image(recon[b, center_ch])
                mask01 = _to_image(mask_full[b, 0])
                masked_input = target * (1.0 - mask01)

                metrics = _masked_metrics(target, recon_c, mask01)

                meta = batch.get("meta")
                meta_str = ""
                if meta is not None:
                    z, y0, x0 = (int(meta[b, 0]), int(meta[b, 1]), int(meta[b, 2]))
                    meta_str = f"z={z} y0={y0} x0={x0}"

                outpath = outdir / f"val_{seen:03d}.png"
                _plot_tile(
                    plt=plt,
                    target=target,
                    masked_input=masked_input,
                    recon=recon_c,
                    mask01=mask01,
                    metrics=metrics,
                    title_prefix=meta_str,
                    outpath=outpath,
                )

                row = {
                    "idx": seen,
                    "batch_idx": batch_idx,
                    "in_ch": mae_cfg.in_ch,
                    "center_ch": center_ch,
                    "tile_h": int(target.shape[0]),
                    "tile_w": int(target.shape[1]),
                    **metrics,
                    "meta": meta_str,
                    "png": str(outpath),
                }
                rows.append(row)
                seen += 1

            if seen >= args.num_samples:
                break

    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {seen} samples to {outdir}")
    print(f"Metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
