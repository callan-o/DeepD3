"""Generate pseudo-labels via the pretrained DeepD3 TensorFlow model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - heavy dependencies
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required for DeepD3 pseudolabel generation. Install tensorflow>=2.15."
    ) from exc

try:  # pragma: no cover - optional deps bundled with the project env
    import zarr
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("Please install zarr to read converted DeepD3 volumes.") from exc

try:  # pragma: no cover - scientific stack
    from scipy import ndimage
    from skimage import morphology
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("scipy and scikit-image are required for morphology post-processing.") from exc

try:  # pragma: no cover - compression filters are optional but preferred
    from zarr.codecs import Blosc as ZarrBlosc
    BLOSC_FACTORY = lambda: ZarrBlosc(cname="zstd", clevel=5, shuffle=2)  # noqa: E731
except ModuleNotFoundError:  # pragma: no cover
    try:
        from numcodecs import Blosc

        BLOSC_FACTORY = lambda: Blosc(cname="zstd", clevel=5, shuffle=2)  # noqa: E731
    except ModuleNotFoundError:
        BLOSC_FACTORY = None

try:
    from transd3.spine_data.utils_io import read_json, write_json
except ModuleNotFoundError:
    # When executing the script as a file, the package root may not be on sys.path.
    # Insert the repository root (/.../TransD3) so 'import transd3' resolves.
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from transd3.spine_data.utils_io import read_json, write_json


DEFAULT_MODEL = Path("models/DeepD3_32F.h5")


@dataclass
class PseudolabelConfig:
    manifest: Path
    output_root: Path
    model_path: Path
    batch_size: int
    shaft_threshold: float
    spine_threshold: float
    min_spine_voxels: int
    area_min: int
    area_max: int
    elongation_max: float
    discard_border: bool
    center_sigma: float
    center_clip: float
    overwrite: bool
    save_probabilities: bool


@dataclass
class VolumeStats:
    volume: Path
    has_original_labels: bool
    num_voxels_shaft: int
    num_voxels_spine: int
    num_instances: int


def parse_args(argv: Sequence[str] | None = None) -> PseudolabelConfig:
    parser = argparse.ArgumentParser(description="Run DeepD3 inference and store pseudolabels.")
    parser.add_argument("--manifest", type=Path, required=True, help="Text file listing Zarr volumes.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for provenance and summary artifacts (arrays are written in-place).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the DeepD3 Keras checkpoint (.h5). Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Number of slices per TF batch.")
    parser.add_argument("--shaft-threshold", type=float, default=0.5, help="Probability threshold for shafts.")
    parser.add_argument("--spine-threshold", type=float, default=0.5, help="Probability threshold for spines.")
    parser.add_argument("--min-spine-voxels", type=int, default=24, help="Minimum 3D voxels per spine instance.")
    parser.add_argument("--area-min", type=int, default=3, help="Minimum 2D area (px) for spine head region.")
    parser.add_argument("--area-max", type=int, default=150, help="Maximum 2D area (px) for spine head region.")
    parser.add_argument(
        "--elongation-max",
        type=float,
        default=6.0,
        help="Maximum elongation ratio (major_axis/minor_axis) allowed for spine head regions.",
    )
    parser.add_argument(
        "--discard-border",
        action="store_true",
        help="Discard spine instances that touch the border of the XY plane.",
    )
    parser.add_argument("--center-sigma", type=float, default=2.0, help="Gaussian sigma for center heatmaps.")
    parser.add_argument("--center-clip", type=float, default=1.0, help="Upper clamp for center heatmaps.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing pseudolabel arrays instead of aborting.",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Persist sigmoid probabilities alongside binarized masks.",
    )
    args = parser.parse_args(argv)
    return PseudolabelConfig(
        manifest=args.manifest,
        output_root=args.output,
        model_path=args.model_path,
        batch_size=max(1, args.batch_size),
        shaft_threshold=float(args.shaft_threshold),
        spine_threshold=float(args.spine_threshold),
        min_spine_voxels=max(1, int(args.min_spine_voxels)),
        area_min=int(args.area_min),
        area_max=int(args.area_max),
        elongation_max=float(args.elongation_max),
        discard_border=bool(args.discard_border),
        center_sigma=float(args.center_sigma),
        center_clip=float(args.center_clip),
        overwrite=bool(args.overwrite),
        save_probabilities=bool(args.save_probabilities),
    )


def resolve_manifest(manifest: Path) -> List[Path]:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    paths: List[Path] = []
    for line in manifest.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        path = Path(stripped)
        if not path.exists():
            raise FileNotFoundError(f"Volume listed in manifest does not exist: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"Manifest {manifest} did not list any volumes.")
    return paths


def load_deepd3_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"DeepD3 checkpoint not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    if len(model.outputs) != 2:
        raise ValueError("Expected DeepD3 model with two outputs (shafts, spines).")
    return model


def percentile_normalize(volume: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    normed = np.empty_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        slice_ = volume[z]
        low = np.percentile(slice_, lo)
        high = np.percentile(slice_, hi)
        denom = max(high - low, 1e-6)
        norm = (slice_ - low) / denom
        normed[z] = np.clip(norm, 0.0, 1.0)
    return normed


def run_inference(model: tf.keras.Model, volume: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    # Pad slices so spatial dims are compatible with the DeepD3 model (avoids concat shape mismatches).
    z, h, w = volume.shape

    def _next_mult(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    target_h = _next_mult(h, 32)
    target_w = _next_mult(w, 32)
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        # pad each slice (z, h, w) -> ((0,0),(pad_top,pad_bottom),(pad_left,pad_right))
        padded = np.pad(
            volume,
            pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )
    else:
        padded = volume

    inputs = padded.astype(np.float32)[..., np.newaxis]
    dataset = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
    shaft_probs: List[np.ndarray] = []
    spine_probs: List[np.ndarray] = []
    for batch in dataset:
        preds = model(batch, training=False)
        shaft_batch, spine_batch = preds
        shaft_probs.append(shaft_batch.numpy())
        spine_probs.append(spine_batch.numpy())
    shaft = np.concatenate(shaft_probs, axis=0)[..., 0]
    spines = np.concatenate(spine_probs, axis=0)[..., 0]

    # Crop back to original spatial size if we padded
    if pad_h > 0 or pad_w > 0:
        shaft = shaft[:, pad_top : pad_top + h, pad_left : pad_left + w]
        spines = spines[:, pad_top : pad_top + h, pad_left : pad_left + w]

    return shaft, spines


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    filtered = morphology.remove_small_objects(mask.astype(bool), min_size=min_voxels, connectivity=3)
    return np.asarray(filtered, dtype=bool)


def label_instances(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, count = ndimage.label(mask, structure=structure)
    return labeled.astype(np.int32, copy=False), int(count)


def gaussian_heatmap(height: int, width: int, center_y: float, center_x: float, sigma: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * sigma ** 2))


def build_center_heatmap(instances: np.ndarray, sigma: float, clip_value: float) -> np.ndarray:
    heatmap = np.zeros_like(instances, dtype=np.float32)
    if instances.max() == 0:
        return heatmap
    coords = ndimage.find_objects(instances)
    for label_id, bbox in enumerate(coords, start=1):
        if bbox is None:
            continue
        region = instances[bbox] == label_id
        if not region.any():
            continue
        rel_indices = np.argwhere(region)
        z_vals = rel_indices[:, 0]
        y_vals = rel_indices[:, 1]
        x_vals = rel_indices[:, 2]
        z_center = int(round(z_vals.mean()))
        y_center = y_vals.mean()
        x_center = x_vals.mean()
        z_abs = bbox[0].start + z_center
        y_abs = bbox[1].start + y_center
        x_abs = bbox[2].start + x_center
        gauss = gaussian_heatmap(
            heatmap.shape[1],
            heatmap.shape[2],
            y_abs,
            x_abs,
            sigma,
        )
        heatmap_slice = heatmap[z_abs]
        np.maximum(heatmap_slice, gauss, out=heatmap_slice)
        heatmap[z_abs] = heatmap_slice
    if clip_value > 0:
        np.clip(heatmap, 0.0, clip_value, out=heatmap)
    return heatmap


def default_compressors() -> Sequence[object] | None:
    if BLOSC_FACTORY is None:
        return None
    try:
        return [BLOSC_FACTORY()]
    except Exception:  # pragma: no cover - codec instantiation failure
        return None


def ensure_array(group: zarr.Group, name: str, data: np.ndarray, chunks: Sequence[int], overwrite: bool, compressors: Sequence[object] | None) -> None:
    if name in group:
        if not overwrite:
            print(f"Notice: array already exists, skipping: {group.name}/{name} (use --overwrite to replace)")
            return
        # remove existing array so we can recreate it
        try:
            del group[name]
        except Exception:
            # best-effort deletion; if it fails, warn and skip
            print(f"Warning: failed to delete existing array {group.name}/{name}; skipping replacement")
            return
    group.create_array(name, data=data, chunks=chunks, compressors=compressors)


def update_meta(volume: Path, provenance: Dict[str, object]) -> None:
    meta_path = volume / "meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    pseudolabels = meta.get("pseudolabels", {})
    pseudolabels["deepd3"] = provenance
    meta["pseudolabels"] = pseudolabels
    write_json(meta_path, meta)


# New helper: locate the appropriate 3D input array inside a zarr group.
def find_raw_array(root: zarr.Group) -> zarr.Array:
    """
    Return a zarr.Array to use as 'raw'. Preference order:
      1. If the opened object is already an Array, return it.
      2. root['raw'] if present
      3. first 3D array found anywhere in the tree (recursive)
      4. if no 3D arrays but some arrays exist, warn and return the first array found
    Raises KeyError with a helpful message if no arrays are present at all.
    """
    # 0) If the opened object is itself an Array
    if isinstance(root, zarr.Array):
        return root

    # 1) explicit 'raw'
    if "raw" in root and isinstance(root["raw"], zarr.Array):
        return root["raw"]

    # 2) recursive search for arrays (collect candidates)
    candidates: List[Tuple[str, zarr.Array]] = []

    def _recurse(group: zarr.Group, prefix: str = "") -> None:
        for key, item in group.items():
            full = f"{prefix}/{key}" if prefix else key
            if isinstance(item, zarr.Array):
                candidates.append((full, item))
            elif isinstance(item, zarr.Group):
                _recurse(item, full)

    _recurse(root)

    # Prefer the first 3D array found
    for name, arr in candidates:
        if getattr(arr, "ndim", None) == 3:
            return arr

    # If there are arrays but none are 3D, pick the first and warn
    if candidates:
        name, arr = candidates[0]
        print(f"Warning: no 3D array found; falling back to first array '{name}' with shape={getattr(arr, 'shape', None)}")
        return arr

    # nothing found — prepare a helpful error
    msg_lines = ["No arrays found in Zarr volume. Available arrays (name -> shape):"]
    # attempt one more listing to show nothing present
    for key, item in root.items():
        if isinstance(item, zarr.Array):
            msg_lines.append(f"  - {key} -> {getattr(item, 'shape', None)}")
        elif isinstance(item, zarr.Group):
            for ck, ci in item.items():
                if isinstance(ci, zarr.Array):
                    msg_lines.append(f"  - {key}/{ck} -> {getattr(ci, 'shape', None)}")
    if len(msg_lines) == 1:
        msg_lines.append("  (no arrays present in this Zarr root)")
    raise KeyError("\n".join(msg_lines))


def process_volume(volume: Path, cfg: PseudolabelConfig, model: tf.keras.Model) -> VolumeStats:
    try:
        # zarr.open can raise zarr.errors.PathNotFoundError (or other errors) when passed an invalid/empty path.
        root = zarr.open(str(volume), mode="r+")
    except Exception as exc:  # defensive: don't let one bad manifest entry stop the whole run
        print(f"Warning: skipping volume {volume!s} - failed to open Zarr store: {exc}")
        return VolumeStats(volume=volume, has_original_labels=False, num_voxels_shaft=0, num_voxels_spine=0, num_instances=0)

    # Use robust discovery of the raw array instead of requiring 'raw' key.
    try:
        raw = find_raw_array(root)
        print(f"    using array '{raw.name}' as input (shape={getattr(raw, 'shape', None)})")
    except KeyError as exc:
        # If the opened root has no arrays, try to find nested .zarr stores under the provided path.
        nested_candidates = list(Path(volume).rglob("*.zarr"))
        if nested_candidates:
            nested = nested_candidates[0]
            print(f"    notice: no arrays at top-level; trying nested Zarr store: {nested}")
            nested_root = zarr.open(str(nested), mode="r+")
            try:
                raw = find_raw_array(nested_root)
                root = nested_root  # switch to nested root for subsequent writes/meta updates
                print(f"    using array '{raw.name}' in nested store '{nested}' (shape={getattr(raw, 'shape', None)})")
            except KeyError:
                # nested store also had no arrays -> skip this volume but continue run
                print(f"Warning: skipping volume {volume} - no arrays found (checked nested store {nested}).")
                return VolumeStats(volume=volume, has_original_labels=False, num_voxels_shaft=0, num_voxels_spine=0, num_instances=0)
        else:
            # no nested .zarr stores found -> skip this volume but continue run
            print(f"Warning: skipping volume {volume} - no arrays found. Details: {exc}")
            return VolumeStats(volume=volume, has_original_labels=False, num_voxels_shaft=0, num_voxels_spine=0, num_instances=0)

    raw_np = raw[...].astype(np.float32, copy=False)
    norm_raw = percentile_normalize(raw_np)
    shaft_prob, spine_prob = run_inference(model, norm_raw, cfg.batch_size)

    shaft_mask = (shaft_prob >= cfg.shaft_threshold).astype(np.uint8)
    spine_mask = spine_prob >= cfg.spine_threshold
    spine_mask = remove_small_components(spine_mask, cfg.min_spine_voxels)
    spine_instances, num_instances = label_instances(spine_mask)

    # Post-filter 3D instances by 2D shape criteria on their largest-slice (head region)
    try:
        from skimage import measure
    except Exception:
        measure = None

    if measure is not None and num_instances > 0:
        keep_map = np.zeros_like(spine_instances, dtype=bool)
        for inst_id in range(1, num_instances + 1):
            inst_mask = spine_instances == inst_id
            # find slice with max area (head region)
            areas = inst_mask.reshape(inst_mask.shape[0], -1).sum(axis=1)
            if areas.max() == 0:
                continue
            z_idx = int(np.argmax(areas))
            slice_mask = inst_mask[z_idx]
            props = measure.regionprops(slice_mask.astype(int))
            if not props:
                continue
            prop = props[0]
            area = prop.area
            # elongation: major_axis_length / minor_axis_length (guard against zero)
            maj = getattr(prop, "major_axis_length", 0.0)
            mino = getattr(prop, "minor_axis_length", 1.0)
            elong = (maj / (mino + 1e-6)) if mino > 0 else float("inf")

            # border touch check
            y0, x0, y1, x1 = prop.bbox
            touches_border = (
                y0 == 0 or x0 == 0 or y1 == slice_mask.shape[0] or x1 == slice_mask.shape[1]
            )

            if (
                area >= cfg.area_min
                and area <= cfg.area_max
                and elong <= cfg.elongation_max
                and (not (cfg.discard_border and touches_border))
            ):
                keep_map |= inst_mask

        # Relabel kept instances
        new_instances = np.zeros_like(spine_instances, dtype=np.int32)
        next_id = 1
        for inst_id in range(1, num_instances + 1):
            mask = (spine_instances == inst_id) & keep_map
            if not mask.any():
                continue
            new_instances[mask] = next_id
            next_id += 1
        spine_instances = new_instances
        num_instances = next_id - 1
    center_map = build_center_heatmap(spine_instances, cfg.center_sigma, cfg.center_clip)

    compressors = default_compressors()
    # Decide whether to write into primary labels or pseudolabels namespace
    chunks = raw.chunks
    labels_group = root.require_group("labels")
    write_primary = not ("shafts_semantic" in labels_group or "spines_instance" in labels_group)
    if write_primary:
        target_group = labels_group
    else:
        target_group = labels_group.require_group("pseudolabels")

    ensure_array(target_group, "shafts_semantic", shaft_mask.astype(np.uint8), chunks, cfg.overwrite, compressors)
    ensure_array(target_group, "spines_instance", spine_instances, chunks, cfg.overwrite, compressors)
    ensure_array(target_group, "spine_center_heatmap", center_map.astype(np.float32), chunks, cfg.overwrite, compressors)

    if cfg.save_probabilities:
        ensure_array(target_group, "shafts_probability", shaft_prob.astype(np.float32), chunks, cfg.overwrite, compressors)
        ensure_array(target_group, "spines_probability", spine_prob.astype(np.float32), chunks, cfg.overwrite, compressors)

    provenance = {
        "model_path": str(cfg.model_path.resolve()),
        "shaft_threshold": cfg.shaft_threshold,
        "spine_threshold": cfg.spine_threshold,
        "min_spine_voxels": cfg.min_spine_voxels,
        "center_sigma": cfg.center_sigma,
        "center_clip": cfg.center_clip,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "command": " ".join([sys.executable, *sys.argv]),
    }
    update_meta(volume, provenance)

    has_gt = "labels/spines_instance" in root and np.any(root["labels/spines_instance"][...])
    stats = VolumeStats(
        volume=volume,
        has_original_labels=bool(has_gt),
        num_voxels_shaft=int(shaft_mask.sum()),
        num_voxels_spine=int(spine_mask.sum()),
        num_instances=num_instances,
    )
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    cfg = parse_args(argv)
    volumes = resolve_manifest(cfg.manifest)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading DeepD3 model from {cfg.model_path} ...")
    model = load_deepd3_model(cfg.model_path)

    all_stats: List[VolumeStats] = []
    for idx, volume in enumerate(volumes, start=1):
        print(f"[{idx}/{len(volumes)}] Generating pseudolabels for {volume} ...")
        stats = process_volume(volume, cfg, model)
        all_stats.append(stats)
        print(
            f"    shaft voxels={stats.num_voxels_shaft:,} | spine voxels={stats.num_voxels_spine:,} | instances={stats.num_instances}"
        )

    summary = {
        "config": {
            "model_path": str(cfg.model_path.resolve()),
            "shaft_threshold": cfg.shaft_threshold,
            "spine_threshold": cfg.spine_threshold,
            "min_spine_voxels": cfg.min_spine_voxels,
            "center_sigma": cfg.center_sigma,
            "center_clip": cfg.center_clip,
            "batch_size": cfg.batch_size,
            "save_probabilities": cfg.save_probabilities,
        },
        "volumes": [
            {
                "path": str(stat.volume.resolve()),
                "has_gt_labels": stat.has_original_labels,
                "shaft_voxels": stat.num_voxels_shaft,
                "spine_voxels": stat.num_voxels_spine,
                "spine_instances": stat.num_instances,
            }
            for stat in all_stats
        ],
    }

    summary_path = cfg.output_root / "pseudolabel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    manifest_out = cfg.output_root / "pseudolabel_manifest.txt"
    manifest_out.write_text("\n".join(str(stat.volume.resolve()) for stat in all_stats) + "\n")
    print(f"Wrote pseudolabel summary to {summary_path}")


if __name__ == "__main__":
    main()
