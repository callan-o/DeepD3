"""Stamp TIFF files with biological resolution metadata in-place."""

from __future__ import annotations

import argparse
import importlib
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    tifffile = importlib.import_module("tifffile")
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("Please install tifffile before running this script.") from exc

DEFAULT_SUFFIXES = (".tif", ".tiff")
# MICROMETERS_PER_CENTIMETER = 1e4
DIMENSION_PREFIXES = (
    "Width:",
    "Height:",
    "Depth:",
    "Size:",
    "Resolution:",
    "Voxel size:",
)
IMAGEJ_VALID_AXES = "TZCYXS"


@dataclass
class StampConfig:
    pixel_size_x: float
    pixel_size_y: float
    pixel_size_z: Optional[float]
    skip_existing: bool
    dry_run: bool
    backup_suffix: str
    overwrite_backup: bool


@dataclass
class DimensionPixels:
    width: int
    height: int
    depth: Optional[int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add biological resolution metadata to TIFF stacks in place."
    )
    parser.add_argument("--src", type=Path, required=True, help="Directory containing TIFF files.")
    parser.add_argument(
        "--pixel-size",
        type=float,
        help="Isotropic XY pixel size in micrometers (used when --pixel-size-x/--pixel-size-y are omitted).",
    )
    parser.add_argument("--pixel-size-x", type=float, help="Pixel size along X in micrometers.")
    parser.add_argument("--pixel-size-y", type=float, help="Pixel size along Y in micrometers.")
    parser.add_argument("--pixel-size-z", type=float, help="Optional slice spacing in micrometers.")
    parser.add_argument(
        "--suffixes",
        nargs="*",
        default=list(DEFAULT_SUFFIXES),
        help="Case-insensitive file suffixes to include (default: .tif .tiff).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for TIFFs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without rewriting any files.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Optional suffix used when keeping the original file (set '' to disable backups).",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Overwrite existing backup files if they already exist.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already contain the dimension summary block.",
    )
    return parser


def normalize_suffixes(raw: Sequence[str]) -> List[str]:
    suffixes: List[str] = []
    for entry in raw:
        if not entry:
            continue
        suffix = entry if entry.startswith(".") else f".{entry}"
        suffixes.append(suffix.lower())
    return sorted(set(suffixes))


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> StampConfig:
    if not args.src.exists():
        parser.error(f"--src {args.src} does not exist.")
    if not args.src.is_dir():
        parser.error("--src must point to a directory.")

    px_x = args.pixel_size_x if args.pixel_size_x is not None else args.pixel_size
    px_y = args.pixel_size_y if args.pixel_size_y is not None else args.pixel_size
    if px_x is None or px_y is None:
        parser.error("Provide --pixel-size for isotropic data or both --pixel-size-x and --pixel-size-y.")

    for label, value in (
        ("pixel-size-x", px_x),
        ("pixel-size-y", px_y),
        ("pixel-size-z", args.pixel_size_z),
    ):
        if value is not None and value <= 0:
            parser.error(f"--{label} must be positive.")
        if value is not None and not math.isfinite(value):
            parser.error(f"--{label} must be finite.")

    args.suffixes = normalize_suffixes(args.suffixes)
    if not args.suffixes:
        parser.error("At least one suffix must be provided.")

    backup_suffix = args.backup_suffix
    if backup_suffix and args.backup_suffix.strip() == "":
        backup_suffix = ""

    return StampConfig(
        pixel_size_x=float(px_x),
        pixel_size_y=float(px_y),
        pixel_size_z=float(args.pixel_size_z) if args.pixel_size_z is not None else None,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
        backup_suffix=backup_suffix,
        overwrite_backup=bool(args.overwrite_backup),
    )


def collect_files(src: Path, recursive: bool, suffixes: Sequence[str]) -> List[Path]:
    walker: Iterable[Path] = src.rglob("*") if recursive else src.glob("*")
    files: List[Path] = []
    for path in walker:
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files)


def sanitize_description(description: Optional[str]) -> str:
    if description is None:
        return ""
    if isinstance(description, bytes):
        description = description.decode("utf-8", errors="ignore")
    return description.replace("\x00", "").strip()


def has_dimension_block(description: str) -> bool:
    for line in description.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in DIMENSION_PREFIXES):
            return True
    return False


def strip_dimension_block(description: str) -> List[str]:
    kept: List[str] = []
    for line in description.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in DIMENSION_PREFIXES):
            continue
        kept.append(stripped)
    return kept


def format_decimal(value: float, decimals: int = 4) -> str:
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_filesize(num_bytes: int) -> str:
    size_mb = num_bytes / (1024 * 1024)
    formatted = f"{size_mb:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}MB"


def axes_length_map(axes: Optional[str], shape: Optional[Sequence[int]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if axes and shape:
        for axis, length in zip(str(axes), tuple(shape)):
            mapping[axis.upper()] = int(length)
    return mapping


def has_color_axis(page: object) -> bool:
    samples = getattr(page, "samplesperpixel", 1)
    return bool(samples and samples > 1)


def fallback_width_height(volume_shape: Sequence[int], color_axis: bool) -> Tuple[int, int]:
    width_idx = -2 if color_axis and len(volume_shape) >= 3 else -1
    height_idx = width_idx - 1
    width = int(volume_shape[width_idx])
    height = int(volume_shape[height_idx])
    return width, height


def fallback_depth(volume_shape: Sequence[int], color_axis: bool) -> Optional[int]:
    if len(volume_shape) < 3:
        return None
    width_idx = -2 if color_axis and len(volume_shape) >= 3 else -1
    height_idx = width_idx - 1
    depth_idx = height_idx - 1
    if abs(depth_idx) > len(volume_shape):
        return None
    return int(volume_shape[depth_idx])


def compute_dimension_pixels(
    volume_shape: Sequence[int],
    series_axes: Optional[str],
    series_shape: Optional[Sequence[int]],
    color_axis: bool,
    expect_depth: bool,
) -> DimensionPixels:
    axis_map = axes_length_map(series_axes, series_shape)

    width_px = axis_map.get("X")
    height_px = axis_map.get("Y")
    depth_px = axis_map.get("Z") if expect_depth else None

    if width_px is None or height_px is None:
        width_px, height_px = fallback_width_height(volume_shape, color_axis)

    if depth_px is None and expect_depth:
        depth_candidate = fallback_depth(volume_shape, color_axis)
        depth_px = depth_candidate

    return DimensionPixels(
        width=int(width_px),
        height=int(height_px),
        depth=int(depth_px) if depth_px is not None else None,
    )


def derive_axes(series_axes: Optional[str]) -> Optional[str]:
    if not series_axes:
        return None
    normalized = "".join(ch for ch in str(series_axes).upper() if ch in IMAGEJ_VALID_AXES)
    return normalized or None


def build_imagej_metadata(
    series_axes: Optional[str],
    config: StampConfig,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"unit": "micron"}
    axes = derive_axes(series_axes)
    if axes:
        metadata["axes"] = axes
    if config.pixel_size_z is not None:
        metadata["spacing"] = config.pixel_size_z
    return metadata


# def pixels_per_centimeter(pixel_size_um: float) -> float:
#     return MICROMETERS_PER_CENTIMETER / pixel_size_um


def build_dimension_lines(
    path: Path,
    volume_shape: Sequence[int],
    series_axes: Optional[str],
    series_shape: Optional[Sequence[int]],
    color_axis: bool,
    config: StampConfig,
) -> List[str]:
    expect_depth = config.pixel_size_z is not None
    pixels = compute_dimension_pixels(volume_shape, series_axes, series_shape, color_axis, expect_depth)

    width_um = pixels.width * config.pixel_size_x
    height_um = pixels.height * config.pixel_size_y
    depth_um = (
        pixels.depth * config.pixel_size_z
        if pixels.depth is not None and config.pixel_size_z is not None
        else None
    )

    size_str = format_filesize(path.stat().st_size)
    if math.isclose(config.pixel_size_x, config.pixel_size_y, rel_tol=1e-9, abs_tol=1e-12):
        resolution_line = f"Resolution:  {format_decimal(config.pixel_size_x)} microns per pixel"
    else:
        resolution_line = (
            "Resolution:  "
            f"{format_decimal(config.pixel_size_x)}x{format_decimal(config.pixel_size_y)} microns per pixel"
        )

    voxel_components = [config.pixel_size_x, config.pixel_size_y]
    if config.pixel_size_z is not None:
        voxel_components.append(config.pixel_size_z)
    voxel_str = "x".join(format_decimal(val) for val in voxel_components)
    unit_power = len(voxel_components)
    unit_label = "micron" if unit_power == 1 else f"micron^{unit_power}"

    lines = [
        f"Width:  {format_decimal(width_um)} microns ({pixels.width})",
        f"Height:  {format_decimal(height_um)} microns ({pixels.height})",
    ]

    if depth_um is not None and pixels.depth is not None:
        lines.append(f"Depth:  {format_decimal(depth_um)} microns ({pixels.depth})")

    lines.extend(
        [
            f"Size:  {size_str}",
            resolution_line,
            f"Voxel size: {voxel_str} {unit_label}",
        ]
    )

    return lines


def backup_path(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def gather_writer_kwargs(page: object) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    for attr in ("photometric", "planarconfig", "compression"):
        value = getattr(page, attr, None)
        if value is not None:
            kwargs[attr] = value
    extrasamples = getattr(page, "extrasamples", None)
    if extrasamples:
        kwargs["extrasamples"] = tuple(extrasamples)
    if getattr(page, "is_imagej", False):
        kwargs["imagej"] = True
    return kwargs


def process_file(path: Path, config: StampConfig) -> str:
    with tifffile.TiffFile(path) as tif:
        page0 = tif.pages[0]
        series = tif.series[0]
        series_axes = getattr(series, "axes", None)
        raw_series_shape = getattr(series, "shape", None)
        description = sanitize_description(page0.description)
        if config.skip_existing and has_dimension_block(description):
            return "skipped-dimensions"
        volume = series.asarray()
        volume_shape = volume.shape
        series_shape = tuple(raw_series_shape) if raw_series_shape is not None else tuple(volume_shape)
        color_axis = has_color_axis(page0)
        writer_kwargs = gather_writer_kwargs(page0)

    dimension_lines = build_dimension_lines(
        path,
        volume_shape,
        series_axes,
        series_shape,
        color_axis,
        config,
    )
    base_lines = strip_dimension_block(description)
    new_lines: List[str] = []
    if base_lines:
        new_lines.extend(base_lines)
        new_lines.append("")
    new_lines.extend(dimension_lines)
    new_description = "\n".join(new_lines)
    # resolution_xy = (
    #     pixels_per_centimeter(config.pixel_size_x),
    #     pixels_per_centimeter(config.pixel_size_y),
    # )
    resolution_xy = (1/config.pixel_size_x, 1/config.pixel_size_y)
    metadata = build_imagej_metadata(series_axes, config)
    if metadata:
        merged_meta = dict(writer_kwargs.get("metadata", {}))
        merged_meta.update(metadata)
        writer_kwargs["metadata"] = merged_meta
        writer_kwargs["imagej"] = True

    if config.dry_run:
        preview = "\n  ".join(dimension_lines)
        print(f"[dry-run] Would update {path} with:\n  {preview}")
        return "dry"

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    bigtiff = volume.nbytes >= (1 << 32)
    tifffile.imwrite(
        tmp_path,
        volume,
        description=new_description,
        resolution=resolution_xy,
        resolutionunit="CENTIMETER",
        bigtiff=bigtiff,
        **writer_kwargs,
    )

    if config.backup_suffix:
        dst = backup_path(path, config.backup_suffix)
        if dst.exists():
            if not config.overwrite_backup:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                raise FileExistsError(f"Backup {dst} already exists. Use --overwrite-backup to replace it.")
            dst.unlink()
        shutil.move(path, dst)
    else:
        path.unlink()

    tmp_path.replace(path)
    return "updated"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = validate_args(args, parser)
    files = collect_files(args.src, args.recursive, args.suffixes)
    if not files:
        print(f"No TIFF files found in {args.src} with suffixes {args.suffixes}", file=sys.stderr)
        sys.exit(1)

    stats = {"updated": 0, "skipped": 0, "dry": 0, "failed": 0}
    for path in files:
        try:
            outcome = process_file(path, config)
        except Exception as exc:  # pragma: no cover - CLI guard
            stats["failed"] += 1
            print(f"[error] Failed to update {path}: {exc}", file=sys.stderr)
            continue
        if outcome == "updated":
            stats["updated"] += 1
        elif outcome == "dry":
            stats["dry"] += 1
        else:
            stats["skipped"] += 1

    print(
        "Updated {updated} | skipped {skipped} | dry-run {dry} | failed {failed}".format(
            **stats
        )
    )
    if stats["failed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
