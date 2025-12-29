"""Convert DeepD3 raw data (TIFF stacks or DeepD3 ``.d3set`` files) into TransD3 Zarr.

This script converts DeepD3-style assets into a common Zarr layout used by TransD3.

**Supported inputs (top-level files in ``--src``)**

- ``.tif`` / ``.tiff``: Single volumetric stack. Output is one Zarr group:

    - ``<stem>.zarr/raw`` (float32)
    - ``<stem>.zarr/meta.json`` (JSON metadata)

- ``.d3set``: DeepD3 HDF5 dataset. Output is one Zarr group per sample in
    ``/data/stacks``:

    - ``<d3set-stem>_<sample>.zarr/raw`` (float32)
    - ``.../labels/shafts_semantic`` (uint8, may be zeros if missing)
    - ``.../labels/spines_instance`` (int32, may be zeros if missing)
    - ``.../meta.json`` (JSON metadata)

The output directory also contains an ``index.txt`` manifest with absolute paths
to each converted ``.zarr`` group.

**Voxel size inference (TIFF)**

For TIFF files, voxel size is inferred from (in priority order):

1) A line starting with ``Voxel size:`` in the TIFF description (if present)
2) TIFF resolution tags (X/Y resolution + resolution unit)
3) ImageJ metadata (``spacing`` + ``unit``) for Z spacing

The voxel size is stored as ``voxel_size_um`` in both Zarr attrs (on ``raw``)
and in ``meta.json`` as a (Z, Y, X) triple.

Examples
--------

Convert a folder of TIFF stacks:

        python -m transd3.scripts.convert_deepd3_to_zarr \
                --src /path/to/tiffs \
                --dst /path/to/output_zarr

Convert the MAE-processed dataset in this workspace:

        python /scratch/gpfs/WANG/callan/DeepD3/transd3/transd3/scripts/convert_deepd3_to_zarr.py \
                --src /scratch/gpfs/WANG/callan/mae_processed \
                --dst /scratch/gpfs/WANG/callan/DeepD3/datasets/mae_processed

Dependencies
------------

Required: ``tifffile``, ``zarr``, ``h5py``, and a Blosc codec via
``zarr[codecs]`` or ``numcodecs``.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MICRONS_PER_CENTIMETER = 1.0e4
MICRONS_PER_INCH = 2.54e4

try:  # pragma: no cover - external dependencies
    tifffile = importlib.import_module("tifffile")
    zarr = importlib.import_module("zarr")
    try:
        zarr_codecs = importlib.import_module("zarr.codecs")
    except ModuleNotFoundError:
        zarr_codecs = None
    try:
        numcodecs = importlib.import_module("numcodecs")
    except ModuleNotFoundError:
        numcodecs = None
    h5py = importlib.import_module("h5py")
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Please install tifffile, zarr, numcodecs, and h5py before running this conversion script."
    ) from exc

try:  # pragma: no cover - optional compression filters
    importlib.import_module("hdf5plugin")
except ModuleNotFoundError:  # pragma: no cover
    pass

ZARR_BLOSC = getattr(zarr_codecs, "Blosc", None) if zarr_codecs is not None else None
NUMCODECS_BLOSC = getattr(numcodecs, "Blosc", None) if numcodecs is not None else None
BITSHUFFLE = getattr(NUMCODECS_BLOSC, "BITSHUFFLE", 2)

from transd3.spine_data.utils_io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DeepD3 TIFF stacks or .d3set files into the TransD3 Zarr schema.",
        epilog=(
            "Notes:\n"
            "- This script scans only the top-level of --src for files ending in .tif/.tiff/.d3set.\n"
            "- Output Zarr groups are created directly under --dst and a manifest index.txt is written.\n"
            "\n"
            "Example:\n"
            "  python transd3/scripts/convert_deepd3_to_zarr.py --src /data --dst /out\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, required=True, help="Directory containing DeepD3 assets.")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory for Zarr groups.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def find_inputs(src: Path) -> List[Path]:
    allowed_suffixes = {".tif", ".tiff", ".d3set"}
    paths: List[Path] = []
    for path in src.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() in allowed_suffixes:
            paths.append(path)
    return sorted(paths)


def zarr_compressor() -> Any:
    if ZARR_BLOSC is not None:
        return ZARR_BLOSC(cname="zstd", clevel=5, shuffle=BITSHUFFLE)
    if NUMCODECS_BLOSC is not None:
        return NUMCODECS_BLOSC(cname="zstd", clevel=5, shuffle=BITSHUFFLE)
    raise RuntimeError("Blosc codec not available. Please install zarr[codecs] or numcodecs.")


def chunk_shape(shape: Sequence[int]) -> Sequence[int]:
    if len(shape) < 2:
        raise ValueError(f"Expected at least 2 spatial dimensions, got shape={shape}")
    y = int(shape[-2])
    x = int(shape[-1])
    if len(shape) == 2:
        z = 1
    else:
        z = int(np.prod(shape[:-2]))
    return (1, min(256, y), min(256, x))


def coerce_spatial_volume(volume: np.ndarray) -> np.ndarray:
    """Return a contiguous (Z, Y, X) view by flattening non-spatial axes."""

    if volume.ndim < 2:
        raise ValueError(f"Volume must have at least 2 dimensions, got shape={volume.shape}")
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    elif volume.ndim > 3:
        y, x = volume.shape[-2], volume.shape[-1]
        z = int(np.prod(volume.shape[:-2]))
        volume = volume.reshape(z, y, x)
    return np.ascontiguousarray(volume)


def _value_to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        pass
    try:
        num, den = value
        if float(den) == 0:
            return None
        return float(num) / float(den)
    except Exception:
        return None


def _unit_scale_microns(unit: Any) -> Optional[float]:
    if unit is None:
        return None
    if hasattr(unit, "value"):
        unit = unit.value
    if hasattr(unit, "name"):
        name = unit.name.lower()
        if "centimeter" in name:
            return MICRONS_PER_CENTIMETER
        if "inch" in name:
            return MICRONS_PER_INCH
    if isinstance(unit, str):
        low = unit.lower()
        if "centimeter" in low:
            return MICRONS_PER_CENTIMETER
        if "inch" in low:
            return MICRONS_PER_INCH
    if isinstance(unit, int):
        if unit == 3:  # TIFF RESUNIT.CENTIMETER
            return MICRONS_PER_CENTIMETER
        if unit == 2:  # TIFF RESUNIT.INCH
            return MICRONS_PER_INCH
    return None


def _resolution_tag_to_um(page: Any, axis: str) -> Optional[float]:
    tag_name = f"{axis.upper()}Resolution"
    tag = page.tags.get(tag_name) if hasattr(page, "tags") else None
    if tag is None:
        return None
    pixels_per_unit = _value_to_float(getattr(tag, "value", None))
    if pixels_per_unit in (None, 0):
        return None
    unit_tag = page.tags.get("ResolutionUnit") if hasattr(page, "tags") else None
    unit_value = getattr(unit_tag, "value", None) if unit_tag is not None else getattr(page, "resolutionunit", None)
    scale = _unit_scale_microns(unit_value)
    if scale is None:
        return None
    return scale / pixels_per_unit


def _imagej_spacing_um(tif: Any) -> Optional[float]:
    meta = getattr(tif, "imagej_metadata", None)
    if not isinstance(meta, dict):
        return None
    spacing = meta.get("spacing")
    try:
        spacing_val = float(spacing)
    except Exception:
        return None
    unit = str(meta.get("unit", "")).lower()
    if "micron" in unit or unit == "um":
        return spacing_val
    return None


def _parse_voxel_size_line(description: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not description:
        return None
    for line in description.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("voxel size"):
            continue
        payload = stripped.split(":", 1)
        if len(payload) != 2:
            continue
        tokens = payload[1].strip().split()
        if not tokens:
            continue
        numeric_block = tokens[0].replace(",", "x")
        parts = [p for p in numeric_block.split("x") if p]
        values: List[float] = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                continue
        if values:
            while len(values) < 3:
                values.append(values[-1])
            return float(values[0]), float(values[1]), float(values[2])
    return None


def infer_voxel_size_xyz(tif: Any) -> Tuple[float, float, float]:
    page0 = tif.pages[0]
    description = getattr(page0, "description", None)
    desc_values = _parse_voxel_size_line(description)
    x_um = _resolution_tag_to_um(page0, "X")
    y_um = _resolution_tag_to_um(page0, "Y")
    z_um = _imagej_spacing_um(tif)
    if desc_values is not None:
        dx, dy, dz = desc_values
        x_um = x_um or dx
        y_um = y_um or dy
        z_um = z_um or dz
    if x_um is None:
        x_um = 1.0
    if y_um is None:
        y_um = x_um
    if z_um is None:
        z_um = 1.0
    return float(x_um), float(y_um), float(z_um)


def convert_tiff(path: Path, dst_root: Path) -> Path:
    with tifffile.TiffFile(path) as tif:
        volume = tif.asarray().astype(np.float32)
        volume = coerce_spatial_volume(volume)
        voxel_xyz = infer_voxel_size_xyz(tif)

    dst = dst_root / f"{path.stem}.zarr"
    root = zarr.open_group(str(dst), mode="w")
    comp = zarr_compressor()
    raw = root.create_array(
        "raw",
        data=volume,
        chunks=chunk_shape(volume.shape),
        compressors=[comp],
    )

    voxel_meta = [voxel_xyz[2], voxel_xyz[1], voxel_xyz[0]]  # store as (Z, Y, X)
    raw.attrs["voxel_size_um"] = voxel_meta
    write_json(
        dst / "meta.json",
        {
            "source": str(path.resolve()),
            "voxel_size_um": voxel_meta,
            "voxel_size_source": "tiff-metadata",
        },
    )
    return dst


def decode_strings(values: Iterable[bytes]) -> List[str]:
    return [v.decode("utf-8") for v in values]


def build_meta_rows(h5: Any) -> Dict[str, Dict[str, float]]:
    meta: Dict[str, Dict[str, float]] = {}
    block1_items = decode_strings(h5["meta/block1_items"][()]) if "meta/block1_items" in h5 else []
    block1_values = h5["meta/block1_values"][()] if "meta/block1_values" in h5 else None
    block2_items = decode_strings(h5["meta/block2_items"][()]) if "meta/block2_items" in h5 else []
    block2_values = h5["meta/block2_values"][()] if "meta/block2_values" in h5 else None

    if block1_values is None and block2_values is None:
        return meta

    num_rows = block1_values.shape[0] if block1_values is not None else block2_values.shape[0]
    for idx in range(num_rows):
        entry: Dict[str, float] = {}
        if block1_values is not None:
            for col_idx, name in enumerate(block1_items):
                entry[name] = float(block1_values[idx, col_idx])
        if block2_values is not None:
            for col_idx, name in enumerate(block2_items):
                entry[name] = float(block2_values[idx, col_idx])
        meta[f"x{idx}"] = entry
    return meta


def convert_d3set(path: Path, dst_root: Path) -> List[Path]:
    outputs: List[Path] = []
    with h5py.File(path, "r") as h5:
        stack_group = h5["data/stacks"]
        dend_group = h5.get("data/dendrites")
        spine_group = h5.get("data/spines")
        meta_rows = build_meta_rows(h5)

        sample_names = sorted(stack_group.keys(), key=lambda name: int(name.lstrip("x")))
        comp = zarr_compressor()

        for sample_name in sample_names:
            stack = stack_group[sample_name][()].astype(np.float32, copy=False)
            shaft = (
                dend_group[sample_name][()].astype(np.uint8, copy=False)
                if dend_group and sample_name in dend_group
                else np.zeros_like(stack, dtype=np.uint8)
            )
            spine = (
                spine_group[sample_name][()].astype(np.int32, copy=False)
                if spine_group and sample_name in spine_group
                else np.zeros_like(stack, dtype=np.int32)
            )

            sample_id = f"{path.stem}_{sample_name}"
            dst = dst_root / f"{sample_id}.zarr"
            root = zarr.open_group(str(dst), mode="w")

            chunks = chunk_shape(stack.shape)
            root.create_array(
                "raw",
                data=stack,
                chunks=chunks,
                compressors=[comp],
            )
            root.create_array(
                "labels/shafts_semantic",
                data=shaft,
                chunks=chunks,
                compressors=[comp],
            )
            root.create_array(
                "labels/spines_instance",
                data=spine,
                chunks=chunks,
                compressors=[comp],
            )

            meta = {
                "source_file": str(path.resolve()),
                "source_key": sample_name,
                "voxel_size_um": [
                    float(meta_rows.get(sample_name, {}).get("Resolution_Z", 1.0)),
                    float(meta_rows.get(sample_name, {}).get("Resolution_XY", 1.0)),
                    float(meta_rows.get(sample_name, {}).get("Resolution_XY", 1.0)),
                ],
            }

            bounds = {
                key: int(value)
                for key, value in meta_rows.get(sample_name, {}).items()
                if key
                in {
                    "Depth",
                    "Height",
                    "Width",
                    "X",
                    "Y",
                    "Z_begin",
                    "Z_end",
                }
            }
            if bounds:
                meta["bounds"] = bounds

            write_json(dst / "meta.json", meta)
            outputs.append(dst)
    return outputs


def convert_input(path: Path, dst_root: Path) -> List[Path]:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return [convert_tiff(path, dst_root)]
    if suffix == ".d3set":
        return convert_d3set(path, dst_root)
    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    args = parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    inputs = find_inputs(args.src)
    if not inputs:
        raise FileNotFoundError(f"No supported DeepD3 inputs found in {args.src}")

    manifest_path = args.dst / "index.txt"
    converted: List[Path] = []
    with manifest_path.open("w") as f:
        for item in inputs:
            outputs = convert_input(item, args.dst)
            for out in outputs:
                converted.append(out)
                f.write(str(out.resolve()) + "\n")

    print(f"Converted {len(converted)} volumes from {len(inputs)} inputs -> {args.dst}")


if __name__ == "__main__":
    main()
