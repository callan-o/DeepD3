"""Convert DeepD3 raw data (TIF or .d3set) into the TransD3 Zarr schema."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Directory with DeepD3 assets")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory for Zarr groups")
    return parser.parse_args()


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
    z, y, x = shape
    return (1, min(256, y), min(256, x))


def convert_tiff(path: Path, dst_root: Path) -> Path:
    volume = tifffile.imread(path)
    volume = np.asarray(volume, dtype=np.float32)
    dst = dst_root / f"{path.stem}.zarr"
    root = zarr.open_group(str(dst), mode="w")
    comp = zarr_compressor()
    root.create_array(
        "raw",
        data=volume,
        chunks=chunk_shape(volume.shape),
        compressors=[comp],
    )
    write_json(
        dst / "meta.json",
        {
            "source": str(path.resolve()),
            "voxel_size_um": [1.0, 1.0, 1.0],
            "notes": "Default voxel size used; update if known.",
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
