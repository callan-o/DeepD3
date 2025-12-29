# TransD3

TransD3 is a reproducible training pipeline for spine and shaft segmentation on two-photon microscopy data. The project follows a staged curriculum consisting of self-supervised masked autoencoding, mixed denoising pretraining, and downstream segmentation fine-tuning. All stages operate on DeepD3-formatted datasets converted to a common Zarr schema.

## Dataset conversion (DeepD3 -> Zarr)

TransD3 expects volumes in a simple Zarr layout (one Zarr group per volume) with
the raw stack stored at ``raw`` and optional label arrays under ``labels/``.

Use the converter script:

	python transd3/transd3/scripts/convert_deepd3_to_zarr.py \
		--src /path/to/deepd3_assets \
		--dst /path/to/output_zarr

In this workspace, to convert the MAE-processed TIFF stacks:

	python /scratch/gpfs/WANG/callan/DeepD3/transd3/transd3/scripts/convert_deepd3_to_zarr.py \
		--src /scratch/gpfs/WANG/callan/mae_processed \
		--dst /scratch/gpfs/WANG/callan/DeepD3/datasets/mae_processed

The destination folder will contain ``.zarr`` directories plus an ``index.txt``
manifest listing all converted outputs.
# TransD3

TransD3 is a reproducible training pipeline for spine and shaft segmentation on two-photon microscopy data. The project follows a staged curriculum consisting of self-supervised masked autoencoding, mixed denoising pretraining, and downstream segmentation fine-tuning. All stages operate on DeepD3-formatted datasets converted to a common Zarr schema.

## Key Features


## Key Features

- Deterministic data loading with 2.5D context tiles
- Masked autoencoder pretraining tailored for volumetric two-photon stacks
- Optional Noise2Void denoising for mixed self-supervision
- Mask2Former-inspired instance decoder for dendritic spines
- Comprehensive evaluation metrics including clDice and small-object AP

See `RUNCARD.md` for experiment tracking and reproduction details.

See `RUNCARD.md` for experiment tracking and reproduction details.
