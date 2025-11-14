# TransD3

TransD3 is a reproducible training pipeline for spine and shaft segmentation on two-photon microscopy data. The project follows a staged curriculum consisting of self-supervised masked autoencoding, mixed denoising pretraining, and downstream segmentation fine-tuning. All stages operate on DeepD3-formatted datasets converted to a common Zarr schema.

## Key Features

- Deterministic data loading with 2.5D context tiles
- Masked autoencoder pretraining tailored for volumetric two-photon stacks
- Optional Noise2Void denoising for mixed self-supervision
- Mask2Former-inspired instance decoder for dendritic spines
- Comprehensive evaluation metrics including clDice and small-object AP

See `RUNCARD.md` for experiment tracking and reproduction details.
