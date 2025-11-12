# Binary Mask Support for Ground-Truth vs Pseudolabel Distinction

## Overview

The DeepD3 training pipeline now supports binary masks to distinguish between ground-truth annotations and pseudolabel annotations. This feature allows users to indicate which pixels in their training data are manually annotated (ground-truth) versus automatically generated (pseudolabels).

## Key Features

1. **Optional Binary Mask**: Users can optionally provide a 2D binary mask when creating training data
2. **2D Mask Applied to All Z-Slices**: The mask is a single 2D image that applies to all slices in the accompanying z-stack
3. **Backward Compatible**: Existing training data without masks will work seamlessly (defaults to all ground-truth)
4. **Sample Weighting**: During training, the mask is returned as sample weights for use in weighted loss functions

## GUI Changes

### Creating Training Data (`deepd3-training`)

When creating a new d3data training file:

1. **New Button**: "Select binary mask (2D)" - appears after selecting the spine annotations
2. **Optional**: This step is optional; if no mask is provided, all pixels are treated as ground-truth
3. **File Format**: Binary mask should be a 2D TIFF image with the same x,y dimensions as the stack
4. **Values**: 
   - `True` (or non-zero): Ground-truth annotation
   - `False` (or zero): Pseudolabel annotation

### Viewing Training Data

The Viewer now displays pseudolabel regions with reduced intensity (50% opacity) compared to ground-truth regions.

## Data Structure Changes

### d3data Files

Old structure:
```python
{
    'data': {
        'stack': np.array,      # Shape: (z, y, x)
        'dendrite': np.array,   # Shape: (z, y, x)
        'spines': np.array      # Shape: (z, y, x)
    },
    'meta': dict
}
```

New structure (with optional mask):
```python
{
    'data': {
        'stack': np.array,      # Shape: (z, y, x)
        'dendrite': np.array,   # Shape: (z, y, x)
        'spines': np.array,     # Shape: (z, y, x)
        'mask': np.array        # Shape: (y, x) - OPTIONAL, 2D only
    },
    'meta': dict
}
```

### d3set Files

Old structure:
```python
{
    'data': {
        'stacks': {'x0': ..., 'x1': ...},
        'dendrites': {'x0': ..., 'x1': ...},
        'spines': {'x0': ..., 'x1': ...}
    },
    'meta': pd.DataFrame
}
```

New structure:
```python
{
    'data': {
        'stacks': {'x0': ..., 'x1': ...},
        'dendrites': {'x0': ..., 'x1': ...},
        'spines': {'x0': ..., 'x1': ...},
        'masks': {'x0': ..., 'x1': ...}  # NEW: 2D masks for each stack
    },
    'meta': pd.DataFrame
}
```

**Note**: If a d3data file doesn't have a mask, the Arrange function automatically creates a mask of all ones (all ground-truth) for that dataset.

## Training Generator Changes

### DataGeneratorStream Return Format

Old format:
```python
X, (Y0, Y1) = generator[i]
```

New format:
```python
X, (Y0, Y1), W = generator[i]
```

Where:
- `X`: Input images, shape `(batch_size, height, width, 1)`
- `Y0`: Dendrite labels, shape `(batch_size, height, width, 1)`
- `Y1`: Spine labels, shape `(batch_size, height, width, 1)`
- `W`: Sample weights from mask, shape `(batch_size, height, width, 1)` - **NEW**

## Usage Example

### Creating Training Data with Mask

1. Open `deepd3-training`
2. Click "Create training data"
3. Select your stack (TIFF file)
4. Select dendrite tracings (SWC or TIFF)
5. Select spine annotations (TIFF or MASK)
6. **NEW**: Optionally click "Select binary mask (2D)" and choose your mask file
7. Set resolution and ROI
8. Save as d3data file

### Training with Masks

When training your model, the mask is automatically used as sample weights:

```python
from deepd3.training.stream import DataGeneratorStream

# Create generator
generator = DataGeneratorStream('training.d3set', batch_size=8)

# Use in training
for X, (Y0, Y1), W in generator:
    # W contains the mask weights
    # Use W in your loss function for weighted training
    loss = weighted_loss(predictions, (Y0, Y1), sample_weight=W)
```

## Backward Compatibility

All existing code and training data will continue to work:

1. **Old d3data files** without masks will automatically get a default mask of all ones (all ground-truth)
2. **Old training scripts** that don't use sample weights will still work (though the mask will be generated and returned)
3. **The GUI** gracefully handles files with or without masks

## Technical Details

- **Mask Storage**: Masks are stored as boolean arrays (dtype: bool)
- **Mask Application**: The 2D mask is broadcast to all z-slices during training
- **Augmentation**: Masks are augmented along with the images and labels (rotations, flips, etc.)
- **Cropping**: Masks are cropped along with the stack when ROI cropping is enabled

## Benefits

1. **Flexible Training**: Mix ground-truth and pseudolabel data in the same training set
2. **Weighted Learning**: Train models that prioritize ground-truth annotations
3. **Data Efficiency**: Use both manually annotated and automatically generated data
4. **Quality Control**: Clearly distinguish between different data quality levels
