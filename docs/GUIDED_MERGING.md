# Guided Merging of Ground-Truth and Pseudolabel Annotations

## Overview

The DeepD3 training pipeline now supports **guided merging** of ground-truth and pseudolabel annotations using a binary mask. This feature allows users to combine manually annotated (ground-truth) data with automatically generated (pseudolabel) data in a single training dataset.

## Key Features

1. **Dual Annotation Loading**: Load both ground-truth and pseudolabel annotations
2. **Binary Mask-Guided Merging**: Use a 2D binary mask to control which annotations are used in each region
3. **Automatic Merging**: Annotations are automatically merged when all three components are provided
4. **Visual Feedback**: The GUI visualizes merged annotations with reduced intensity in pseudolabel regions
5. **Backward Compatible**: Existing workflows without pseudolabels continue to work unchanged

## How It Works

### Merging Logic

When you provide:
- Ground-truth dendrite and spine annotations
- Pseudolabel dendrite and spine annotations  
- A binary mask

The system merges them as follows:

```python
# Where mask == 1 (True): Use ground-truth annotations
# Where mask == 0 (False): Use pseudolabel annotations

merged_dendrite = np.where(mask, ground_truth_dendrite, pseudolabel_dendrite)
merged_spines = np.where(mask, ground_truth_spines, pseudolabel_spines)
```

### Binary Mask Format

- **Shape**: 2D image with same x,y dimensions as the stack
- **Values**: 
  - `1` (or non-zero): Ground-truth region
  - `0` (or zero): Pseudolabel region
- **File Format**: TIFF image
- **Application**: The 2D mask is automatically broadcast to all z-slices

## GUI Workflow

### Creating Training Data with Guided Merging

1. **Open the training GUI**: Run `deepd3-training` from the command line
2. **Click "Create training data"**
3. **Select your stack**: Click "Select stack" and choose your TIFF file
4. **Select ground-truth annotations**:
   - Click "Select dendrite tracings" (supports .swc or .tif)
   - Click "Select spine annotations" (supports .tif or .mask)
5. **Select binary mask**: Click "Select binary mask (2D)" and choose your mask file
6. **Select pseudolabel annotations** (NEW):
   - Click "Select pseudolabel dendrite" (supports .swc or .tif)
   - Click "Select pseudolabel spines" (supports .tif or .mask)
7. **Set resolution and ROI**: Configure as usual
8. **Save**: Click "Save annotation stack"
   - A message will confirm that merging was successful
   - The merged annotations are saved to the d3data file

### Visual Feedback

- **During editing**: The overlay shows the merged result
- **Pseudolabel regions**: Displayed with 50% intensity
- **Ground-truth regions**: Displayed at full intensity

## Use Cases

### 1. Semi-Supervised Training

Combine a small set of manually annotated ground-truth data with a larger set of pseudolabels:

```
Mask regions:
- Center region (high quality): Ground-truth (mask = 1)
- Peripheral regions: Pseudolabels (mask = 0)
```

### 2. Quality-Based Annotation

Use high-confidence predictions as pseudolabels in some regions:

```
Mask based on prediction confidence:
- High confidence regions: Pseudolabels (mask = 0)
- Low confidence or critical regions: Ground-truth (mask = 1)
```

### 3. Progressive Refinement

Start with pseudolabels everywhere, then manually refine specific regions:

```
Iterative workflow:
1. Generate pseudolabels for entire stack
2. Create mask identifying regions needing correction
3. Manually annotate those regions (ground-truth)
4. Merge using the mask
```

## Data Structure

### d3data File Structure (with merged annotations)

```python
{
    'data': {
        'stack': np.array,      # Shape: (z, y, x)
        'dendrite': np.array,   # Shape: (z, y, x) - MERGED
        'spines': np.array,     # Shape: (z, y, x) - MERGED
        'mask': np.array        # Shape: (y, x) - 2D mask
    },
    'meta': dict
}
```

**Important**: Only the **merged** annotations are saved to the d3data file, not the separate ground-truth and pseudolabel sets. The mask indicates which regions came from which source.

## Examples

### Example 1: Simple Top/Bottom Split

```python
# Ground-truth in top half, pseudolabels in bottom half
mask = np.zeros((height, width), dtype=bool)
mask[0:height//2, :] = True  # Top half = ground-truth
# Bottom half remains False (pseudolabel)
```

### Example 2: Center Region Ground-Truth

```python
# Ground-truth in center, pseudolabels on edges
mask = np.zeros((height, width), dtype=bool)
center_y, center_x = height // 2, width // 2
radius = min(height, width) // 4
y, x = np.ogrid[:height, :width]
mask[(y - center_y)**2 + (x - center_x)**2 <= radius**2] = True
```

### Example 3: Manual ROI Selection

Use ImageJ or another tool to create a binary mask marking specific regions:
1. Open your stack in ImageJ
2. Use ROI tools to select ground-truth regions
3. Create a binary mask (Edit → Selection → Create Mask)
4. Save as TIFF
5. Use in DeepD3 training GUI

## Backward Compatibility

### Without Pseudolabels

If you don't provide pseudolabel annotations, the workflow is unchanged:
- Load only ground-truth annotations
- Optionally provide a mask (for sample weighting during training)
- Save as d3data file

### Without Mask

If you provide both ground-truth and pseudolabels but no mask:
- Only ground-truth annotations are used
- Pseudolabels are ignored
- A warning is recommended (not currently implemented)

## Tips and Best Practices

1. **Mask Resolution**: Ensure the mask has the same x,y dimensions as your stack
2. **Consistent Annotations**: Both ground-truth and pseudolabel annotations should have the same shape
3. **Visual Inspection**: Use the overlay visualization to verify the merging is correct
4. **Quality Control**: Use the Viewer to inspect saved d3data files before training
5. **Mask Creation**: Use image analysis tools to create masks based on:
   - Prediction confidence maps
   - Manual ROI selection
   - Image quality metrics
   - Anatomical criteria

## Technical Details

### Merging Implementation

The merging happens in the `save()` method of `addStackWidget`:

```python
if (mask is not None and 
    dendrite_pseudo is not None and 
    spines_pseudo is not None):
    
    # Broadcast 2D mask to 3D
    mask_3d = np.broadcast_to(mask[np.newaxis, :, :], dendrite.shape)
    
    # Merge using numpy.where
    dendrite_merged = np.where(mask_3d, dendrite_gt, dendrite_pseudo)
    spines_merged = np.where(mask_3d, spines_gt, spines_pseudo)
```

### Performance Considerations

- Merging is performed in memory using NumPy operations (very fast)
- No additional disk space required (only merged result is saved)
- Negligible impact on save time

## Troubleshooting

### Issue: Mask shape mismatch

**Error**: Broadcast error during merging

**Solution**: Ensure mask has shape `(y, x)` matching the stack's spatial dimensions

### Issue: No merging performed

**Symptom**: Only ground-truth annotations saved

**Cause**: One or more components missing (mask, pseudolabels)

**Solution**: Verify all three components are loaded (check labels show filenames)

### Issue: Unexpected merged result

**Symptom**: Wrong regions showing ground-truth vs pseudolabel

**Cause**: Mask values inverted

**Solution**: Verify mask values (1 = ground-truth, 0 = pseudolabel)

## See Also

- [Binary Mask Support Documentation](BINARY_MASK_SUPPORT.md)
- [Binary Mask Visual Guide](BINARY_MASK_VISUAL_GUIDE.md)
- [DeepD3 Training Pipeline](source/userguide/train.rst)
