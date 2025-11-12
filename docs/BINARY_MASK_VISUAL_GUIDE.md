# Binary Mask Feature - Visual Guide

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Data Creation                      │
└─────────────────────────────────────────────────────────────────┘

  1. Stack (3D)          2. Annotations (3D)      3. Mask (2D)
  ┌───────────┐          ┌───────────┐            ┌───────────┐
  │ Z=0       │          │ Z=0       │            │ 1 1 1 1 0 │
  ├───────────┤          ├───────────┤            │ 1 1 1 0 0 │
  │ Z=1       │    +     │ Z=1       │      +     │ 1 1 0 0 0 │
  ├───────────┤          ├───────────┤            │ 1 0 0 0 0 │
  │ Z=2       │          │ Z=2       │            │ 0 0 0 0 0 │
  └───────────┘          └───────────┘            └───────────┘
   Microscopy            Dendrites +                Mask:
   Image Stack           Spine Labels              1 = Ground-truth
                                                   0 = Pseudolabel

                              ↓

            ┌─────────────────────────────────────┐
            │       d3data File Created           │
            ├─────────────────────────────────────┤
            │ • stack: 3D array                   │
            │ • dendrite: 3D boolean              │
            │ • spines: 3D boolean                │
            │ • mask: 2D boolean (NEW!)           │
            └─────────────────────────────────────┘

```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Generator Flow                      │
└─────────────────────────────────────────────────────────────────┘

  d3set File                   DataGeneratorStream
  ┌──────────┐                 ┌────────────────────┐
  │ stacks   │────────────────▶│ Random sampling    │
  │ dendrites│────────────────▶│ Cropping           │
  │ spines   │────────────────▶│ Augmentation       │
  │ masks    │────────────────▶│ Normalization      │
  └──────────┘                 └────────────────────┘
       │                              │
       │                              ▼
       │                    ┌──────────────────────┐
       │                    │ Batch Output         │
       │                    ├──────────────────────┤
       │                    │ X:  Images           │
       │                    │ Y0: Dendrite labels  │
       │                    │ Y1: Spine labels     │
       │                    │ W:  Sample weights ★ │
       │                    └──────────────────────┘
       │                              │
       │                              ▼
       │                    ┌──────────────────────┐
       └───────────────────▶│ Training Loop        │
                            │ (with weighted loss) │
                            └──────────────────────┘
```

## Mask Application During Training

```
┌─────────────────────────────────────────────────────────────────┐
│              How Mask Affects Training Sample                   │
└─────────────────────────────────────────────────────────────────┘

   Input Image        Label           Mask          Weight Applied
   ┌─────────┐    ┌─────────┐    ┌─────────┐     ┌─────────┐
   │█████████│    │▓▓       │    │1 1 0 0 0│     │▓▓       │
   │█████████│ +  │▓▓▓      │ +  │1 1 1 0 0│  =  │▓▓▓ (50%)│
   │█████████│    │ ▓▓▓     │    │1 1 1 1 0│     │ ▓▓▓ (50%)│
   │█████████│    │  ▓▓     │    │1 1 1 1 1│     │  ▓▓     │
   └─────────┘    └─────────┘    └─────────┘     └─────────┘
   
   Where:
   • 1 = Ground-truth annotation → Full weight in loss
   • 0 = Pseudolabel annotation → Reduced weight in loss
   • Allows model to learn from both types of data
```

## GUI Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                 GUI: Create Training Data                       │
└─────────────────────────────────────────────────────────────────┘

   Step 1                Step 2                Step 3 (NEW!)
   ┌────────────┐       ┌────────────┐        ┌────────────┐
   │ Select     │       │ Select     │        │ Select     │
   │ Stack      │  →    │ Dendrite & │   →    │ Binary     │
   │ (*.tif)    │       │ Spines     │        │ Mask       │
   └────────────┘       └────────────┘        │ (*.tif)    │
                                               │ OPTIONAL   │
                                               └────────────┘
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              ↓
                    ┌───────────────────┐
                    │ Set resolution    │
                    │ & ROI             │
                    └───────────────────┘
                              ↓
                    ┌───────────────────┐
                    │ Save as d3data    │
                    └───────────────────┘
```

## Visualization in Viewer

```
┌─────────────────────────────────────────────────────────────────┐
│              Viewer: How Mask is Displayed                      │
└─────────────────────────────────────────────────────────────────┘

   Without Mask                      With Mask
   ┌─────────────────┐              ┌─────────────────┐
   │ ████████████    │              │ ████████████    │
   │ █ DENDRITE █    │              │ █ DENDRITE █    │
   │ ████████████    │              │ ████████████    │
   │      ▓▓▓        │              │      ▓▓▓   ░░░  │
   │     ▓▓▓▓        │              │     ▓▓▓▓  ░░░░  │
   │      ▓▓         │              │      ▓▓    ░░   │
   └─────────────────┘              └─────────────────┘
   
   █ = Dendrite (magenta)            █ = Ground-truth (full intensity)
   ▓ = Spines (green)                ▓ = Ground-truth (full intensity)
                                     ░ = Pseudolabel (50% intensity)
```

## File Format Examples

### d3data Structure
```python
{
    'data': {
        'stack': np.array(shape=(z, y, x), dtype=uint16),
        'dendrite': np.array(shape=(z, y, x), dtype=bool),
        'spines': np.array(shape=(z, y, x), dtype=bool),
        'mask': np.array(shape=(y, x), dtype=bool)  # NEW: 2D mask
    },
    'meta': {
        'Resolution_XY': 0.094,  # microns
        'Resolution_Z': 0.5,     # microns
        'Width': 512,
        'Height': 512,
        'Depth': 50,
        # ... other metadata
    }
}
```

### d3set Structure
```python
{
    'data': {
        'stacks': {
            'x0': np.array(...),
            'x1': np.array(...),
        },
        'dendrites': {
            'x0': np.array(...),
            'x1': np.array(...),
        },
        'spines': {
            'x0': np.array(...),
            'x1': np.array(...),
        },
        'masks': {              # NEW: Masks for each stack
            'x0': np.array(shape=(y, x), dtype=bool),
            'x1': np.array(shape=(y, x), dtype=bool),
        }
    },
    'meta': pd.DataFrame(...)
}
```

## Common Use Cases

### Use Case 1: High-Quality Manual Annotations Only
```
All pixels marked as ground-truth (mask = all 1s)
Perfect for high-quality manually annotated data
```

### Use Case 2: Mixed Quality Data
```
Region A: Manually annotated → mask = 1
Region B: Auto-generated    → mask = 0
Allows leveraging both data types in training
```

### Use Case 3: Progressive Refinement
```
Start: All pseudolabels      → mask = all 0s
Refine: Some verified        → mask = mixed
Final: All verified          → mask = all 1s
```

## Key Points

✅ **2D Mask**: Single slice applies to entire z-stack
✅ **Optional**: Not required for backward compatibility  
✅ **Binary**: 0 or 1 (False or True)
✅ **Format**: TIFF image file
✅ **Size**: Must match stack x,y dimensions
✅ **Default**: If not provided, all pixels treated as ground-truth

---

For detailed documentation, see: `docs/BINARY_MASK_SUPPORT.md`
