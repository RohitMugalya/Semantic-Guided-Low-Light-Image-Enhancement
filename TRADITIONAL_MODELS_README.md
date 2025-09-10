# Traditional Low-Light Image Enhancement Models

This directory contains implementations of traditional (non-deep learning) methods for low-light image enhancement, as referenced in the paper "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video Enhancement" (2110.00970v4.pdf).

## Implemented Methods

### 1. Histogram Equalization (HE)
- **Description**: Classic method that redistributes pixel intensities to improve contrast
- **Implementation**: Applied to luminance channel in YUV color space
- **Pros**: Simple, fast, good for global contrast enhancement
- **Cons**: Can cause over-enhancement and loss of natural appearance

### 2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
- **Description**: Improved version of histogram equalization with local adaptation
- **Implementation**: Applied to L channel in LAB color space with clip limit
- **Pros**: Better local contrast, reduces over-enhancement
- **Cons**: Can introduce artifacts in smooth regions

### 3. Gamma Correction
- **Description**: Power-law transformation to adjust brightness
- **Implementation**: Fixed gamma (0.5) and adaptive gamma based on image statistics
- **Pros**: Simple, preserves color relationships
- **Cons**: Global operation, may not handle varying illumination well

### 4. Single Scale Retinex (SSR)
- **Description**: Based on Retinex theory for color constancy and lightness perception
- **Implementation**: log(I) - log(I*G) where G is Gaussian blur
- **Pros**: Good color preservation, handles illumination variations
- **Cons**: Can introduce halos around edges

### 5. Multi Scale Retinex (MSR)
- **Description**: Combines multiple SSR results with different scales
- **Implementation**: Average of SSR with sigmas [15, 80, 250]
- **Pros**: Better balance between local and global enhancement
- **Cons**: More computationally expensive than SSR

### 6. LIME Enhancement
- **Description**: Low-light Image Enhancement via Illumination Map Estimation
- **Implementation**: Estimates illumination map and enhances based on it
- **Pros**: Preserves natural appearance, good for non-uniform illumination
- **Cons**: May not enhance very dark regions sufficiently

### 7. Exposure Correction
- **Description**: Simple linear scaling of pixel intensities
- **Implementation**: Multiply by exposure factor (default 1.5)
- **Pros**: Very fast, simple
- **Cons**: Can cause saturation, doesn't handle noise well

### 8. Adaptive Gamma Correction
- **Description**: Gamma correction with parameter adapted to image statistics
- **Implementation**: Gamma = -log2(mean_intensity), clipped to [0.3, 2.5]
- **Pros**: Automatically adapts to image brightness
- **Cons**: Still a global operation

## Usage

### Method 1: Command Line Interface

```bash
# Run all methods on a dataset
python traditional_models_experiment.py --input_dir data/test_data/DICM --output_dir results

# Run specific methods only
python traditional_models_experiment.py --input_dir data/test_data/DICM --output_dir results --methods histogram_eq clahe gamma_correction

# Create comparison grid for single image
python traditional_models_experiment.py --comparison_image path/to/image.jpg --comparison_output comparison.png
```

### Method 2: Simple Python Script

```bash
# Run with default settings
python run_traditional_experiment.py
```

### Method 3: Programmatic Usage

```python
from traditional_models_experiment import TraditionalExperiment

# Initialize experiment
experiment = TraditionalExperiment("input_dir", "output_dir")

# Run experiment
results = experiment.run_experiment(methods=['histogram_eq', 'clahe', 'lime'])

# Create comparison for single image
experiment.create_comparison_grid("image.jpg", save_path="comparison.png")
```

## Requirements

Install the required packages:

```bash
pip install opencv-python numpy matplotlib scipy scikit-image pillow
```

## Output Structure

```
traditional_results/
├── histogram_eq/
│   ├── image1_histogram_eq.jpg
│   └── image2_histogram_eq.jpg
├── clahe/
│   ├── image1_clahe.jpg
│   └── image2_clahe.jpg
├── gamma_correction/
├── retinex_ssr/
├── retinex_msr/
├── lime/
├── exposure_correction/
└── adaptive_gamma/
```

## Performance Comparison

The script automatically measures and reports:
- Average processing time per image for each method
- Total number of images processed
- Success/failure rates

Example output:
```
EXPERIMENT SUMMARY
==================================================
histogram_eq        : 0.0234s avg, 50 images
clahe              : 0.0456s avg, 50 images
gamma_correction   : 0.0123s avg, 50 images
retinex_ssr        : 0.1234s avg, 50 images
lime               : 0.0789s avg, 50 images
```

## Comparison with Deep Learning Methods

These traditional methods serve as baselines for comparison with the main deep learning approach in this repository. Key differences:

**Traditional Methods:**
- ✅ Fast processing
- ✅ No training required
- ✅ Interpretable parameters
- ❌ Limited adaptability
- ❌ May not handle complex scenarios well

**Deep Learning (Main Model):**
- ✅ Better perceptual quality
- ✅ Handles complex illumination
- ✅ Semantic-aware enhancement
- ❌ Requires training
- ❌ Slower inference
- ❌ Less interpretable

## References

1. **Histogram Equalization**: Gonzalez, R.C. and Woods, R.E., 2017. Digital image processing.
2. **CLAHE**: Zuiderveld, K., 1994. Contrast limited adaptive histogram equalization.
3. **Retinex Theory**: Land, E.H. and McCann, J.J., 1971. Lightness and retinex theory.
4. **LIME**: Guo, X., Li, Y. and Ling, H., 2016. LIME: Low-light image enhancement via illumination map estimation.

## Notes

- All methods are implemented to handle both grayscale and color images
- Color images are processed in appropriate color spaces (YUV, LAB, RGB) depending on the method
- Error handling is included for robustness
- Results are saved in the same format as input images

## Troubleshooting

1. **"No images found"**: Check that your input directory contains supported image formats (.jpg, .jpeg, .png, .bmp, .tiff)
2. **Memory errors**: Reduce batch size or image resolution for large datasets
3. **Import errors**: Ensure all required packages are installed with correct versions
4. **Permission errors**: Check write permissions for output directory