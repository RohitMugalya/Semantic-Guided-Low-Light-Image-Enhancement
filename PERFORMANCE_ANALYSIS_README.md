# Performance Analysis for Semantic-Guided Low-Light Image Enhancement

This directory contains comprehensive performance analysis tools for the Semantic-Guided Low-Light Image Enhancement model. The analysis provides detailed metrics, visualizations, and reports to evaluate model performance across various dimensions.

## 📁 Files Overview

- `performance_analysis.py` - Main performance analysis script
- `performance_usage_example.py` - Usage examples and demonstrations
- `PERFORMANCE_ANALYSIS_README.md` - This documentation file

## 🚀 Quick Start

### Prerequisites

Ensure you have the following installed:
```bash
pip install torch torchvision opencv-python scikit-image matplotlib seaborn psutil GPUtil
```

### Basic Usage

1. **Quick Analysis (5 images):**
```bash
python performance_usage_example.py
```

2. **Full Analysis (all test images):**
```bash
python performance_analysis.py --test_data_path data/test_data/ --output_dir results/
```

3. **Custom Analysis:**
```bash
python performance_analysis.py --max_images 20 --test_data_path data/test_data/LOL/ --output_dir lol_analysis/
```

## 📊 Analysis Features

### 1. **Image Quality Metrics**
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **SSIM (Structural Similarity Index)**: Evaluates structural preservation
- **MSE (Mean Squared Error)**: Pixel-wise difference measurement
- **MAE (Mean Absolute Error)**: Average absolute pixel differences
- **RMSE (Root Mean Square Error)**: Root of mean squared differences
- **NCC (Normalized Cross-Correlation)**: Correlation between images

### 2. **Enhancement-Specific Metrics**
- **Brightness Improvement**: Average brightness enhancement
- **Contrast Improvement**: Standard deviation enhancement
- **Dynamic Range Improvement**: Range of pixel values enhancement
- **Histogram Correlation**: Similarity of intensity distributions
- **Edge Preservation**: Structural detail preservation

### 3. **Performance Metrics**
- **Inference Time**: Model processing speed
- **Memory Usage**: RAM consumption during inference
- **GPU Memory**: VRAM usage (if CUDA available)
- **Throughput**: Images processed per second

### 4. **Loss Function Analysis**
- **Color Loss**: Color consistency evaluation
- **Spatial Loss**: Spatial structure preservation
- **Exposure Loss**: Brightness level optimization
- **Total Variation Loss**: Smoothness regularization
- **Segmentation Loss**: Semantic preservation

## 📈 Generated Reports

### 1. **Text Reports**
- `detailed_report.txt` - Comprehensive analysis with all metrics
- `summary_report.txt` - Executive summary with key findings

### 2. **Data Files**
- `results.json` - Machine-readable results in JSON format
- `baseline_comparison.json` - Comparison with baseline methods

### 3. **Visualizations**
- `performance_visualizations.png` - Statistical distribution plots
- `correlation_matrix.png` - Metrics correlation heatmap

### 4. **Publication-Ready Outputs**
- `results_table.tex` - LaTeX table for academic papers

## 🔧 Command Line Options

```bash
python performance_analysis.py [OPTIONS]

Options:
  --weight_path PATH        Path to model weights (default: weight/Epoch99.pth)
  --test_data_path PATH     Path to test data directory (default: data/test_data/)
  --output_dir PATH         Output directory for results (default: performance_results/)
  --max_images N            Maximum number of images to analyze (default: all)
  --device DEVICE           Device to use: cuda/cpu (default: auto-detect)
  --help                    Show help message
```

## 📋 Example Output

### Summary Report
```
EXECUTIVE SUMMARY - PERFORMANCE ANALYSIS
============================================================

Analysis completed on 150 images
Model: Semantic-Guided Low-Light Image Enhancement
Date: 2024-01-15 14:30:25

KEY PERFORMANCE INDICATORS
------------------------------
• Average Inference Time: 0.2341 seconds
• Throughput: 4.27 images/second
• Average PSNR: 28.45 dB
• Average SSIM: 0.892
• Average Brightness Improvement: 0.156
• Average Contrast Improvement: 0.089

MODEL EFFICIENCY
------------------------------
• Total Parameters: 23,456,789
• Model Size: 89.4 MB
• Average Memory Usage: 1,234.5 MB
```

### Detailed Metrics
```
IMAGE QUALITY METRICS
------------------------------
PSNR:
  Mean: 28.4523 ± 3.2145
  Median: 28.1234
  Min: 22.1234
  Max: 35.6789

SSIM:
  Mean: 0.8921 ± 0.0456
  Median: 0.8956
  Min: 0.7890
  Max: 0.9456
```

## 🎯 Model Architecture Analysis

The performance analyzer evaluates the following model components:

### Enhancement Network (`enhance_net_nopool`)
- **Type**: Zero-DCE inspired enhancement network
- **Convolution**: Depthwise Separable Convolution (DSC)
- **Scale Factor**: 12 (for multi-scale processing)
- **Parameters**: ~23M parameters

### Segmentation Network (`fpn`)
- **Type**: Feature Pyramid Network
- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Classes**: 21 (VOC dataset classes)
- **Purpose**: Semantic guidance for enhancement

### Loss Functions
- **Color Loss**: Maintains color consistency
- **Spatial Loss**: Preserves spatial structure (8-directional)
- **Exposure Loss**: Controls brightness levels
- **TV Loss**: Ensures smooth parameter maps
- **Segmentation Loss**: Preserves semantic information

## 🔍 Advanced Usage

### Custom Metrics
You can extend the analyzer by adding custom metrics:

```python
def custom_metric(self, original, enhanced):
    # Your custom metric calculation
    return metric_value

# Add to PerformanceAnalyzer class
analyzer.custom_metric = custom_metric
```

### Batch Processing
For large datasets, use batch processing:

```python
analyzer = PerformanceAnalyzer()
results = analyzer.run_comprehensive_analysis(max_images=1000)
```

### Memory Optimization
For memory-constrained environments:

```python
analyzer = PerformanceAnalyzer()
# Process in smaller batches
for batch in batches:
    batch_results = analyzer.analyze_batch(batch)
```

## 📊 Interpretation Guidelines

### PSNR Interpretation
- **> 30 dB**: Excellent quality
- **25-30 dB**: Good quality
- **20-25 dB**: Fair quality
- **< 20 dB**: Poor quality

### SSIM Interpretation
- **> 0.9**: Excellent structural similarity
- **0.8-0.9**: Good structural similarity
- **0.7-0.8**: Fair structural similarity
- **< 0.7**: Poor structural similarity

### Inference Time Guidelines
- **< 0.1s**: Real-time capable
- **0.1-0.5s**: Near real-time
- **0.5-1.0s**: Interactive
- **> 1.0s**: Batch processing recommended

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python performance_analysis.py --device cpu
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Weights Not Found**
   - Ensure `weight/Epoch99.pth` exists
   - Check file permissions

4. **Test Data Not Found**
   - Verify `data/test_data/` directory exists
   - Check image file formats (jpg, png, bmp)

### Performance Tips

1. **GPU Acceleration**: Use CUDA for faster inference
2. **Batch Processing**: Process multiple images together
3. **Memory Management**: Use `max_images` parameter for large datasets
4. **Parallel Processing**: Use multiple workers for data loading

## 📚 References

- Original Paper: "Semantic-guided zero-shot learning for low-light image/video enhancement"
- Conference: WACV 2022
- GitHub: [Semantic-Guided-Low-Light-Image-Enhancement](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement)

## 🤝 Contributing

To contribute to the performance analysis tools:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This performance analysis tool follows the same license as the main project.

---

**Note**: This performance analysis tool is designed to work with the Semantic-Guided Low-Light Image Enhancement model. Ensure you have the correct model weights and test data before running the analysis.
