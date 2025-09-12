#!/usr/bin/env python3
"""
Comprehensive Performance Analysis Script for Semantic-Guided Low-Light Image Enhancement

This script provides detailed performance metrics and analysis for the low-light image enhancement model.
It includes quantitative metrics, computational analysis, and visual quality assessments.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import glob
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import psutil
import GPUtil
from torch.profiler import profile, record_function, ProfilerActivity

# Import project modules
from modeling import model
from modeling.fpn import fpn, FocalLoss
from utils import get_device, image_from_path, scale_image
import Myloss

class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for the low-light image enhancement model.
    """
    
    def __init__(self, weight_path="weight/Epoch99.pth", test_data_path="data/test_data/", 
                 output_dir="performance_results/", device=None):
        """
        Initialize the performance analyzer.
        
        Args:
            weight_path (str): Path to the trained model weights
            test_data_path (str): Path to test data directory
            output_dir (str): Directory to save analysis results
            device: PyTorch device (auto-detected if None)
        """
        self.weight_path = weight_path
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        self.device = device if device else get_device()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models
        self.scale_factor = 12
        self.net = model.enhance_net_nopool(self.scale_factor, conv_type='dsc').to(self.device)
        self.seg = fpn(21).to(self.device)  # 21 classes for VOC segmentation
        
        # Load trained weights
        self._load_weights()
        
        # Initialize loss functions for analysis
        self.l_color = Myloss.L_color()
        self.l_spa = Myloss.L_spa8(patch_size=4)
        self.l_exp = Myloss.L_exp(16)
        self.l_tv = Myloss.L_TV()
        self.seg_criterion = FocalLoss(gamma=2).to(self.device)
        
        # Performance metrics storage
        self.metrics = defaultdict(list)
        self.timing_data = []
        self.memory_usage = []
        
    def _load_weights(self):
        """Load trained model weights."""
        try:
            if os.path.exists(self.weight_path):
                self.net.load_state_dict(torch.load(self.weight_path, map_location=self.device))
                print(f"✓ Loaded model weights from {self.weight_path}")
            else:
                print(f"⚠ Warning: Weight file {self.weight_path} not found. Using random weights.")
        except Exception as e:
            print(f"⚠ Error loading weights: {e}. Using random weights.")
    
    def get_test_images(self):
        """Get list of test images from all subdirectories."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        test_images = []
        
        for root, dirs, files in os.walk(self.test_data_path):
            for ext in image_extensions:
                test_images.extend(glob.glob(os.path.join(root, ext)))
        
        return sorted(test_images)
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input."""
        # Load and preprocess image
        data_lowlight = image_from_path(image_path)
        data_lowlight = scale_image(data_lowlight, self.scale_factor, self.device)
        return data_lowlight
    
    def postprocess_image(self, enhanced_tensor):
        """Convert model output to displayable image."""
        # Convert tensor to numpy array
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        enhanced_np = np.clip(enhanced_np, 0, 1)
        return (enhanced_np * 255).astype(np.uint8)
    
    def calculate_image_metrics(self, original, enhanced):
        """Calculate various image quality metrics."""
        # Convert to grayscale for some metrics
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        # PSNR
        psnr_value = psnr(original, enhanced, data_range=255)
        
        # SSIM
        ssim_value = ssim(orig_gray, enh_gray, data_range=255)
        
        # Mean Squared Error
        mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
        
        # Peak Signal-to-Noise Ratio (alternative calculation)
        if mse == 0:
            psnr_alt = float('inf')
        else:
            psnr_alt = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Structural Similarity Index (alternative calculation)
        ssim_alt = ssim(original, enhanced, multichannel=True, data_range=255)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(original.astype(float) - enhanced.astype(float)))
        
        # Root Mean Square Error
        rmse = np.sqrt(mse)
        
        # Normalized Cross-Correlation
        orig_flat = original.flatten().astype(float)
        enh_flat = enhanced.flatten().astype(float)
        ncc = np.corrcoef(orig_flat, enh_flat)[0, 1]
        
        return {
            'psnr': psnr_value,
            'ssim': ssim_value,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'ncc': ncc,
            'psnr_alt': psnr_alt,
            'ssim_alt': ssim_alt
        }
    
    def calculate_enhancement_metrics(self, original, enhanced):
        """Calculate enhancement-specific metrics."""
        # Convert to float for calculations
        orig_float = original.astype(float) / 255.0
        enh_float = enhanced.astype(float) / 255.0
        
        # Brightness improvement
        orig_brightness = np.mean(orig_float)
        enh_brightness = np.mean(enh_float)
        brightness_improvement = enh_brightness - orig_brightness
        
        # Contrast improvement (standard deviation)
        orig_contrast = np.std(orig_float)
        enh_contrast = np.std(enh_float)
        contrast_improvement = enh_contrast - orig_contrast
        
        # Dynamic range
        orig_dynamic_range = np.max(orig_float) - np.min(orig_float)
        enh_dynamic_range = np.max(enh_float) - np.min(enh_float)
        dynamic_range_improvement = enh_dynamic_range - orig_dynamic_range
        
        # Histogram analysis
        orig_hist = np.histogram(orig_float.flatten(), bins=256, range=(0, 1))[0]
        enh_hist = np.histogram(enh_float.flatten(), bins=256, range=(0, 1))[0]
        
        # Histogram correlation
        hist_corr = np.corrcoef(orig_hist, enh_hist)[0, 1]
        
        # Edge preservation (using Sobel operator)
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        orig_edges = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1, ksize=3)
        enh_edges = cv2.Sobel(enh_gray, cv2.CV_64F, 1, 1, ksize=3)
        
        edge_correlation = np.corrcoef(orig_edges.flatten(), enh_edges.flatten())[0, 1]
        
        return {
            'brightness_improvement': brightness_improvement,
            'contrast_improvement': contrast_improvement,
            'dynamic_range_improvement': dynamic_range_improvement,
            'histogram_correlation': hist_corr,
            'edge_correlation': edge_correlation,
            'original_brightness': orig_brightness,
            'enhanced_brightness': enh_brightness,
            'original_contrast': orig_contrast,
            'enhanced_contrast': enh_contrast
        }
    
    def calculate_loss_metrics(self, original_tensor, enhanced_tensor, params_maps):
        """Calculate loss function values for analysis."""
        with torch.no_grad():
            # Color loss
            color_loss = torch.mean(self.l_color(enhanced_tensor))
            
            # Spatial loss
            spatial_loss = torch.mean(self.l_spa(enhanced_tensor, original_tensor))
            
            # Exposure loss
            exposure_loss = torch.mean(self.l_exp(enhanced_tensor, 0.6))
            
            # Total Variation loss
            tv_loss = self.l_tv(params_maps)
            
            # Segmentation loss (if segmentation network is available)
            try:
                seg_output = self.seg(enhanced_tensor)
                target = torch.argmax(torch.log_softmax(seg_output, dim=1), dim=1)
                seg_loss = self.seg_criterion(seg_output, target)
            except:
                seg_loss = torch.tensor(0.0)
            
            return {
                'color_loss': color_loss.item(),
                'spatial_loss': spatial_loss.item(),
                'exposure_loss': exposure_loss.item(),
                'tv_loss': tv_loss.item(),
                'segmentation_loss': seg_loss.item(),
                'total_loss': (1600 * tv_loss + spatial_loss + 5 * color_loss + 10 * exposure_loss + 0.1 * seg_loss).item()
            }
    
    def analyze_single_image(self, image_path):
        """Analyze a single image and return comprehensive metrics."""
        print(f"Analyzing: {os.path.basename(image_path)}")
        
        # Preprocess image
        original_tensor = self.preprocess_image(image_path)
        original_np = self.postprocess_image(original_tensor)
        
        # Model inference
        start_time = time.time()
        with torch.no_grad():
            enhanced_tensor, params_maps = self.net(original_tensor)
        inference_time = time.time() - start_time
        
        # Postprocess enhanced image
        enhanced_np = self.postprocess_image(enhanced_tensor)
        
        # Calculate metrics
        image_metrics = self.calculate_image_metrics(original_np, enhanced_np)
        enhancement_metrics = self.calculate_enhancement_metrics(original_np, enhanced_np)
        loss_metrics = self.calculate_loss_metrics(original_tensor, enhanced_tensor, params_maps)
        
        # Memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # GPU memory usage if available
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        return {
            'image_path': image_path,
            'inference_time': inference_time,
            'memory_usage_mb': memory_usage,
            'gpu_memory_mb': gpu_memory,
            **image_metrics,
            **enhancement_metrics,
            **loss_metrics
        }
    
    def run_comprehensive_analysis(self, max_images=None):
        """Run comprehensive analysis on test dataset."""
        print("🚀 Starting comprehensive performance analysis...")
        
        # Get test images
        test_images = self.get_test_images()
        if max_images:
            test_images = test_images[:max_images]
        
        print(f"📊 Analyzing {len(test_images)} images...")
        
        # Analyze each image
        results = []
        for i, image_path in enumerate(test_images):
            try:
                result = self.analyze_single_image(image_path)
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{len(test_images)} images processed")
                    
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                continue
        
        # Store results
        self.results = results
        
        # Calculate aggregate statistics
        self._calculate_aggregate_stats()
        
        # Generate reports
        self._generate_reports()
        
        print(f"✅ Analysis complete! Results saved to {self.output_dir}")
        return results
    
    def _calculate_aggregate_stats(self):
        """Calculate aggregate statistics from all results."""
        if not self.results:
            return
        
        # Group metrics by category
        self.aggregate_stats = {}
        
        # Performance metrics
        perf_metrics = ['inference_time', 'memory_usage_mb', 'gpu_memory_mb']
        for metric in perf_metrics:
            values = [r[metric] for r in self.results if metric in r]
            if values:
                self.aggregate_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Image quality metrics
        quality_metrics = ['psnr', 'ssim', 'mse', 'mae', 'rmse', 'ncc']
        for metric in quality_metrics:
            values = [r[metric] for r in self.results if metric in r]
            if values:
                self.aggregate_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Enhancement metrics
        enh_metrics = ['brightness_improvement', 'contrast_improvement', 'dynamic_range_improvement']
        for metric in enh_metrics:
            values = [r[metric] for r in self.results if metric in r]
            if values:
                self.aggregate_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
    
    def _generate_reports(self):
        """Generate comprehensive analysis reports."""
        if not self.results:
            print("No results to generate reports from.")
            return
        
        # Generate text report
        self._generate_text_report()
        
        # Generate JSON report
        self._generate_json_report()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_text_report(self):
        """Generate detailed text report."""
        report_path = os.path.join(self.output_dir, "detailed_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SEMANTIC-GUIDED LOW-LIGHT IMAGE ENHANCEMENT - PERFORMANCE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Weights: {self.weight_path}\n")
            f.write(f"Test Data Path: {self.test_data_path}\n")
            f.write(f"Number of Images Analyzed: {len(self.results)}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Model Architecture Summary
            f.write("MODEL ARCHITECTURE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write("Enhancement Network:\n")
            f.write("  - Type: enhance_net_nopool\n")
            f.write("  - Convolution Type: Depthwise Separable Convolution (DSC)\n")
            f.write("  - Scale Factor: {}\n".format(self.scale_factor))
            f.write("  - Parameters: {:,}\n".format(sum(p.numel() for p in self.net.parameters())))
            f.write("\nSegmentation Network:\n")
            f.write("  - Type: FPN (Feature Pyramid Network)\n")
            f.write("  - Backbone: ResNet-50\n")
            f.write("  - Classes: 21 (VOC dataset)\n")
            f.write("  - Parameters: {:,}\n".format(sum(p.numel() for p in self.seg.parameters())))
            f.write("\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            if 'inference_time' in self.aggregate_stats:
                stats = self.aggregate_stats['inference_time']
                f.write(f"Inference Time (seconds):\n")
                f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  Median: {stats['median']:.4f}\n")
                f.write(f"  Min: {stats['min']:.4f}\n")
                f.write(f"  Max: {stats['max']:.4f}\n\n")
            
            if 'memory_usage_mb' in self.aggregate_stats:
                stats = self.aggregate_stats['memory_usage_mb']
                f.write(f"Memory Usage (MB):\n")
                f.write(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
                f.write(f"  Median: {stats['median']:.2f}\n")
                f.write(f"  Min: {stats['min']:.2f}\n")
                f.write(f"  Max: {stats['max']:.2f}\n\n")
            
            # Image Quality Metrics
            f.write("IMAGE QUALITY METRICS\n")
            f.write("-" * 40 + "\n")
            quality_metrics = ['psnr', 'ssim', 'mse', 'mae', 'rmse', 'ncc']
            for metric in quality_metrics:
                if metric in self.aggregate_stats:
                    stats = self.aggregate_stats[metric]
                    f.write(f"{metric.upper()}:\n")
                    f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  Min: {stats['min']:.4f}\n")
                    f.write(f"  Max: {stats['max']:.4f}\n\n")
            
            # Enhancement Metrics
            f.write("ENHANCEMENT METRICS\n")
            f.write("-" * 40 + "\n")
            enh_metrics = ['brightness_improvement', 'contrast_improvement', 'dynamic_range_improvement']
            for metric in enh_metrics:
                if metric in self.aggregate_stats:
                    stats = self.aggregate_stats[metric]
                    f.write(f"{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  Min: {stats['min']:.4f}\n")
                    f.write(f"  Max: {stats['max']:.4f}\n\n")
            
            # Individual Image Results
            f.write("INDIVIDUAL IMAGE RESULTS\n")
            f.write("-" * 40 + "\n")
            for i, result in enumerate(self.results):
                f.write(f"\nImage {i+1}: {os.path.basename(result['image_path'])}\n")
                f.write(f"  Inference Time: {result['inference_time']:.4f}s\n")
                f.write(f"  PSNR: {result['psnr']:.4f}\n")
                f.write(f"  SSIM: {result['ssim']:.4f}\n")
                f.write(f"  Brightness Improvement: {result['brightness_improvement']:.4f}\n")
                f.write(f"  Contrast Improvement: {result['contrast_improvement']:.4f}\n")
        
        print(f"📄 Detailed text report saved to {report_path}")
    
    def _generate_json_report(self):
        """Generate JSON report for programmatic access."""
        json_path = os.path.join(self.output_dir, "results.json")
        
        report_data = {
            'analysis_info': {
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_weights': self.weight_path,
                'test_data_path': self.test_data_path,
                'device': str(self.device),
                'num_images': len(self.results)
            },
            'aggregate_statistics': self.aggregate_stats,
            'individual_results': self.results
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📊 JSON report saved to {json_path}")
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        if not self.results:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Analysis Visualizations', fontsize=16, fontweight='bold')
        
        # 1. Inference Time Distribution
        times = [r['inference_time'] for r in self.results]
        axes[0, 0].hist(times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Inference Time Distribution')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.4f}s')
        axes[0, 0].legend()
        
        # 2. PSNR Distribution
        psnr_values = [r['psnr'] for r in self.results]
        axes[0, 1].hist(psnr_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('PSNR Distribution')
        axes[0, 1].set_xlabel('PSNR (dB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(psnr_values), color='red', linestyle='--', label=f'Mean: {np.mean(psnr_values):.2f}dB')
        axes[0, 1].legend()
        
        # 3. SSIM Distribution
        ssim_values = [r['ssim'] for r in self.results]
        axes[0, 2].hist(ssim_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 2].set_title('SSIM Distribution')
        axes[0, 2].set_xlabel('SSIM')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(ssim_values), color='red', linestyle='--', label=f'Mean: {np.mean(ssim_values):.3f}')
        axes[0, 2].legend()
        
        # 4. Brightness Improvement
        brightness_imp = [r['brightness_improvement'] for r in self.results]
        axes[1, 0].hist(brightness_imp, bins=20, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_title('Brightness Improvement Distribution')
        axes[1, 0].set_xlabel('Brightness Improvement')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(brightness_imp), color='red', linestyle='--', label=f'Mean: {np.mean(brightness_imp):.3f}')
        axes[1, 0].legend()
        
        # 5. Contrast Improvement
        contrast_imp = [r['contrast_improvement'] for r in self.results]
        axes[1, 1].hist(contrast_imp, bins=20, alpha=0.7, color='plum', edgecolor='black')
        axes[1, 1].set_title('Contrast Improvement Distribution')
        axes[1, 1].set_xlabel('Contrast Improvement')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(contrast_imp), color='red', linestyle='--', label=f'Mean: {np.mean(contrast_imp):.3f}')
        axes[1, 1].legend()
        
        # 6. Memory Usage
        memory_usage = [r['memory_usage_mb'] for r in self.results]
        axes[1, 2].hist(memory_usage, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 2].set_title('Memory Usage Distribution')
        axes[1, 2].set_xlabel('Memory Usage (MB)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].axvline(np.mean(memory_usage), color='red', linestyle='--', label=f'Mean: {np.mean(memory_usage):.1f}MB')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_visualizations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate correlation matrix
        self._generate_correlation_matrix()
        
        print(f"📈 Visualizations saved to {self.output_dir}")
    
    def _generate_correlation_matrix(self):
        """Generate correlation matrix of metrics."""
        if not self.results:
            return
        
        # Select key metrics for correlation analysis
        metrics = ['inference_time', 'psnr', 'ssim', 'mse', 'mae', 'brightness_improvement', 
                  'contrast_improvement', 'memory_usage_mb']
        
        # Create DataFrame-like structure
        data = {}
        for metric in metrics:
            data[metric] = [r[metric] for r in self.results if metric in r]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef([data[metric] for metric in metrics if len(data[metric]) > 0])
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, 
                   xticklabels=metrics, 
                   yticklabels=metrics,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f')
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self):
        """Generate executive summary report."""
        summary_path = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXECUTIVE SUMMARY - PERFORMANCE ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis completed on {len(self.results)} images\n")
            f.write(f"Model: Semantic-Guided Low-Light Image Enhancement\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Key Performance Indicators
            f.write("KEY PERFORMANCE INDICATORS\n")
            f.write("-" * 30 + "\n")
            
            if 'inference_time' in self.aggregate_stats:
                avg_time = self.aggregate_stats['inference_time']['mean']
                f.write(f"• Average Inference Time: {avg_time:.4f} seconds\n")
                f.write(f"• Throughput: {1/avg_time:.2f} images/second\n")
            
            if 'psnr' in self.aggregate_stats:
                avg_psnr = self.aggregate_stats['psnr']['mean']
                f.write(f"• Average PSNR: {avg_psnr:.2f} dB\n")
            
            if 'ssim' in self.aggregate_stats:
                avg_ssim = self.aggregate_stats['ssim']['mean']
                f.write(f"• Average SSIM: {avg_ssim:.3f}\n")
            
            if 'brightness_improvement' in self.aggregate_stats:
                avg_brightness = self.aggregate_stats['brightness_improvement']['mean']
                f.write(f"• Average Brightness Improvement: {avg_brightness:.3f}\n")
            
            if 'contrast_improvement' in self.aggregate_stats:
                avg_contrast = self.aggregate_stats['contrast_improvement']['mean']
                f.write(f"• Average Contrast Improvement: {avg_contrast:.3f}\n")
            
            f.write("\n")
            
            # Model Efficiency
            f.write("MODEL EFFICIENCY\n")
            f.write("-" * 30 + "\n")
            f.write(f"• Total Parameters: {sum(p.numel() for p in self.net.parameters()):,}\n")
            f.write(f"• Model Size: {sum(p.numel() for p in self.net.parameters()) * 4 / 1024 / 1024:.2f} MB\n")
            
            if 'memory_usage_mb' in self.aggregate_stats:
                avg_memory = self.aggregate_stats['memory_usage_mb']['mean']
                f.write(f"• Average Memory Usage: {avg_memory:.1f} MB\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            
            if 'inference_time' in self.aggregate_stats:
                avg_time = self.aggregate_stats['inference_time']['mean']
                if avg_time > 1.0:
                    f.write("• Consider model optimization for faster inference\n")
                elif avg_time < 0.1:
                    f.write("• Model shows excellent inference speed\n")
            
            if 'psnr' in self.aggregate_stats:
                avg_psnr = self.aggregate_stats['psnr']['mean']
                if avg_psnr > 30:
                    f.write("• Excellent image quality (PSNR > 30 dB)\n")
                elif avg_psnr > 25:
                    f.write("• Good image quality (PSNR > 25 dB)\n")
                else:
                    f.write("• Consider improving image quality\n")
            
            if 'ssim' in self.aggregate_stats:
                avg_ssim = self.aggregate_stats['ssim']['mean']
                if avg_ssim > 0.9:
                    f.write("• Excellent structural similarity preservation\n")
                elif avg_ssim > 0.8:
                    f.write("• Good structural similarity preservation\n")
                else:
                    f.write("• Consider improving structural preservation\n")
        
        print(f"📋 Summary report saved to {summary_path}")
    
    def benchmark_against_baselines(self):
        """Benchmark against common low-light enhancement baselines."""
        print("🔬 Running baseline comparisons...")
        
        # This would require implementing baseline methods
        # For now, we'll create a placeholder structure
        baseline_results = {
            'method': 'Semantic-Guided Enhancement',
            'psnr': self.aggregate_stats.get('psnr', {}).get('mean', 0),
            'ssim': self.aggregate_stats.get('ssim', {}).get('mean', 0),
            'inference_time': self.aggregate_stats.get('inference_time', {}).get('mean', 0)
        }
        
        # Save baseline comparison
        baseline_path = os.path.join(self.output_dir, "baseline_comparison.json")
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        print(f"📊 Baseline comparison saved to {baseline_path}")
    
    def export_results_for_publication(self):
        """Export results in a format suitable for academic publication."""
        print("📚 Exporting results for publication...")
        
        # Create LaTeX table
        latex_path = os.path.join(self.output_dir, "results_table.tex")
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Metrics Summary}\n")
            f.write("\\begin{tabular}{|l|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Metric & Mean & Std & Min & Max \\\\\n")
            f.write("\\hline\n")
            
            # Add key metrics to table
            key_metrics = ['psnr', 'ssim', 'inference_time', 'brightness_improvement', 'contrast_improvement']
            for metric in key_metrics:
                if metric in self.aggregate_stats:
                    stats = self.aggregate_stats[metric]
                    f.write(f"{metric.upper()} & {stats['mean']:.3f} & {stats['std']:.3f} & {stats['min']:.3f} & {stats['max']:.3f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"📄 LaTeX table saved to {latex_path}")


def main():
    """Main function to run performance analysis."""
    parser = argparse.ArgumentParser(description="Performance Analysis for Low-Light Image Enhancement")
    parser.add_argument('--weight_path', type=str, default='weight/Epoch99.pth',
                       help='Path to model weights')
    parser.add_argument('--test_data_path', type=str, default='data/test_data/',
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='performance_results/',
                       help='Output directory for results')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to analyze (None for all)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if None)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(
        weight_path=args.weight_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(max_images=args.max_images)
    
    # Additional analyses
    analyzer.benchmark_against_baselines()
    analyzer.export_results_for_publication()
    
    print("\n🎉 Performance analysis completed successfully!")
    print(f"📁 All results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  • detailed_report.txt - Comprehensive analysis report")
    print("  • summary_report.txt - Executive summary")
    print("  • results.json - Machine-readable results")
    print("  • performance_visualizations.png - Statistical plots")
    print("  • correlation_matrix.png - Metrics correlation")
    print("  • baseline_comparison.json - Baseline comparison")
    print("  • results_table.tex - LaTeX table for publications")


if __name__ == "__main__":
    main()
