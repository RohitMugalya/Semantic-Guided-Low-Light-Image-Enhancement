"""
Traditional Low-Light Image Enhancement Models Experimentation Script

This script implements traditional methods for low-light image enhancement
as referenced in the paper "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video Enhancement"

Traditional methods implemented:
1. Histogram Equalization (HE)
2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
3. Gamma Correction
4. Retinex-based Enhancement
5. LIME (Low-light Image Enhancement via Illumination Map Estimation)
6. Simple Exposure Correction

Author: Based on research paper 2110.00970v4.pdf
"""

import os
import cv2
import numpy as np
import glob
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import exposure, filters
from skimage.restoration import denoise_bilateral
import warnings
warnings.filterwarnings('ignore')

class TraditionalEnhancer:
    """Traditional low-light image enhancement methods"""
    
    def __init__(self):
        self.methods = {
            'histogram_eq': self.histogram_equalization,
            'clahe': self.clahe_enhancement,
            'gamma_correction': self.gamma_correction,
            'retinex_ssr': self.single_scale_retinex,
            'retinex_msr': self.multi_scale_retinex,
            'lime': self.lime_enhancement,
            'exposure_correction': self.exposure_correction,
            'adaptive_gamma': self.adaptive_gamma_correction
        }  
  
    def histogram_equalization(self, image):
        """Traditional Histogram Equalization"""
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            enhanced = cv2.equalizeHist(image)
        return enhanced
    
    def clahe_enhancement(self, image, clip_limit=3.0, tile_grid_size=(8,8)):
        """Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = clahe.apply(image)
        return enhanced
    
    def gamma_correction(self, image, gamma=0.5):
        """Gamma correction for brightness adjustment"""
        # Normalize to [0,1]
        normalized = image.astype(np.float32) / 255.0
        # Apply gamma correction
        corrected = np.power(normalized, gamma)
        # Convert back to [0,255]
        enhanced = (corrected * 255).astype(np.uint8)
        return enhanced
    
    def adaptive_gamma_correction(self, image):
        """Adaptive gamma correction based on image statistics"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate adaptive gamma based on mean intensity
        mean_intensity = np.mean(gray) / 255.0
        gamma = -np.log2(mean_intensity)
        gamma = np.clip(gamma, 0.3, 2.5)  # Limit gamma range
        
        return self.gamma_correction(image, gamma)
    
    def single_scale_retinex(self, image, sigma=15):
        """Single Scale Retinex (SSR)"""
        image_float = image.astype(np.float32) + 1.0  # Add 1 to avoid log(0)
        
        if len(image.shape) == 3:
            enhanced = np.zeros_like(image_float)
            for i in range(3):
                # Gaussian blur
                blurred = gaussian_filter(image_float[:,:,i], sigma=sigma)
                # Retinex: log(I) - log(I*G)
                enhanced[:,:,i] = np.log(image_float[:,:,i]) - np.log(blurred)
        else:
            blurred = gaussian_filter(image_float, sigma=sigma)
            enhanced = np.log(image_float) - np.log(blurred)
        
        # Normalize to [0,255]
        enhanced = exposure.rescale_intensity(enhanced, out_range=(0, 255))
        return enhanced.astype(np.uint8)
    
    def multi_scale_retinex(self, image, sigmas=[15, 80, 250]):
        """Multi Scale Retinex (MSR)"""
        retinex_sum = np.zeros_like(image, dtype=np.float32)
        
        for sigma in sigmas:
            retinex_sum += self.single_scale_retinex(image, sigma).astype(np.float32)
        
        # Average the results
        enhanced = retinex_sum / len(sigmas)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        return enhanced
    
    def lime_enhancement(self, image, alpha=0.5, rho=2.0):
        """LIME: Low-light Image Enhancement via Illumination Map Estimation"""
        if len(image.shape) == 3:
            # Convert to grayscale for illumination estimation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Estimate illumination map using max RGB
        if len(image.shape) == 3:
            illum_map = np.max(image, axis=2).astype(np.float32) / 255.0
        else:
            illum_map = gray.astype(np.float32) / 255.0
        
        # Smooth the illumination map
        illum_map = gaussian_filter(illum_map, sigma=5)
        
        # Avoid division by zero
        illum_map = np.maximum(illum_map, 0.1)
        
        # Enhance the image
        enhanced = image.astype(np.float32) / 255.0
        if len(image.shape) == 3:
            for i in range(3):
                enhanced[:,:,i] = enhanced[:,:,i] / (illum_map ** alpha)
        else:
            enhanced = enhanced / (illum_map ** alpha)
        
        # Normalize and convert back
        enhanced = exposure.rescale_intensity(enhanced, out_range=(0, 255))
        return enhanced.astype(np.uint8)
    
    def exposure_correction(self, image, exposure_factor=1.5):
        """Simple exposure correction"""
        enhanced = image.astype(np.float32) * exposure_factor
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        return enhanced

class TraditionalExperiment:
    """Experiment runner for traditional enhancement methods"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enhancer = TraditionalEnhancer()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for method in self.enhancer.methods.keys():
            os.makedirs(os.path.join(output_dir, method), exist_ok=True)
    
    def load_image(self, image_path):
        """Load image using OpenCV"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    
    def save_image(self, image, output_path):
        """Save enhanced image"""
        cv2.imwrite(output_path, image)
    
    def enhance_single_image(self, image_path, method_name):
        """Enhance single image with specified method"""
        image = self.load_image(image_path)
        
        if method_name not in self.enhancer.methods:
            raise ValueError(f"Unknown method: {method_name}")
        
        start_time = time.time()
        enhanced = self.enhancer.methods[method_name](image)
        processing_time = time.time() - start_time
        
        return enhanced, processing_time
    
    def run_experiment(self, methods=None):
        """Run enhancement experiment on all images"""
        if methods is None:
            methods = list(self.enhancer.methods.keys())
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.input_dir, ext.upper())))
        
        if not image_files:
            print(f"No images found in {self.input_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        print(f"Testing methods: {methods}")
        
        results = {}
        
        for method in methods:
            print(f"\nProcessing with {method}...")
            method_times = []
            
            for i, image_path in enumerate(image_files):
                try:
                    enhanced, proc_time = self.enhance_single_image(image_path, method)
                    method_times.append(proc_time)
                    
                    # Save enhanced image
                    filename = os.path.basename(image_path)
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(self.output_dir, method, f"{name}_{method}{ext}")
                    self.save_image(enhanced, output_path)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(image_files)} images")
                        
                except Exception as e:
                    print(f"  Error processing {image_path}: {e}")
            
            avg_time = np.mean(method_times) if method_times else 0
            results[method] = {
                'avg_time': avg_time,
                'total_images': len(method_times)
            }
            print(f"  Average processing time: {avg_time:.4f}s")
        
        return results    
    
    def create_comparison_grid(self, image_path, methods=None, save_path=None):
        """Create a comparison grid for a single image"""
        if methods is None:
            methods = list(self.enhancer.methods.keys())
        
        original = self.load_image(image_path)
        
        # Calculate grid size
        n_methods = len(methods) + 1  # +1 for original
        cols = min(4, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Show original
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Show enhanced versions
        for i, method in enumerate(methods):
            row = (i + 1) // cols
            col = (i + 1) % cols
            
            try:
                enhanced, _ = self.enhance_single_image(image_path, method)
                axes[row, col].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(method.replace('_', ' ').title())
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error: {str(e)}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(method.replace('_', ' ').title())
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_methods, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison grid saved to: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Traditional Low-Light Enhancement Experiment')
    parser.add_argument('--input_dir', type=str, default='data/test_data/DICM', 
                       help='Input directory containing low-light images')
    parser.add_argument('--output_dir', type=str, default='traditional_results',
                       help='Output directory for enhanced images')
    parser.add_argument('--methods', nargs='+', 
                       choices=['histogram_eq', 'clahe', 'gamma_correction', 'retinex_ssr', 
                               'retinex_msr', 'lime', 'exposure_correction', 'adaptive_gamma'],
                       help='Enhancement methods to test')
    parser.add_argument('--comparison_image', type=str,
                       help='Path to single image for comparison grid')
    parser.add_argument('--comparison_output', type=str, default='comparison_grid.png',
                       help='Output path for comparison grid')
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = TraditionalExperiment(args.input_dir, args.output_dir)
    
    if args.comparison_image:
        # Create comparison grid for single image
        if os.path.exists(args.comparison_image):
            experiment.create_comparison_grid(
                args.comparison_image, 
                methods=args.methods,
                save_path=args.comparison_output
            )
        else:
            print(f"Comparison image not found: {args.comparison_image}")
    else:
        # Run full experiment
        if not os.path.exists(args.input_dir):
            print(f"Input directory not found: {args.input_dir}")
            return
        
        results = experiment.run_experiment(methods=args.methods)
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        for method, stats in results.items():
            print(f"{method:20s}: {stats['avg_time']:.4f}s avg, {stats['total_images']} images")

if __name__ == "__main__":
    main()