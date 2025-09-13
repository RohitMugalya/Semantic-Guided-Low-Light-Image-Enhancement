#!/usr/bin/env python3
"""
Simple runner script for traditional models experiment

This script provides easy-to-use functions to run traditional low-light enhancement experiments
without command line arguments.
"""

import os
import sys

try:
    from traditional_models_experiment import TraditionalExperiment
except ImportError as e:
    print(f"Error importing TraditionalExperiment: {e}")
    print("Please make sure traditional_models_experiment.py is in the same directory.")
    sys.exit(1)

def run_basic_experiment():
    """Run basic experiment with default settings"""
    input_dir = "data/test_data/DICM"  # Change this to your test data path
    output_dir = "traditional_results"
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("Please update the input_dir path in this script or create test data")
        return
    
    experiment = TraditionalExperiment(input_dir, output_dir)
    
    # Run with selected methods (faster than all methods)
    selected_methods = ['histogram_eq', 'clahe', 'gamma_correction', 'retinex_ssr', 'retinex_msr', 'lime', 'exposure_correction', 'adaptive_gamma']
    
    print("Running traditional enhancement experiment...")
    results = experiment.run_experiment(methods=selected_methods)
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    for method, stats in results.items():
        print(f"{method:20s}: {stats['avg_time']:.4f}s avg, {stats['total_images']} images")

def create_single_comparison():
    """Create comparison for a single image"""
    # Use glob to find the first available image in DICM directory
    import glob
    dicm_dir = "data/test_data/DICM"
    image_files = glob.glob(os.path.join(dicm_dir, "*.jpg"))
    
    if not image_files:
        print(f"No .jpg images found in {dicm_dir}")
        print("Please check the directory path and ensure it contains .jpg files")
        return
    
    # Use the first available image
    image_path = image_files[0]
    image_path = "data/test_data/lowCUT/1.png"
    print(f"Using image: {image_path}")
    
    experiment = TraditionalExperiment(".", "traditional_results")
    
    # Create comparison grid
    methods = ['histogram_eq', 'clahe', 'gamma_correction', 'retinex_ssr', 'retinex_msr', 'lime', 'exposure_correction', 'adaptive_gamma']
    experiment.create_comparison_grid(
        image_path, 
        methods=methods,
        save_path="traditional_comparison.png"
    )

if __name__ == "__main__":
    print("Traditional Low-Light Enhancement Experiment")
    print("=" * 50)
    print("1. Running basic experiment...")
    run_basic_experiment()
    
    print("\n2. Creating single image comparison...")
    create_single_comparison()