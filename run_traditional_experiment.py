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
    selected_methods = ['histogram_eq', 'clahe', 'gamma_correction', 'retinex_ssr', 'lime']
    
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
    # Update this path to point to a specific low-light image
    image_path = "data/test_data/DICM/1.jpg"  # Change this path
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update the image_path in this script")
        return
    
    experiment = TraditionalExperiment(".", "traditional_results")
    
    # Create comparison grid
    methods = ['histogram_eq', 'clahe', 'gamma_correction', 'retinex_ssr', 'lime']
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