#!/usr/bin/env python3
"""
Usage Example for Performance Analysis Script

This script demonstrates how to use the performance_analysis.py script
to evaluate the Semantic-Guided Low-Light Image Enhancement model.
"""

import os
import sys
from performance_analysis import PerformanceAnalyzer

def quick_analysis_example():
    """Run a quick analysis on a subset of test images."""
    print("🚀 Running Quick Performance Analysis Example")
    print("=" * 50)
    
    # Initialize analyzer with default settings
    analyzer = PerformanceAnalyzer(
        weight_path="weight/Epoch99.pth",
        test_data_path="data/test_data/",
        output_dir="quick_analysis_results/"
    )
    
    # Run analysis on first 5 images for quick testing
    print("Analyzing first 5 test images...")
    results = analyzer.run_comprehensive_analysis(max_images=5)
    
    # Print quick summary
    if results:
        print("\n📊 Quick Summary:")
        print(f"  • Images analyzed: {len(results)}")
        
        # Calculate average metrics
        avg_psnr = sum(r['psnr'] for r in results) / len(results)
        avg_ssim = sum(r['ssim'] for r in results) / len(results)
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        
        print(f"  • Average PSNR: {avg_psnr:.2f} dB")
        print(f"  • Average SSIM: {avg_ssim:.3f}")
        print(f"  • Average inference time: {avg_time:.4f} seconds")
        print(f"  • Throughput: {1/avg_time:.2f} images/second")
    
    print("\n✅ Quick analysis completed!")
    print("📁 Check 'quick_analysis_results/' directory for detailed results")

def full_analysis_example():
    """Run full analysis on all test images."""
    print("🔬 Running Full Performance Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(
        weight_path="weight/Epoch99.pth",
        test_data_path="data/test_data/",
        output_dir="full_analysis_results/"
    )
    
    # Run comprehensive analysis on all images
    print("Analyzing all test images...")
    results = analyzer.run_comprehensive_analysis()
    
    # Print summary
    if results:
        print(f"\n📊 Analysis Summary:")
        print(f"  • Total images analyzed: {len(results)}")
        
        # Get aggregate statistics
        if hasattr(analyzer, 'aggregate_stats'):
            stats = analyzer.aggregate_stats
            
            if 'psnr' in stats:
                print(f"  • PSNR: {stats['psnr']['mean']:.2f} ± {stats['psnr']['std']:.2f} dB")
            if 'ssim' in stats:
                print(f"  • SSIM: {stats['ssim']['mean']:.3f} ± {stats['ssim']['std']:.3f}")
            if 'inference_time' in stats:
                print(f"  • Inference time: {stats['inference_time']['mean']:.4f} ± {stats['inference_time']['std']:.4f} seconds")
                print(f"  • Throughput: {1/stats['inference_time']['mean']:.2f} images/second")
    
    print("\n✅ Full analysis completed!")
    print("📁 Check 'full_analysis_results/' directory for detailed results")

def custom_analysis_example():
    """Run analysis with custom parameters."""
    print("⚙️ Running Custom Performance Analysis")
    print("=" * 50)
    
    # Custom parameters
    custom_params = {
        'weight_path': 'weight/Epoch99.pth',
        'test_data_path': 'data/test_data/LOL/',  # Analyze only LOL dataset
        'output_dir': 'custom_analysis_results/',
        'max_images': 10  # Limit to 10 images
    }
    
    # Initialize analyzer with custom parameters
    analyzer = PerformanceAnalyzer(**custom_params)
    
    # Run analysis
    print(f"Analyzing {custom_params['max_images']} images from {custom_params['test_data_path']}...")
    results = analyzer.run_comprehensive_analysis(max_images=custom_params['max_images'])
    
    if results:
        print(f"\n📊 Custom Analysis Results:")
        print(f"  • Images analyzed: {len(results)}")
        print(f"  • Dataset: {custom_params['test_data_path']}")
        
        # Show individual results
        print("\nIndividual Image Results:")
        for i, result in enumerate(results[:3]):  # Show first 3
            print(f"  Image {i+1}: {os.path.basename(result['image_path'])}")
            print(f"    PSNR: {result['psnr']:.2f} dB")
            print(f"    SSIM: {result['ssim']:.3f}")
            print(f"    Time: {result['inference_time']:.4f}s")
    
    print("\n✅ Custom analysis completed!")
    print("📁 Check 'custom_analysis_results/' directory for detailed results")

def main():
    """Main function to run examples."""
    print("Semantic-Guided Low-Light Image Enhancement - Performance Analysis Examples")
    print("=" * 80)
    
    # Check if required files exist
    if not os.path.exists("weight/Epoch99.pth"):
        print("❌ Error: Model weights not found at 'weight/Epoch99.pth'")
        print("   Please ensure the model weights are available before running analysis.")
        return
    
    if not os.path.exists("data/test_data/"):
        print("❌ Error: Test data directory not found at 'data/test_data/'")
        print("   Please ensure the test data is available before running analysis.")
        return
    
    print("✅ Required files found. Starting examples...\n")
    
    # Run examples
    try:
        # Example 1: Quick analysis
        quick_analysis_example()
        print("\n" + "="*80 + "\n")
        
        # Example 2: Custom analysis
        custom_analysis_example()
        print("\n" + "="*80 + "\n")
        
        # Example 3: Full analysis (commented out to avoid long execution)
        # Uncomment the following lines to run full analysis
        # print("Note: Full analysis is commented out to avoid long execution time.")
        # print("Uncomment the following lines in the script to run full analysis:")
        # print("# full_analysis_example()")
        
        print("🎉 All examples completed successfully!")
        print("\nTo run the full analysis script directly, use:")
        print("python performance_analysis.py --help")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check that all dependencies are installed and paths are correct.")

if __name__ == "__main__":
    main()
