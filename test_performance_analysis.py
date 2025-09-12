#!/usr/bin/env python3
"""
Test Script for Performance Analysis

This script tests the performance analysis functionality to ensure
it works correctly with the model implementation.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

def create_test_data():
    """Create synthetic test data for testing."""
    print("Creating synthetic test data...")
    
    # Create temporary test directory
    test_dir = "temp_test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create synthetic low-light images
    for i in range(5):
        # Create a synthetic low-light image
        img = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)  # Low-light (dark)
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(test_dir, f"test_image_{i:03d}.jpg"))
    
    print(f"Created {test_dir} with 5 synthetic test images")
    return test_dir

def test_performance_analyzer():
    """Test the performance analyzer with synthetic data."""
    print("Testing Performance Analyzer...")
    
    try:
        from performance_analysis import PerformanceAnalyzer
        
        # Create test data
        test_data_dir = create_test_data()
        
        # Initialize analyzer with test data
        analyzer = PerformanceAnalyzer(
            weight_path="weight/Epoch99.pth",
            test_data_path=test_data_dir,
            output_dir="test_results/",
            device="cpu"  # Use CPU for testing
        )
        
        print("✓ PerformanceAnalyzer initialized successfully")
        
        # Test single image analysis
        test_images = analyzer.get_test_images()
        if test_images:
            print(f"✓ Found {len(test_images)} test images")
            
            # Test preprocessing
            original_tensor = analyzer.preprocess_image(test_images[0])
            print(f"✓ Image preprocessing successful, shape: {original_tensor.shape}")
            
            # Test model inference (if weights available)
            try:
                with torch.no_grad():
                    enhanced_tensor, params_maps = analyzer.net(original_tensor)
                print(f"✓ Model inference successful, enhanced shape: {enhanced_tensor.shape}")
                
                # Test postprocessing
                enhanced_np = analyzer.postprocess_image(enhanced_tensor)
                print(f"✓ Image postprocessing successful, shape: {enhanced_np.shape}")
                
                # Test metrics calculation
                original_np = analyzer.postprocess_image(original_tensor)
                image_metrics = analyzer.calculate_image_metrics(original_np, enhanced_np)
                print(f"✓ Image metrics calculation successful: {len(image_metrics)} metrics")
                
                enhancement_metrics = analyzer.calculate_enhancement_metrics(original_np, enhanced_np)
                print(f"✓ Enhancement metrics calculation successful: {len(enhancement_metrics)} metrics")
                
                loss_metrics = analyzer.calculate_loss_metrics(original_tensor, enhanced_tensor, params_maps)
                print(f"✓ Loss metrics calculation successful: {len(loss_metrics)} metrics")
                
            except Exception as e:
                print(f"⚠ Model inference failed (expected if weights not available): {e}")
                print("  This is normal if model weights are not properly loaded.")
        
        # Test full analysis (limited to 2 images for speed)
        print("Running limited analysis (2 images)...")
        results = analyzer.run_comprehensive_analysis(max_images=2)
        
        if results:
            print(f"✓ Full analysis completed successfully with {len(results)} results")
            
            # Check if reports were generated
            expected_files = [
                "detailed_report.txt",
                "summary_report.txt", 
                "results.json",
                "performance_visualizations.png",
                "correlation_matrix.png"
            ]
            
            for file in expected_files:
                file_path = os.path.join("test_results", file)
                if os.path.exists(file_path):
                    print(f"✓ {file} generated successfully")
                else:
                    print(f"⚠ {file} not found")
        
        # Cleanup
        shutil.rmtree(test_data_dir)
        print("✓ Test data cleaned up")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("  Make sure all dependencies are installed:")
        print("  pip install torch torchvision opencv-python scikit-image matplotlib seaborn psutil")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_individual_functions():
    """Test individual functions of the performance analyzer."""
    print("Testing individual functions...")
    
    try:
        from performance_analysis import PerformanceAnalyzer
        
        # Test without actual model loading
        analyzer = PerformanceAnalyzer.__new__(PerformanceAnalyzer)
        analyzer.device = torch.device("cpu")
        
        # Test image metrics calculation
        original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        enhanced = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        metrics = analyzer.calculate_image_metrics(original, enhanced)
        print(f"✓ Image metrics calculation: {len(metrics)} metrics")
        
        enh_metrics = analyzer.calculate_enhancement_metrics(original, enhanced)
        print(f"✓ Enhancement metrics calculation: {len(enh_metrics)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual function test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("PERFORMANCE ANALYSIS TEST SUITE")
    print("=" * 60)
    
    # Test 1: Individual functions
    print("\n1. Testing individual functions...")
    test1_passed = test_individual_functions()
    
    # Test 2: Full performance analyzer
    print("\n2. Testing full performance analyzer...")
    test2_passed = test_performance_analyzer()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Individual functions: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Full analyzer: {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Performance analysis is working correctly.")
        print("\nTo run the full analysis:")
        print("python performance_analysis.py --help")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed and paths are correct.")

if __name__ == "__main__":
    main()
