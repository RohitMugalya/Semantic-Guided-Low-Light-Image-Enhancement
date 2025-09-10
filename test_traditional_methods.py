#!/usr/bin/env python3
"""
Test script for traditional enhancement methods

This script tests the traditional enhancement methods with a simple synthetic image
to verify the implementation works correctly.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from traditional_models_experiment import TraditionalEnhancer

def create_test_image():
    """Create a synthetic low-light test image"""
    # Create a simple test pattern
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Add some geometric shapes with low brightness
    cv2.rectangle(img, (50, 50), (150, 150), (30, 30, 30), -1)  # Dark square
    cv2.circle(img, (100, 100), 30, (60, 60, 60), -1)  # Slightly brighter circle
    cv2.rectangle(img, (75, 75), (125, 125), (90, 90, 90), 2)  # Border
    
    # Add some noise to simulate real low-light conditions
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def test_all_methods():
    """Test all traditional enhancement methods"""
    print("Testing Traditional Enhancement Methods")
    print("=" * 50)
    
    # Create test image
    test_img = create_test_image()
    
    # Initialize enhancer
    enhancer = TraditionalEnhancer()
    
    # Test each method
    results = {}
    for method_name, method_func in enhancer.methods.items():
        try:
            print(f"Testing {method_name}...")
            enhanced = method_func(test_img)
            
            # Basic validation
            assert enhanced.shape == test_img.shape, f"Shape mismatch in {method_name}"
            assert enhanced.dtype == np.uint8, f"Wrong dtype in {method_name}"
            assert np.all(enhanced >= 0) and np.all(enhanced <= 255), f"Invalid range in {method_name}"
            
            results[method_name] = enhanced
            print(f"  ✓ {method_name} passed")
            
        except Exception as e:
            print(f"  ✗ {method_name} failed: {e}")
            results[method_name] = None
    
    return test_img, results

def visualize_results(original, results):
    """Create visualization of all results"""
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_methods = len(valid_results) + 1  # +1 for original
    
    cols = 4
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Show original
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original (Synthetic Low-Light)')
    axes[0, 0].axis('off')
    
    # Show enhanced versions
    for i, (method_name, enhanced) in enumerate(valid_results.items()):
        row = (i + 1) // cols
        col = (i + 1) % cols
        
        axes[row, col].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(method_name.replace('_', ' ').title())
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_methods, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('traditional_methods_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'traditional_methods_test.png'")
    plt.show()

def main():
    """Main test function"""
    # Test all methods
    original, results = test_all_methods()
    
    # Count successful methods
    successful = sum(1 for v in results.values() if v is not None)
    total = len(results)
    
    print(f"\nTest Summary: {successful}/{total} methods passed")
    
    if successful > 0:
        print("Creating visualization...")
        visualize_results(original, results)
    else:
        print("No methods passed. Please check the implementation.")

if __name__ == "__main__":
    main()