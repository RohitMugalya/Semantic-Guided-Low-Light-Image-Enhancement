#!/usr/bin/env python3
"""
Minimal working example to test traditional enhancement methods
"""

import numpy as np
import cv2
from traditional_models_experiment import TraditionalEnhancer

def create_sample_image():
    """Create a simple dark test image"""
    # Create a dark image with some patterns
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Add some dark shapes
    cv2.rectangle(img, (20, 20), (80, 80), (40, 40, 40), -1)
    cv2.circle(img, (50, 50), 15, (80, 80, 80), -1)
    
    return img

def test_enhancement_methods():
    """Test a few enhancement methods"""
    print("Creating sample dark image...")
    dark_image = create_sample_image()
    
    print("Initializing enhancer...")
    enhancer = TraditionalEnhancer()
    
    # Test a few fast methods
    test_methods = ['histogram_eq', 'gamma_correction', 'exposure_correction']
    
    print(f"Testing {len(test_methods)} enhancement methods...")
    
    for method_name in test_methods:
        try:
            print(f"  Testing {method_name}...")
            enhanced = enhancer.methods[method_name](dark_image)
            
            # Basic validation
            assert enhanced.shape == dark_image.shape
            assert enhanced.dtype == np.uint8
            assert np.all(enhanced >= 0) and np.all(enhanced <= 255)
            
            print(f"  ✓ {method_name} - Success")
            
        except Exception as e:
            print(f"  ✗ {method_name} - Failed: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    print("Minimal Traditional Enhancement Test")
    print("=" * 40)
    test_enhancement_methods()