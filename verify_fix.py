#!/usr/bin/env python3
"""
Verification script to test that the syntax errors have been fixed
"""

import sys
import traceback

def test_import():
    """Test importing the module"""
    try:
        from traditional_models_experiment import TraditionalExperiment, TraditionalEnhancer
        print("✓ Import successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_instantiation():
    """Test creating instances"""
    try:
        from traditional_models_experiment import TraditionalExperiment, TraditionalEnhancer
        
        # Test TraditionalEnhancer
        enhancer = TraditionalEnhancer()
        print(f"✓ TraditionalEnhancer created with {len(enhancer.methods)} methods")
        
        # Test TraditionalExperiment
        experiment = TraditionalExperiment(".", "test_output")
        print("✓ TraditionalExperiment created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_methods():
    """Test that methods are callable"""
    try:
        from traditional_models_experiment import TraditionalEnhancer
        import numpy as np
        
        enhancer = TraditionalEnhancer()
        
        # Create a simple test image
        test_image = np.random.randint(0, 100, (50, 50, 3), dtype=np.uint8)
        
        # Test a few methods
        test_methods = ['histogram_eq', 'gamma_correction', 'exposure_correction']
        
        for method_name in test_methods:
            if method_name in enhancer.methods:
                try:
                    result = enhancer.methods[method_name](test_image)
                    print(f"✓ {method_name} works correctly")
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
                    return False
        
        return True
    except Exception as e:
        print(f"✗ Method testing failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Verifying traditional_models_experiment.py fix...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Instantiation Test", test_instantiation),
        ("Methods Test", test_methods)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"Failed: {test_name}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The syntax errors have been fixed.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())