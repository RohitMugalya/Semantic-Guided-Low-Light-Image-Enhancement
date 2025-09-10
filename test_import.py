#!/usr/bin/env python3
"""
Simple test to verify the import works correctly
"""

try:
    from traditional_models_experiment import TraditionalExperiment, TraditionalEnhancer
    print("✓ Import successful!")
    
    # Test basic instantiation
    enhancer = TraditionalEnhancer()
    print(f"✓ TraditionalEnhancer created with {len(enhancer.methods)} methods")
    
    # List available methods
    print("Available methods:")
    for method in enhancer.methods.keys():
        print(f"  - {method}")
    
    print("\n✓ All tests passed! The syntax error has been fixed.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")