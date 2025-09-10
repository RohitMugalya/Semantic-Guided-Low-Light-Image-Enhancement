#!/usr/bin/env python3
"""
Quick test to verify the syntax is fixed
"""

def test_syntax():
    """Test that the file can be compiled without syntax errors"""
    try:
        import py_compile
        py_compile.compile('traditional_models_experiment.py', doraise=True)
        print("✓ traditional_models_experiment.py compiles successfully")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ Compilation error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_import():
    """Test importing the module"""
    try:
        from traditional_models_experiment import TraditionalExperiment, TraditionalEnhancer
        print("✓ Import successful")
        
        # Test basic instantiation
        enhancer = TraditionalEnhancer()
        print(f"✓ Created enhancer with {len(enhancer.methods)} methods")
        
        return True
    except Exception as e:
        print(f"✗ Import/instantiation error: {e}")
        return False

if __name__ == "__main__":
    print("Testing syntax fix...")
    print("=" * 30)
    
    syntax_ok = test_syntax()
    import_ok = test_import()
    
    if syntax_ok and import_ok:
        print("\n🎉 All tests passed! The syntax errors are fixed.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")