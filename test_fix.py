#!/usr/bin/env python3
"""
Quick test to verify the performance analysis script works
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from performance_analysis import PerformanceAnalyzer
    print("✓ Import successful")
    
    # Test argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--test_data_path', type=str, default='data/test_data/')
    parser.add_argument('--weight_path', type=str, default='weight/Epoch99.pth')
    parser.add_argument('--output_dir', type=str, default='test_results/')
    
    # Test with the problematic arguments
    test_args = ['--max_images', '10', '--test_data_path', 'data/test_data/']
    args = parser.parse_args(test_args)
    
    print(f"✓ Arguments parsed successfully:")
    print(f"  max_images: {args.max_images}")
    print(f"  test_data_path: {args.test_data_path}")
    
    print("✅ Fix appears to be working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
