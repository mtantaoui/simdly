#!/usr/bin/env python3
"""
Test the updated add function with numpy arrays
"""

import numpy as np
import simdly


def test_numpy_add():
    """Test the add function with numpy arrays"""
    
    print("Testing numpy array add function:")
    print("=" * 40)
    
    # Test 1: Basic addition
    print("\n1. Basic addition:")
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = simdly.add(a, b)
    
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result:  {result}")
    print(f"Type: {type(result)}")
    print(f"dtype: {result.dtype}")
    
    # Test 2: Zero addition
    print("\n2. Zero addition:")
    zeros = np.zeros(5, dtype=np.float32)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = simdly.add(values, zeros)
    simdly.print_array(result, "Values + Zeros")
    
    # Test 3: Negative values
    print("\n3. Negative values:")
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    neg = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    result = simdly.add(pos, neg)
    simdly.print_array(result, "Pos + Neg")
    
    # Test 4: Large arrays (performance comparison coming up)
    print("\n4. Large array (1000 elements):")
    large_a = np.arange(1000, dtype=np.float32)
    large_b = np.arange(1000, 2000, dtype=np.float32)
    result = simdly.add(large_a, large_b)
    print(f"Sum of first 10 elements: {result[:10]}")
    
    # Test 5: Error handling - different lengths
    print("\n5. Error handling - different lengths:")
    try:
        a_short = np.array([1.0, 2.0], dtype=np.float32)
        b_long = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = simdly.add(a_short, b_long)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test 6: Error handling - empty arrays
    print("\n6. Error handling - empty arrays:")
    try:
        empty = np.array([], dtype=np.float32)
        result = simdly.add(empty, empty)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\n" + "=" * 40)
    print("All tests completed!")


if __name__ == "__main__":
    test_numpy_add()