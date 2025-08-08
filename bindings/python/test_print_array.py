#!/usr/bin/env python3
"""
Test module for the print_array function
"""

import numpy as np
import simdly


def test_print_array():
    """Test the print_array function with various inputs"""
    
    print("Testing print_array function:")
    print("=" * 40)
    
    # Test 1: Basic array
    print("\n1. Basic array:")
    arr1 = np.array([1.0, 2.5, 3.14159, -4.7], dtype=np.float32)
    simdly.print_array(arr1, "Basic")
    
    # Test 2: Default prefix
    print("\n2. Default prefix:")
    simdly.print_array(arr1)
    
    # Test 3: Empty array
    print("\n3. Empty array:")
    empty_arr = np.array([], dtype=np.float32)
    simdly.print_array(empty_arr, "Empty")
    
    # Test 4: Single element
    print("\n4. Single element:")
    single = np.array([42.123456], dtype=np.float32)
    simdly.print_array(single, "Single")
    
    # Test 5: Large values
    print("\n5. Large values:")
    large = np.array([1e6, -1e-6, 0.0, np.pi], dtype=np.float32)
    simdly.print_array(large, "Large")
    
    # Test 6: Result from simdly operations
    print("\n6. Results from simdly operations:")
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = simdly.add(a.tolist(), b.tolist())
    result_array = np.array(result, dtype=np.float32)
    simdly.print_array(result_array, "Add Result")
    
    # Test 7: Cosine results
    print("\n7. Cosine results:")
    angles = np.array([0.0, np.pi/4, np.pi/2, np.pi], dtype=np.float32)
    cos_result = simdly.cos(angles.tolist())
    cos_array = np.array(cos_result, dtype=np.float32)
    simdly.print_array(cos_array, "Cos Result")
    
    print("\n" + "=" * 40)
    print("All tests completed!")


if __name__ == "__main__":
    test_print_array()