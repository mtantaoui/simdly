#!/usr/bin/env python3
"""
Test script for simdly-py Python bindings
"""

import math
import time
import sys
import numpy as np

try:
    import simdly
except ImportError:
    print("ERROR: simdly module not found. Please install it first:")
    print("  maturin develop")
    sys.exit(1)


def test_add():
    """Test element-wise addition functionality"""
    print("Testing element-wise addition...")

    # Basic test
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    result = simdly.add(a, b)
    expected = [6.0, 8.0, 10.0, 12.0]

    print(f"  Input a: {a}")
    print(f"  Input b: {b}")
    print(f"  Result:  {result}")
    print(f"  Expected: {expected}")

    # Check results
    for i, (r, e) in enumerate(zip(result, expected)):
        if abs(r - e) > 1e-6:
            print(f"  ERROR: Mismatch at index {i}: got {r}, expected {e}")
            return False

    print("  ✓ Basic addition test passed")

    # # Test error handling
    # try:
    #     simdly.add([1.0, 2.0], [1.0])  # Different lengths
    #     print("  ERROR: Should have raised ValueError for different lengths")
    #     return False
    # except ValueError:
    #     print("  ✓ Error handling test passed")

    return True


def test_cos():
    """Test cosine computation functionality"""
    print("Testing cosine computation...")

    # Basic test with known values
    angles = np.array([0.0, math.pi / 4, math.pi / 2, math.pi], dtype=np.float32)
    result = simdly.cos(angles)
    expected = [1.0, math.cos(math.pi / 4), 0.0, -1.0]

    print(f"  Input angles: {angles}")
    print(f"  Result:       {result}")
    print(f"  Expected:     {expected}")

    # Check results with tolerance for floating point precision
    tolerances = [1e-6, 1e-5, 1e-5, 1e-6]  # Different tolerances for different values
    for i, (r, e, tol) in enumerate(zip(result, expected, tolerances)):
        if abs(r - e) > tol:
            print(
                f"  ERROR: Mismatch at index {i}: got {r}, expected {e}, tolerance {tol}"
            )
            return False

    print("  ✓ Basic cosine test passed")

    # # Test error handling
    # try:
    #     simdly.cos([])  # Empty array
    #     print("  ERROR: Should have raised ValueError for empty array")
    #     return False
    # except ValueError:
    #     print("  ✓ Error handling test passed")

    return True


def benchmark_performance():
    """Benchmark performance of SIMD operations"""
    print("Running performance benchmarks...")

    # Test different array sizes
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    for size in sizes:
        print(f"\n  Array size: {size}")

        # Create test data
        a = np.array([float(i) for i in range(size)], dtype=np.float32)
        b = np.array([float(i * 2) for i in range(size)], dtype=np.float32)
        angles = np.array([float(i) * 0.01 for i in range(size)], dtype=np.float32)

        # Benchmark addition
        start_time = time.perf_counter()
        result_add = simdly.add(a, b)
        add_time = time.perf_counter() - start_time
        print(f"    Addition time: {add_time:.6f}s")

        # Benchmark cosine
        start_time = time.perf_counter()
        result_cos = simdly.cos(angles)
        cos_time = time.perf_counter() - start_time
        print(f"    Cosine time:   {cos_time:.6f}s")

        # Compare with Python built-ins for reference
        start_time = time.perf_counter()
        python_add = a + b
        python_add_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        python_cos = np.cos(angles)
        python_cos_time = time.perf_counter() - start_time

        print(
            f"    Python add:    {python_add_time:.6f}s (speedup: {python_add_time/add_time:.1f}x)"
        )
        print(
            f"    Python cos:    {python_cos_time:.6f}s (speedup: {python_cos_time/cos_time:.1f}x)"
        )


def main():
    """Run all tests"""
    print("simdly-py Test Suite")
    print("=" * 50)

    success = True

    if not test_add():
        success = False

    print()

    if not test_cos():
        success = False

    print()
    benchmark_performance()

    print("\n" + "=" * 50)
    if success:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
