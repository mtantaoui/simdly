#!/usr/bin/env python3
"""
Criterion-style benchmarking script for simdly-py using pytest-benchmark
"""

import pytest
import numpy as np
import sys

try:
    import simdly
except ImportError:
    print("ERROR: simdly module not found. Please install it first:")
    print("  maturin develop")
    sys.exit(1)


class TestBenchmarks:
    """Benchmark suite for simdly operations"""

    @pytest.mark.parametrize("size", [100, 1_000, 10_000, 100_000])
    def test_bench_add(self, benchmark, size):
        """Benchmark addition operation"""
        a = np.array([float(i) for i in range(size)], dtype=np.float32)
        b = np.array([float(i * 2) for i in range(size)], dtype=np.float32)

        result = benchmark(simdly.add, a, b)
        assert len(result) == size

    @pytest.mark.parametrize("size", [100, 1_000, 10_000, 100_000])
    def test_bench_cos(self, benchmark, size):
        """Benchmark cosine operation"""
        angles = np.array([float(i) * 0.01 for i in range(size)], dtype=np.float32)

        result = benchmark(simdly.cos, angles)
        assert len(result) == size

    @pytest.mark.parametrize("size", [100, 1_000, 10_000, 100_000])
    def test_bench_numpy_add(self, benchmark, size):
        """Benchmark numpy addition for comparison"""
        a = np.array([float(i) for i in range(size)], dtype=np.float32)
        b = np.array([float(i * 2) for i in range(size)], dtype=np.float32)

        result = benchmark(np.add, a, b)
        assert len(result) == size

    @pytest.mark.parametrize("size", [100, 1_000, 10_000, 100_000])
    def test_bench_numpy_cos(self, benchmark, size):
        """Benchmark numpy cosine for comparison"""
        angles = np.array([float(i) * 0.01 for i in range(size)], dtype=np.float32)

        result = benchmark(np.cos, angles)
        assert len(result) == size


if __name__ == "__main__":
    print("Run with: pytest bench.py --benchmark-only --benchmark-sort=name")
    print("Or install pytest-benchmark: pip install pytest-benchmark")
