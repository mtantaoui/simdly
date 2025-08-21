"""
Comprehensive benchmark comparison between simdly and numpy functions.

This benchmark compares performance of:
- simdly.cos vs numpy.cos for 1D arrays
- simdly.outer_product vs numpy.outer for 1D arrays

Demonstrates SIMD acceleration benefits across various array sizes.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Any
import sys
import os
import math

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bindings/python"))

try:
    import simdly

    SIMDLY_AVAILABLE = True
except ImportError:
    print("Warning: simdly not available. Please build the Python bindings first.")
    print("Run: cd bindings/python && pip install -e .")
    SIMDLY_AVAILABLE = False


def time_function(func: Callable, *args, num_runs: int = 100) -> float:
    """Time a function over multiple runs and return average time."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)


def verify_cos_correctness():
    """Verify that simdly.cos produces the same results as numpy.cos."""
    if not SIMDLY_AVAILABLE:
        print("Cannot verify cos correctness: simdly not available")
        return

    print("Verifying cos correctness...")

    test_cases = [
        [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi],
        [-math.pi, -math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2],
        [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
    ]

    for i, test_values in enumerate(test_cases):
        a = np.array(test_values, dtype=np.float32)

        numpy_result = np.cos(a)
        simdly_result = simdly.cos(a)

        if np.allclose(numpy_result, simdly_result, rtol=1e-6):
            print(f"  Cos test case {i+1}: ✓ PASSED")
        else:
            print(f"  Cos test case {i+1}: ✗ FAILED")
            print(f"    NumPy result: {numpy_result}")
            print(f"    SIMDLY result: {simdly_result}")
            print(f"    Max difference: {np.max(np.abs(numpy_result - simdly_result))}")


def verify_outer_correctness():
    """Verify that simdly.outer_product and par_outer_product produce the same results as numpy.outer."""
    if not SIMDLY_AVAILABLE:
        print("Cannot verify outer correctness: simdly not available")
        return

    print("Verifying outer product correctness...")

    test_cases = [
        ([1.0, 2.0, 3.0], [4.0, 5.0]),
        ([1.0], [1.0, 2.0, 3.0, 4.0]),
        ([2.5, -1.5, 0.0, 3.7], [1.1, -2.2]),
        ([-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]),
    ]

    for i, (a_list, b_list) in enumerate(test_cases):
        a = np.array(a_list, dtype=np.float32)
        b = np.array(b_list, dtype=np.float32)

        numpy_result = np.outer(a, b)
        simdly_result = simdly.outer_product(a, b)
        par_simdly_result = simdly.par_outer_product(a, b)

        if np.allclose(numpy_result, simdly_result, rtol=1e-6):
            print(f"  Outer test case {i+1}: ✓ PASSED")
        else:
            print(f"  Outer test case {i+1}: ✗ FAILED")
            print(f"    NumPy result: {numpy_result}")
            print(f"    SIMDLY result: {simdly_result}")
            print(f"    Max difference: {np.max(np.abs(numpy_result - simdly_result))}")

        if np.allclose(numpy_result, par_simdly_result, rtol=1e-6):
            print(f"  Par Outer test case {i+1}: ✓ PASSED")
        else:
            print(f"  Par Outer test case {i+1}: ✗ FAILED")
            print(f"    NumPy result: {numpy_result}")
            print(f"    Par SIMDLY result: {par_simdly_result}")
            print(
                f"    Max difference: {np.max(np.abs(numpy_result - par_simdly_result))}"
            )


def benchmark_cos(
    sizes: List[int], num_runs: int = 10
) -> Tuple[List[float], List[float]]:
    """Benchmark both simdly and numpy cos implementations."""
    simdly_times = []
    numpy_times = []

    print("\n--- Cosine Function Benchmark ---")

    for size in sizes:
        print(f"Benchmarking cos with {size} elements...")

        # Create test array with values in [0, 2π]
        a = np.random.uniform(0, 2 * math.pi, size).astype(np.float32)

        # Benchmark numpy.cos
        numpy_time = time_function(np.cos, a, num_runs=num_runs)
        numpy_times.append(numpy_time)

        if SIMDLY_AVAILABLE:
            # Benchmark simdly.cos
            simdly_time = time_function(simdly.cos, a, num_runs=num_runs)
            simdly_times.append(simdly_time)

            speedup = numpy_time / simdly_time if simdly_time > 0 else 0
            print(
                f"  Size {size}: NumPy: {numpy_time*1000:.3f}ms, SIMDLY: {simdly_time*1000:.3f}ms, Speedup: {speedup:.2f}x"
            )
        else:
            simdly_times.append(0)
            print(f"  Size {size}: NumPy: {numpy_time*1000:.3f}ms")

    return simdly_times, numpy_times


def benchmark_outer_product(
    sizes: List[int], num_runs: int = 10
) -> Tuple[List[float], List[float], List[float]]:
    """Benchmark simdly, parallel simdly, and numpy outer product implementations."""
    simdly_times = []
    par_simdly_times = []
    numpy_times = []

    print("\n--- Outer Product Benchmark ---")

    for size in sizes:
        print(f"Benchmarking outer product {size}x{size}...")

        # Create test arrays
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)

        # Benchmark numpy.outer
        numpy_time = time_function(np.outer, a, b, num_runs=num_runs)
        numpy_times.append(numpy_time)

        if SIMDLY_AVAILABLE:
            # Benchmark simdly.outer_product
            simdly_time = time_function(simdly.outer_product, a, b, num_runs=num_runs)
            simdly_times.append(simdly_time)

            # Benchmark simdly.par_outer_product
            par_simdly_time = time_function(
                simdly.par_outer_product, a, b, num_runs=num_runs
            )
            par_simdly_times.append(par_simdly_time)

            speedup = numpy_time / simdly_time if simdly_time > 0 else 0
            par_speedup = numpy_time / par_simdly_time if par_simdly_time > 0 else 0
            par_vs_seq = simdly_time / par_simdly_time if par_simdly_time > 0 else 0
            print(
                f"  Size {size}: NumPy: {numpy_time*1000:.3f}ms, SIMDLY: {simdly_time*1000:.3f}ms, Par-SIMDLY: {par_simdly_time*1000:.3f}ms"
            )
            print(
                f"    Speedups - vs NumPy: {speedup:.2f}x, Par vs NumPy: {par_speedup:.2f}x, Par vs Seq: {par_vs_seq:.2f}x"
            )
        else:
            simdly_times.append(0)
            par_simdly_times.append(0)
            print(f"  Size {size}: NumPy: {numpy_time*1000:.3f}ms")

    return simdly_times, par_simdly_times, numpy_times


def plot_benchmark_results(
    cos_sizes: List[int],
    outer_sizes: List[int],
    cos_results: Tuple[List[float], List[float]],
    outer_results: Tuple[List[float], List[float], List[float]],
):
    """Plot benchmark results for both cos and outer product."""
    if not SIMDLY_AVAILABLE:
        print("Skipping plot generation: simdly not available")
        return

    cos_simdly_times, cos_numpy_times = cos_results
    outer_simdly_times, par_outer_simdly_times, outer_numpy_times = outer_results

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "SIMDLY vs NumPy Performance Comparison", fontsize=16, fontweight="bold"
    )

    # Cosine benchmarks
    # Plot 1: Cosine execution times
    axes[0, 0].loglog(
        cos_sizes,
        [t * 1000 for t in cos_numpy_times],
        "b-o",
        label="NumPy",
        linewidth=2,
        markersize=6,
    )
    axes[0, 0].loglog(
        cos_sizes,
        [t * 1000 for t in cos_simdly_times],
        "r-s",
        label="SIMDLY",
        linewidth=2,
        markersize=6,
    )
    axes[0, 0].set_xlabel("Array Size")
    axes[0, 0].set_ylabel("Execution Time (ms)")
    axes[0, 0].set_title("Cosine Function Performance")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Cosine speedup
    axes[0, 1].semilogx(
        cos_sizes,
        [n / s if s > 0 else 0 for n, s in zip(cos_numpy_times, cos_simdly_times)],
        "g-^",
        linewidth=2,
        markersize=8,
    )
    axes[0, 1].axhline(y=1, color="k", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Array Size")
    axes[0, 1].set_ylabel("Speedup (NumPy time / SIMDLY time)")
    axes[0, 1].set_title("Cosine SIMDLY Speedup over NumPy")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cosine throughput
    axes[0, 2].loglog(
        cos_sizes,
        [size / t for size, t in zip(cos_sizes, cos_numpy_times)],
        "b-o",
        label="NumPy",
        linewidth=2,
        markersize=6,
    )
    axes[0, 2].loglog(
        cos_sizes,
        [size / t for size, t in zip(cos_sizes, cos_simdly_times)],
        "r-s",
        label="SIMDLY",
        linewidth=2,
        markersize=6,
    )
    axes[0, 2].set_xlabel("Array Size")
    axes[0, 2].set_ylabel("Throughput (elements/second)")
    axes[0, 2].set_title("Cosine Throughput Comparison")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Outer product benchmarks
    # Plot 4: Outer product execution times
    axes[1, 0].loglog(
        outer_sizes,
        [t * 1000 for t in outer_numpy_times],
        "b-o",
        label="NumPy",
        linewidth=2,
        markersize=6,
    )
    axes[1, 0].loglog(
        outer_sizes,
        [t * 1000 for t in outer_simdly_times],
        "r-s",
        label="SIMDLY",
        linewidth=2,
        markersize=6,
    )
    axes[1, 0].loglog(
        outer_sizes,
        [t * 1000 for t in par_outer_simdly_times],
        "g-^",
        label="Par-SIMDLY",
        linewidth=2,
        markersize=6,
    )
    axes[1, 0].set_xlabel("Vector Size (N for N×N matrix)")
    axes[1, 0].set_ylabel("Execution Time (ms)")
    axes[1, 0].set_title("Outer Product Performance")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Outer product speedup
    seq_speedups = [
        n / s if s > 0 else 0 for n, s in zip(outer_numpy_times, outer_simdly_times)
    ]
    par_speedups = [
        n / s if s > 0 else 0 for n, s in zip(outer_numpy_times, par_outer_simdly_times)
    ]
    axes[1, 1].semilogx(
        outer_sizes,
        seq_speedups,
        "r-s",
        linewidth=2,
        markersize=6,
        label="SIMDLY vs NumPy",
    )
    axes[1, 1].semilogx(
        outer_sizes,
        par_speedups,
        "g-^",
        linewidth=2,
        markersize=6,
        label="Par-SIMDLY vs NumPy",
    )
    axes[1, 1].axhline(y=1, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Vector Size (N for N×N matrix)")
    axes[1, 1].set_ylabel("Speedup over NumPy")
    axes[1, 1].set_title("Outer Product Speedup Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Outer product throughput
    axes[1, 2].loglog(
        outer_sizes,
        [size * size / t for size, t in zip(outer_sizes, outer_numpy_times)],
        "b-o",
        label="NumPy",
        linewidth=2,
        markersize=6,
    )
    axes[1, 2].loglog(
        outer_sizes,
        [size * size / t for size, t in zip(outer_sizes, outer_simdly_times)],
        "r-s",
        label="SIMDLY",
        linewidth=2,
        markersize=6,
    )
    axes[1, 2].loglog(
        outer_sizes,
        [size * size / t for size, t in zip(outer_sizes, par_outer_simdly_times)],
        "g-^",
        label="Par-SIMDLY",
        linewidth=2,
        markersize=6,
    )
    axes[1, 2].set_xlabel("Vector Size (N for N×N matrix)")
    axes[1, 2].set_ylabel("Throughput (matrix elements/second)")
    axes[1, 2].set_title("Outer Product Throughput Comparison")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simdly_vs_numpy_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_summary_tables(
    cos_sizes: List[int],
    cos_results: Tuple[List[float], List[float]],
    outer_sizes: List[int],
    outer_results: Tuple[List[float], List[float], List[float]],
):
    """Print formatted summary tables for both benchmarks."""
    cos_simdly_times, cos_numpy_times = cos_results
    outer_simdly_times, par_outer_simdly_times, outer_numpy_times = outer_results

    print("\n" + "=" * 80)
    print("COSINE FUNCTION BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Size':<8} {'NumPy (ms)':<12} {'SIMDLY (ms)':<13} {'Speedup':<8} {'Throughput Gain':<15}"
    )
    print("-" * 80)

    for i, size in enumerate(cos_sizes):
        if SIMDLY_AVAILABLE and cos_simdly_times[i] > 0:
            speedup = cos_numpy_times[i] / cos_simdly_times[i]
            throughput_gain = (
                speedup  # For element-wise operations, throughput gain equals speedup
            )
            print(
                f"{size:<8} {cos_numpy_times[i]*1000:<12.3f} {cos_simdly_times[i]*1000:<13.3f} "
                f"{speedup:<8.2f}x {throughput_gain:<15.2f}x"
            )
        else:
            print(
                f"{size:<8} {cos_numpy_times[i]*1000:<12.3f} {'N/A':<13} {'N/A':<8} {'N/A':<15}"
            )

    print("\n" + "=" * 120)
    print("OUTER PRODUCT BENCHMARK SUMMARY")
    print("=" * 120)
    print(
        f"{'Size':<8} {'NumPy (ms)':<12} {'SIMDLY (ms)':<13} {'Par-SIMDLY (ms)':<16} {'Seq Speedup':<12} {'Par Speedup':<12} {'Par vs Seq':<10}"
    )
    print("-" * 120)

    for i, size in enumerate(outer_sizes):
        if SIMDLY_AVAILABLE and outer_simdly_times[i] > 0:
            seq_speedup = outer_numpy_times[i] / outer_simdly_times[i]
            par_speedup = (
                outer_numpy_times[i] / par_outer_simdly_times[i]
                if par_outer_simdly_times[i] > 0
                else 0
            )
            par_vs_seq = (
                outer_simdly_times[i] / par_outer_simdly_times[i]
                if par_outer_simdly_times[i] > 0
                else 0
            )
            print(
                f"{size:<8} {outer_numpy_times[i]*1000:<12.3f} {outer_simdly_times[i]*1000:<13.3f} "
                f"{par_outer_simdly_times[i]*1000:<16.3f} {seq_speedup:<12.2f}x {par_speedup:<12.2f}x {par_vs_seq:<10.2f}x"
            )
        else:
            print(
                f"{size:<8} {outer_numpy_times[i]*1000:<12.3f} {'N/A':<13} {'N/A':<16} {'N/A':<12} {'N/A':<12} {'N/A':<10}"
            )


def main():
    """Main benchmark function."""
    print("=== SIMDLY vs NumPy Comprehensive Benchmark ===\n")

    # Verify correctness first
    verify_cos_correctness()
    print()
    verify_outer_correctness()
    print()

    if not SIMDLY_AVAILABLE:
        print("Skipping performance benchmark: simdly not available")
        return

    # Define test sizes
    cos_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    outer_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Run benchmarks
    print("Running comprehensive benchmarks...")
    cos_results = benchmark_cos(cos_sizes, num_runs=5)
    outer_results = benchmark_outer_product(outer_sizes, num_runs=5)

    # Print summaries with a single, correct function call
    print_summary_tables(cos_sizes, cos_results, outer_sizes, outer_results)

    # Plot results
    try:
        plot_benchmark_results(cos_sizes, outer_sizes, cos_results, outer_results)
        print(f"\nBenchmark plots saved as 'simdly_vs_numpy_benchmark.png'")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

    # Print analysis
    if SIMDLY_AVAILABLE:
        cos_simdly_times, cos_numpy_times = cos_results
        outer_simdly_times, par_outer_simdly_times, outer_numpy_times = outer_results

        # Calculate average speedups
        cos_speedups = [
            n / s for n, s in zip(cos_numpy_times, cos_simdly_times) if s > 0
        ]
        outer_speedups = [
            n / s for n, s in zip(outer_numpy_times, outer_simdly_times) if s > 0
        ]
        par_outer_speedups = [
            n / s for n, s in zip(outer_numpy_times, par_outer_simdly_times) if s > 0
        ]
        par_vs_seq_speedups = [
            s / p
            for s, p in zip(outer_simdly_times, par_outer_simdly_times)
            if p > 0 and s > 0
        ]

        print(f"\n=== PERFORMANCE ANALYSIS ===")
        if cos_speedups:
            print(f"Average cosine speedup: {sum(cos_speedups)/len(cos_speedups):.2f}x")
            print(f"Best cosine speedup: {max(cos_speedups):.2f}x")
        if outer_speedups:
            print(
                f"Average sequential outer product speedup: {sum(outer_speedups)/len(outer_speedups):.2f}x"
            )
            print(f"Best sequential outer product speedup: {max(outer_speedups):.2f}x")
        if par_outer_speedups:
            print(
                f"Average parallel outer product speedup: {sum(par_outer_speedups)/len(par_outer_speedups):.2f}x"
            )
            print(
                f"Best parallel outer product speedup: {max(par_outer_speedups):.2f}x"
            )
        if par_vs_seq_speedups:
            print(
                f"Average parallel vs sequential speedup: {sum(par_vs_seq_speedups)/len(par_vs_seq_speedups):.2f}x"
            )
            print(
                f"Best parallel vs sequential speedup: {max(par_vs_seq_speedups):.2f}x"
            )


if __name__ == "__main__":
    main()
